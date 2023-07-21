import torch
import torchvision
import torchvision.transforms as transforms
from omegaconf import OmegaConf

import os
import argparse
import wandb
import clip
import numpy as np
import collections 
import random
from tqdm import tqdm
from PIL import Image
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import datasets
import models
from utils import nest_dict, read_unknowns, flatten_config
# from filtering.filtering_utils import get_clip_features, get_features, load_checkpoint
from cleanlab.count import get_confident_thresholds
from datasets.base import CombinedDataset
from helpers.load_dataset import get_train_transform, get_dataset

parser = argparse.ArgumentParser(description='Dataset Understanding')
parser.add_argument('--config', default='configs/base.yaml', help="config file")
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
flags, unknown = parser.parse_known_args()

overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base      = OmegaConf.load('configs/base.yaml')
args      = OmegaConf.merge(base, cfg, overrides)
if len(unknown) > 0:
    print(unknown)
    config = nest_dict(read_unknowns(unknown))
    to_merge = OmegaConf.create(config)
    args = OmegaConf.merge(args, to_merge)
args.yaml = flags.config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.wandb_silent:
    os.environ['WANDB_SILENT']="true"

if args.test:
    print("-------------------------------------------------------------")
    print("------------------------- TEST MODE -------------------------")
    print("-------------------------------------------------------------")
    run = wandb.init(project="CleanLab", name='test', group=args.name, config=flatten_config(args))
else:
    run = wandb.init(project="CleanLab", group=args.name, config=flatten_config(args))

torch.manual_seed(args.seed)
np.random.seed(args.seed)

def load_checkpoint(args, net):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.filter.checkpoint_name:
        checkpoint_name = f'./checkpoint/{args.filter.checkpoint_name}'
    checkpoint = torch.load(checkpoint_name)

    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['net'].items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v
    
    print(f"Loaded checkpoint at epoch {checkpoint['epoch']} from {checkpoint_name}")
    # net.load_state_dict(checkpoint['net'])
    net.load_state_dict(new_state_dict)

    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return net, best_acc, start_epoch

def get_aug_dataset(args, transform):
    if type(args.data.extra_root) != str:
        print("Roots", list(args.data.extra_root))
        roots, dsets = list(args.data.extra_root), []
        for i, r in enumerate(roots):
            dsets.append(getattr(datasets.base, args.data.extra_dataset)(r, transform=transform, cfg=args, group=i))
        dataset = CombinedDataset(dsets)
    else:
        dataset = getattr(datasets.base, args.data.extra_dataset)(args.data.extra_root, transform=transform, cfg=args)
        print("DATASET SIZE", len(dataset))
    return dataset


# load clip model
clip_model, clip_transform = clip.load(args.filter.model, device="cuda")

# Data
print('==> Preparing data..')
transform = get_train_transform(args.data.base_dataset, model=args.model, augmentation=args.data.augmentation)
if 'Extra' in args.data.base_dataset:
    trainset, valset, testset, dataset = get_dataset(args.data.base_dataset, transform=transform, val_transform=transform, root=args.data.base_root)
else:
    dataset = get_aug_dataset(args, transform)
    clip_dataset = get_aug_dataset(args, clip_transform)
    trainset, valset, testset, _ = get_dataset(args.data.base_dataset, transform=transform, val_transform=transform, root=args.data.base_root)
    clip_trainset, clip_valset, clip_testset, _ = get_dataset(args.data.base_dataset, transform=clip_transform, val_transform=transform, root=args.data.base_root)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.data.batch, shuffle=False, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.data.batch, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.data.batch, shuffle=False, num_workers=2)
augloader = torch.utils.data.DataLoader(dataset, batch_size=args.data.batch, shuffle=False, num_workers=2)

clip_trainloader = torch.utils.data.DataLoader(clip_trainset, batch_size=args.data.batch, shuffle=False, num_workers=2)
clip_valloader = torch.utils.data.DataLoader(clip_valset, batch_size=args.data.batch, shuffle=False, num_workers=2)
clip_testloader = torch.utils.data.DataLoader(clip_testset, batch_size=args.data.batch, shuffle=False, num_workers=2)
clip_augloader = torch.utils.data.DataLoader(clip_dataset, batch_size=args.data.batch, shuffle=False, num_workers=2)

def get_clip_features(model, loader):
    model.eval()
    all_features = []
    all_labels = []
    all_groups = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            images, labels, groups = batch
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)
            all_groups.append(groups)

    return torch.cat(all_features).cpu(), torch.cat(all_labels).cpu(), torch.cat(all_groups).cpu()

# get cosine similarity per class between dataset and base
def get_cosine_similarity(embeddings, labels, base_embeddings, base_labels):
    # get cosine similarity per class between dataset and base
    # normalize
    embeddings /= embeddings.norm(dim=-1, keepdim=True)
    base_embeddings /= base_embeddings.norm(dim=-1, keepdim=True)
    cosine_sim = []
    for i in np.unique(labels):
        class_embeddings = embeddings[labels == i]
        class_base_embeddings = base_embeddings[base_labels == i]
        # get cos sim to nn in training set
        cosine_sim_cls = []
        for emb in class_embeddings:
            cos_sim = torch.unsqueeze(emb, 0) @ class_base_embeddings.T
            cosine_sim_cls.append(torch.max(cos_sim, dim=1)[0])
        cosine_sim.append(torch.stack(cosine_sim_cls))
    return cosine_sim

def semantic_filter(dataset, text, negative_text = ["a photo of an object", "a photo of a scene", "a photo of geometric shapes", "a photo", "an image"], threshold=0.9):
    """Filter out images that are not similar to the text prompt"""
    model, preprocess = clip.load("ViT-L/14", device="cuda")
    text = [text] if type(text) == str else text
    texts = clip.tokenize(text + negative_text).to("cuda")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.data.batch, shuffle=False, num_workers=4)
    text_features = model.encode_text(texts)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    ret = []
    embeddings = []
    with torch.no_grad():
        for images, labels, _, _ in tqdm(loader):
            image_features = model.encode_image(images.cuda())
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embeddings += [image_features]
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            ret.append(similarity)
    results = torch.cat(ret)
    predictions = torch.argmax(results, dim=1).cpu().numpy()
    idxs = [p for p in range(len(predictions)) if predictions[p] in list(range(len(text)))]
    remove_idxs = [p for p in range(len(predictions)) if predictions[p] not in list(range(len(text)))]
    return predictions, remove_idxs, idxs, torch.cat(embeddings)

def semantic_filter_saved(embeddings, text, negative_text):
    """Filter out images that are not similar to the text prompt"""
    model, preprocess = clip.load("ViT-L/14", device="cuda")
    text = [text] if type(text) == str else text
    texts = clip.tokenize(text + negative_text).to("cuda")
    text_features = model.encode_text(texts)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        image_features = embeddings
        print(image_features.shape, text_features.shape)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predictions = torch.argmax(similarity, dim=1).cpu().numpy()
        idxs = [p for p in range(len(predictions)) if predictions[p] in list(range(len(text)))]
        remove_idxs = [p for p in range(len(predictions)) if predictions[p] not in list(range(len(text)))]
    return predictions, remove_idxs, idxs

# Load Model
model = getattr(models, args.model)(num_classes = len(trainset.classes))
model = torch.nn.DataParallel(model)
model = model.to(device)
net, best_acc, start_epoch = load_checkpoint(args, model)
model.eval()

def plot_imgs(images, captions, n_rows=1, save_path=None):
    n_cols = len(images) // n_rows
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(n_rows, n_cols),  # creates 2x2 grid of axes
                    axes_pad=0.25,  # pad between axes in inch.
                    )

    for ax, im, cap in zip(grid, images, captions):
        # Iterating over the grid returns the Axes.
        ax.imshow(im.resize((224, 224)))
        ax.set_title(dataset.classes[cap], fontsize=20)
        ax.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_features(model, loader):
    ## Get the feautres from the penultimate layer
    features = []
    logits = []
    def forward_hook(module, input, output):
        features.append(input)
    model.module.fc.register_forward_hook(forward_hook)
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            logits.append(probs)
    model.module.fc._forward_hooks.clear()
    return features, logits

if not args.filter.load:
    # compute clip embeddings
    train_emb, train_labels, train_groups = get_clip_features(clip_model, clip_trainloader)
    aug_emb, aug_labels, aug_groups = get_clip_features(clip_model, clip_augloader)
    val_emb, val_labels, val_groups = get_clip_features(clip_model, clip_valloader)
    test_emb, test_labels, test_groups = get_clip_features(clip_model, clip_testloader)

    # get model features and logits
    features, logits = get_features(model, augloader)
    features = [torch.cat(features[i]).cuda() for i in range(len(features))]
    features, logits = torch.cat(features, dim=0), torch.cat(logits, dim=0)

    train_features, train_logits = get_features(model, trainloader)
    train_features = [torch.cat(train_features[i]).cuda() for i in range(len(train_features))]
    train_features, train_logits = torch.cat(train_features, dim=0), torch.cat(train_logits, dim=0)

    val_features, val_logits = get_features(model, valloader)
    val_features = [torch.cat(val_features[i]).cuda() for i in range(len(val_features))]
    val_features, val_logits = torch.cat(val_features, dim=0), torch.cat(val_logits, dim=0)

    test_features, test_logits = get_features(model, testloader)
    test_features = [torch.cat(test_features[i]).cuda() for i in range(len(test_features))]
    test_features, test_logits = torch.cat(test_features, dim=0), torch.cat(test_logits, dim=0)

    if not args.test:
        if not os.path.exists(f"{args.filter.save_dir}/{args.name}"):
            os.makedirs(f"{args.filter.save_dir}/{args.name}")
            os.makedirs(f"{args.filter.save_dir}/{args.name}/samples")
            os.makedirs(f"{args.filter.save_dir}/{args.name}/filtered_idxs")
        
        if not os.path.exists(f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}"):
            os.makedirs(f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}")
        print("Saving predictions...")
        train_data = {"clip_embeddings": train_emb, "labels": train_labels , "logits": train_logits, "features": train_features, "groups": train_groups}
        aug_data = {"clip_embeddings": aug_emb, "labels": aug_labels, "logits": logits, "features": features, "groups": aug_groups}
        val_data = {"clip_embeddings": val_emb, "labels": val_labels, "logits": val_logits, "features": val_features, "groups": val_groups}
        test_data = {"clip_embeddings": test_emb, "labels": test_labels, "logits": test_logits, "features": test_features, "groups": test_groups}
        torch.save(train_data, f"{args.data.embedding_root}/{args.data.base_dataset}/train_data.pt")
        torch.save(aug_data, f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}/train_data.pt")
        torch.save(val_data, f"{args.data.embedding_root}/{args.data.base_dataset}/val_data.pt")
        torch.save(test_data, f"{args.data.embedding_root}/{args.data.base_dataset}/test_data.pt")
else:
    train_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/train_data.pt")
    aug_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/{args.name}/train_data.pt")
    val_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/val_data.pt")
    test_data = torch.load(f"{args.data.embedding_root}/{args.data.base_dataset}/test_data.pt")
    train_emb, train_labels, train_logits, train_features = train_data["clip_embeddings"], train_data["labels"], train_data["logits"], train_data["features"]
    aug_emb, aug_labels, logits, features = aug_data["clip_embeddings"], aug_data["labels"], aug_data["logits"], aug_data["features"]
    val_emb, val_labels, val_logits, val_features = val_data["clip_embeddings"], val_data["labels"], val_data["logits"], val_data["features"]
    test_emb, test_labels, test_logits, test_features = test_data["clip_embeddings"], test_data["labels"], test_data["logits"], test_data["features"]

train_emb, aug_emb, val_emb, test_emb = train_emb.cuda(), aug_emb.cuda(), val_emb.cuda(), test_emb.cuda()

aug_labels, base_labels = [int(s[1]) for s in dataset.samples], [int(s[1]) for s in trainset.samples]

##############################################################################################################
################################ 1. Find label errors #######################################################
##############################################################################################################
def get_pred_and_conf(data):
    logits = data['logits']
    pred = logits.argmax(dim=1)
    return pred.cpu().numpy(), logits.max(dim=1)[0].cpu().numpy()

preds, conf = get_pred_and_conf(train_data)
conf_thresh = get_confident_thresholds(train_data['labels'].numpy(), train_data['logits'].cpu().numpy())
print('conf_thresh', conf_thresh)
aug_preds, aug_conf = get_pred_and_conf(aug_data)
conf_correct_idxs = np.where((aug_conf > conf_thresh[aug_preds]) & (aug_preds == aug_data['labels'].numpy()))[0]
conf_incorrect_idxs = np.where((aug_conf > conf_thresh[aug_preds]) & (aug_preds != aug_data['labels'].numpy()))[0]

print(f"Number of too easy examples: {len(conf_correct_idxs)}")
filtered_labels = [dataset.samples[i][1] for i in conf_correct_idxs]

if not os.path.exists(f"{args.filter.save_dir}/{args.name}/easy/samples/"):
    os.makedirs(f"{args.filter.save_dir}/{args.name}/easy/samples/")

for i in np.unique(filtered_labels):
    print(f"{dataset.classes[i]}: {filtered_labels.count(i)}/{list(aug_data['labels']).count(i)}")
    label_samples = conf_correct_idxs[np.where(filtered_labels == i)]
    vis_samples = np.random.choice(label_samples, size=min([10, len(label_samples)]), replace=False)
    samples = [dataset.samples[i] for i in vis_samples]
    images = [Image.open(sample[0]) for sample in samples]
    captions = [sample[1] for sample in samples]
    plot_imgs(images, captions, n_rows=1, save_path=f"{args.filter.save_dir}/{args.name}/easy/samples/easy_filtered-{i}.png")
    images = wandb.Image(Image.open(f"{args.filter.save_dir}/{args.name}/easy/samples/easy_filtered-{i}.png"), caption="cleanlab")
    wandb.log({f"Easy examples filtered {dataset.classes[i]}": images})

np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/too_easy_filtered.npy", conf_correct_idxs)
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/too_easy_kept.npy", [kept for kept in range(len(dataset)) if kept not in conf_correct_idxs])
wandb.summary[f"Easy filtered"] = len(conf_correct_idxs)
wandb.summary[f"Easy kept"] = len(dataset) - len(conf_correct_idxs)

print(f"Number of possibly mislabled examples: {len(conf_incorrect_idxs)}")
filtered_labels = [dataset.samples[i][1] for i in conf_incorrect_idxs]

if not os.path.exists(f"{args.filter.save_dir}/{args.name}/mislabled/samples/"):
    os.makedirs(f"{args.filter.save_dir}/{args.name}/mislabled/samples/")

for i in np.unique(filtered_labels):
    print(f"{dataset.classes[i]}: {filtered_labels.count(i)}/{list(aug_data['labels']).count(i)}")
    label_samples = conf_incorrect_idxs[np.where(filtered_labels == i)]
    vis_samples = np.random.choice(label_samples, size=min([10, len(label_samples)]), replace=False)
    samples = [dataset.samples[i] for i in vis_samples]
    images = [Image.open(sample[0]) for sample in samples]
    captions = [sample[1] for sample in samples]
    plot_imgs(images, captions, n_rows=1, save_path=f"{args.filter.save_dir}/{args.name}/mislabled/samples/mislabled_filtered-{i}.png")
    images = wandb.Image(Image.open(f"{args.filter.save_dir}/{args.name}/mislabled/samples/mislabled_filtered-{i}.png"), caption="cleanlab")
    wandb.log({f"Easy examples filtered {dataset.classes[i]}": images})

np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/mislabled_filtered.npy", conf_incorrect_idxs)
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/mislabled_kept.npy", [kept for kept in range(len(dataset)) if kept not in conf_incorrect_idxs])
wandb.summary[f"Mislabeled filtered"] = len(conf_incorrect_idxs)
wandb.summary[f"Mislabeled kept"] = len(dataset) - len(conf_incorrect_idxs)

##############################################################################################################
################################ 2. Find semantic errors ####################################################
##############################################################################################################

predictions, semantic_filtered, kept_idxs = semantic_filter_saved(aug_emb, args.filter.prompt, list(args.filter.negative_prompts))
base_predictions, base_idxs, _ = semantic_filter_saved(train_emb, args.filter.prompt, list(args.filter.negative_prompts))
wandb.summary["semantic filter removed"] = len(semantic_filtered)
wandb.summary["semantic filter kept"] = len(kept_idxs)
wandb.summary["semantic filter removed ratio"] = len(semantic_filtered) / len(aug_emb)
wandb.summary['semantic filter removed original'] = len(base_idxs)
wandb.summary["semantic filter removed ratio original"] = len(base_idxs) / len(train_emb)
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/semantic_filtered.npy", semantic_filtered)
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/semantic_kept.npy", kept_idxs)


filtered = np.unique(np.concatenate((semantic_filtered, conf_incorrect_idxs, conf_correct_idxs)))
kept = np.array([i for i in range(len(aug_emb)) if i not in filtered])
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/filtered.npy", filtered)
np.save(f"{args.filter.save_dir}/{args.name}/filtered_idxs/kept.npy", kept)
print(f"Total filter removed {len(filtered)}/{len(aug_emb)} images")
wandb.summary["total filter removed"] = len(filtered)
wandb.summary["total filter removed ratio"] = len(filtered) / len(aug_emb)

aug_labels = np.array(aug_labels)

# get num filtered by class counts
semantic_filtered_counts = collections.Counter([l for i, l in enumerate(aug_labels) if i in semantic_filtered])
easy_filtered_counts = collections.Counter([l for i, l in enumerate(aug_labels) if i in conf_correct_idxs])
mislabled_filtered_counts = collections.Counter([l for i, l in enumerate(aug_labels) if i in conf_incorrect_idxs])
filtered_counts = collections.Counter([l for i, l in enumerate(aug_labels) if i in filtered])
# put into dataframe of label, semantic_filtered, nn_filtered, filtered, total
counts = []
for l in range(len(dataset.classes)):
    counts.append([dataset.classes[l], semantic_filtered_counts[l], easy_filtered_counts[l], mislabled_filtered_counts[l], filtered_counts[l], len(aug_labels[aug_labels == l])])
counts = pd.DataFrame(counts, columns=["label", "semantic_filtered", "easy_filtered","mislabeled_filtered", "filtered", "total"])
counts.to_csv(f"{args.filter.save_dir}/{args.name}/label_predictions.csv")
label_table = wandb.Table(dataframe=counts[counts['filtered'] > 0])
wandb.log({"Label predictions": label_table})

edit_filenames = [dataset.samples[i][0] for i in range(len(dataset))]
# plot random sample of filtered vs original dataset
filtered_sample = np.random.choice(filtered, 10)
print(len(edit_filenames), filtered_sample)
filtered_vis = [Image.open(edit_filenames[i]) for i in filtered_sample]
filtered_captions = [aug_labels[i] for i in filtered_sample]
unfiltered_sample = np.random.choice(kept, 10)
unfiltered_vis = [Image.open(edit_filenames[i]) for i in unfiltered_sample]
unfiltered_captions = [aug_labels[i] for i in unfiltered_sample]
plot_imgs(filtered_vis+unfiltered_vis, filtered_captions+unfiltered_captions, n_rows=2, save_path=f"{args.filter.save_dir}/{args.name}/samples/filtered_vs_unfiltered.png")
