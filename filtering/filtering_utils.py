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

def get_clip_features(model, loader, device='cuda'):
    model.eval()
    all_features = []
    all_labels = []
    all_groups, all_domains = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            images, labels, groups, domains = batch
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)
            all_groups.append(groups)
            all_domains.append(domains)

    return torch.cat(all_features).cpu(), torch.cat(all_labels).cpu(), torch.cat(all_groups).cpu(), torch.cat(all_domains).cpu()

def get_features(model, loader, device='cuda'):
    ## Get the feautres from the penultimate layer
    features = []
    logits = []
    def forward_hook(module, input, output):
        features.append(input)
    model.module.fc.register_forward_hook(forward_hook)
    with torch.no_grad():
        for batch_idx, (inputs, targets, _, _) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            logits.append(probs)
    model.module.fc._forward_hooks.clear()
    return features, logits


# def filter_data(dataset, text, negative_text = ["a photo of an object", "a photo of a scene", "a photo of geometric shapes", "a photo", "an image"], threshold=0.9):
#     """Filter out images that are not similar to the text prompt"""
#     model, preprocess = clip.load("ViT-L/14", device="cuda")
#     texts = clip.tokenize([text] + negative_text).to("cuda")
#     loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
#     text_features = model.encode_text(texts)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#     ret = []
#     embeddings = []
#     with torch.no_grad():
#         for images, labels, _, _ in tqdm(loader):
#             image_features = model.encode_image(images.cuda())
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             embeddings += [image_features]
#             similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#             ret.append(similarity)
#     results = torch.cat(ret)
#     predictions = torch.argmax(results, dim=1).cpu().numpy()
#     idxs = np.where(predictions != 0)[0]
#     remove_idxs = np.where(predictions == 0)[0]
#     return predictions, remove_idxs, Subset(dataset, idxs), torch.cat(embeddings)

# def emb_pairs(edit_filenames, embeddings_orig, embeddings_edit):
#     """
#     Returns a list of tuples of (original, edited) image embeddings
#     """
#     ret = []
#     idx_pairings = []
#     for i, filename in enumerate(edit_filenames):
#         idx = int(filename.split("/")[-1].split('-')[0])
#         idx_pairings.append(idx)
#         ret.append(torch.stack((embeddings_orig[idx], embeddings_edit[i])))
#     return idx_pairings, torch.stack(ret)

# def plot_imgs(images, captions, class_labels, n_rows=1, save_path=None):
#     n_cols = len(images) // n_rows
#     fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
#     grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                     nrows_ncols=(n_rows, n_cols),  # creates 2x2 grid of axes
#                     axes_pad=0.25,  # pad between axes in inch.
#                     )

#     for ax, im, cap in zip(grid, images, captions):
#         # Iterating over the grid returns the Axes.
#         ax.imshow(im.resize((224, 224)))
#         ax.set_title(class_labels[cap], fontsize=20)
#         ax.axis('off')

#     if save_path is not None:
#         plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

# # compare to random sample of each class
# def compare_class_edits(edit, features, labels, k=10):
#     """
#     Get the similarity between the edit and the k nearest neighbors of the class
#     """
#     edit, features = torch.Tensor(edit), torch.Tensor(features)
#     similarities = []
#     for l in sorted(np.unique(labels)):
#         perm = torch.randperm(features[labels == l].size(0))
#         sample = features[labels == l][perm[:k]]
#         similarities.append(torch.max(edit.view(1, -1) @ sample.T).item())
#     return similarities, np.argmax(similarities)

# def image_based_filter(edit_features, edit_labels, features, labels, k=100):
#     """
#     Filter the edits based on the similarity to a random sample of training
#     images from that class
#     """
#     nns = []
#     for i in range(len(edit_features)):
#         sim, nn = compare_class_edits(edit_features[i], features, labels, k=k)
#         nns.append(nn)
#     nns = np.array(nns)
#     return nns, [n for i, n in enumerate(nns) if n != edit_labels[i]], [n for i, n in enumerate(nns) if n == edit_labels[i]]