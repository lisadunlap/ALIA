import torch
import numpy as np
import torchvision
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import os
from PIL import Image 
import clip

def get_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    sorted_tuples = zip(*sorted(zip(values, counts))) # this just ensures we are getting the counts in the sorted order of the keys
    values, counts = [ list(tuple) for tuple in  sorted_tuples]
    fracs   = 1 / torch.Tensor(counts)
    return fracs / torch.max(fracs)

def is_monochromatic_image(img):
    extr = img.getextrema()
    a = 0
    for i in extr:
        if isinstance(i, tuple):
            a += abs(i[0] - i[1])
        else:
            a = abs(extr[0] - extr[1])
            break
    return a == 0

def filter_data(dataset, text, negative_text = ["a photo of an object", "a photo of a scene", "a photo of geometric shapes", "a photo", "an image"], threshold=0.9):
    """Filter out images that are not similar to the text prompt"""
    model, preprocess = clip.load("ViT-L/14", device="cuda")
    texts = clip.tokenize([text] + negative_text).to("cuda")

    # trainset = Cub2011Diffusion(root='/shared/lisabdunlap/data/txt2img/cub', subset=False, transform=preprocess)  
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    sim, ret = [], []
    with torch.no_grad():
        for images, labels, group, _ in loader:
            # imgs = torch.stack([preprocess(i).to("cuda") for i in images])
            image_features = model.encode_image(images.cuda())
            text_features = model.encode_text(texts)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            ret.append(torch.argmax(similarity, dim=1))
            sim.append(similarity)
    results = torch.cat(ret).cpu().numpy()
    sim = torch.cat(sim)
    idxs = np.where(results == 0)[0]
    print(f"Removing {len(dataset) - len(idxs)} ({idxs[:5]}) samples...")
    # if len(idxs:
    #     return sim, idxs, dataset
    return sim, np.where(results != 0)[0], Subset(dataset, idxs)


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.idx_to_dataset, self.idx_mapping, curr = [], [], 0 # maps index to what dataset it came from
        for j, d in enumerate(datasets):
            self.idx_to_dataset.extend([j] * len(d))
            self.idx_mapping.extend([i for i in range(len(d))])
            curr += len(d)
        
        assert len(datasets[0].classes) == len(datasets[1].classes) # only works for two datasets rn
        self.samples = np.concatenate([d.samples for d in datasets])
        self.samples = [(s[0], int(s[1])) for s in self.samples]
        print("samples new ",self.samples[:5])
        self.groups = np.concatenate([d.groups for d in datasets])
        self.targets = np.concatenate([[s[1] for s in d.samples] for d in datasets])
        self.class_weights = get_counts([s[1] for s in self.samples]) # returns class weight for XE
        # self.classes = set(sorted([i for i in d.classes for d in datasets]))
        self.classes = datasets[0].classes
        self.group_names = np.concatenate([d.group_names for d in datasets])
        self.class_names = datasets[0].class_names
        print(f"Combining datasets of size {[len(d) for d in datasets]} \t Total = {len(self.samples)}")

    def __getitem__(self, index):
        return self.datasets[self.idx_to_dataset[index]][self.idx_mapping[index]]

    def vis_dsets(self, idx, save=False):
        # plot a grid of images from each dataset at inedx x with their label as the caption
        fig, axs = plt.subplots(1, len(self.datasets), figsize=(20, 20), constrained_layout=True)
        imgs = []
        for i, d in enumerate(self.datasets):
            img, label, _, _ = d[idx]
            print(d.samples[idx][0])
            imgs.append(img)
            axs[i].imshow(img)
            axs[i].set_title(label)
            axs[i].axis('off')
        if save:
            plt.savefig(f"figs/vis_dsets_{idx}.png")
        return imgs

    def __len__(self):
        return len(self.samples)

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.samples = [self.dataset.samples[i] for i in indices]
        print(f"num samples {len(self.samples)}")
        self.groups = [self.dataset.groups[i] for i in indices]
        self.classes = self.dataset.classes
        self.group_names = self.dataset.group_names
        self.class_names = self.dataset.class_names
        self.targets = [s[1] for s in self.samples]
        self.class_weights = get_counts([s[1] for s in self.samples])

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

class BasicDataset(torchvision.datasets.ImageFolder):
    """
    Wrapper class for torchvision.datasets.ImageFolder.
    """
    def __init__(self, root, transform=None, group=0, cfg=None):
        self.group = group
        super().__init__(root, transform=transform)
        self.groups = [0] * len(self.samples)
        self.group_names = ["all"]
        if not cfg or not cfg.data.extra_classes:
            self.class_names = self.classes
            self.class_map = None
        else:
            self.class_names = list(cfg.data.extra_classes)
            assert [c in cfg.data.extra_classes for c in self.classes]
            self.class_map = [cfg.data.extra_classes.index(c) for c in self.classes]
            self.samples = [(s[0], self.class_map[s[1]]) for s in self.samples]
            self.classes = self.class_names
            print("reindex samples")
        # make sure we dont log the summary examples
        self.samples = [(s[0], int(s[1])) for s in self.samples if "samples" not in s[0]]
        self.targets = [s[1] for s in self.samples]
        self.class_weights = get_counts(self.targets)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, self.group, target
    
class EmbeddingDataset:
    """
    Returns precomputed clip embeddings for each image.
    """
    def __init__(self, root, dataset, split='train'):
        self.classes = dataset.classes
        self.class_names = dataset.class_names
        self.group_names = dataset.group_names
        # load embeddings
        if not os.path.exists(os.path.join(root, f"{split}_data.pt")):
            raise FileNotFoundError(f"Embeddings not found at {root}")
        self.data = torch.load(os.path.join(root, f"{split}_data.pt"))
        self.embeddings = self.data['clip_embeddings']
        self.embeddings /= self.embeddings.norm(dim=-1, keepdim=True)
        self.embeddings = self.embeddings.float()
        print("---------------------------------")
        print("embeddings size: ", self.embeddings.shape)
        print("---------------------------------")
        self.targets = self.data['labels'].numpy()
        self.targets
        self.groups = self.data['groups'].numpy()
        self.domains = self.data['domains'].numpy()
        self.samples = list(zip(self.embeddings, self.targets))
        self.class_weights = get_counts(self.targets)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.embeddings[index], self.targets[index], self.groups[index], self.domains[index]


class Img2ImgDataset(BasicDataset):
    """
    Wrapper class for torchvision.datasets.ImageFolder that randomly selects 1 gen
    image for each real image.
    """
    def __init__(self, root, transform=None, num_imgs=1, cfg=None, group=0):
        super().__init__(root, transform=transform, cfg=cfg, group=group)
        # np.random.seed(0)
        sample_groups = {}
        for i, s in enumerate(self.samples):
            filename = s[0].split("/")[-1] 
            if len(filename.split('.')[0].split("-")) == 1:
                print(f"skipping {filename}")
                continue
            else:
                idx, j = filename.split('.')[0].split("-")[0], filename.split('.')[0].split("-")[1]
            if idx not in sample_groups:
                sample_groups[idx] = [s]
            else:
                sample_groups[idx].append(s)
        self.sample_groups = sample_groups
        self.new_samples = []
        for k, v in self.sample_groups.items():
            # randomly select one sample from each group
            # chosen = np.random.choice(list(range(len(v))), num_imgs, replace=False)
            chosen = [0]
            # print(np.random.choice(list(range(len(v))), num_imgs, replace=False))
            # print([v[i] for i in chosen])
            self.new_samples += [(v[i][0], int(v[i][1])) for i in chosen]
        self.samples = self.new_samples
        self.targets = [s[1] for s in self.samples]
        self.groups = [0] * len(self.samples)
        self.group_names = [0]
        self.class_names = self.classes
        self.class_weights = get_counts([s[1] for s in self.samples])

    # def __getitem__(self, index):
    #     img, target = super().__getitem__(index)
    #     return img, target, self.groups[index], target
        

def subsample(dataset1, dataset2, attr='classes', seed=0):
    """
    Subsamples dataset2 to match the number of images and the
    number of images in each class of dataset1.
    """
    np.random.seed(seed)
    classes = getattr(dataset1, attr)
    # print(f"1 = {np.unique(np.array(dataset1.targets))} 2 = {np.unique(np.array(dataset2.targets))}")
    class_counts = dict(Counter((dataset1.targets)))
    class_counts2 = dict(Counter(dataset2.targets))
    # print("class counts ", class_counts)
    # print("class counts 2 ", class_counts2)
    indices = []
    for c in class_counts.keys():
        replace = False
        idx_filtered = [i for i, t in enumerate(dataset2.targets) if t == c]
        if len(idx_filtered) < class_counts[c]:
            print(f"Could only get {len(idx_filtered)} samples instead of {class_counts[c]} for class {c}.")
            replace = True
        else:
            print(f"Getting {class_counts[c]} samples for class {c}.")
        if len(idx_filtered) == 0:
            continue
        indices.extend(np.random.choice([i for i, t in enumerate(dataset2.targets) if t == c], class_counts[c], replace=replace))
    return Subset(dataset2, indices)

def get_class_balanced_subset(dataset, k=5):
    """
    Given a dataset, returns a subset of size k for each class.
    """
    class_counts = dict(Counter((dataset.targets)))
    indices = []
    for c in class_counts.keys():
        idx_filtered = [i for i, t in enumerate(dataset.targets) if t == c]
        if len(idx_filtered) < k:
            print(f"Could only get {len(idx_filtered)} samples instead of {k} for class {c}.")
        else:
            print(f"Getting {k} samples for class {c}.")
        if len(idx_filtered) == 0:
            continue
        indices.extend(np.random.choice([i for i, t in enumerate(dataset.targets) if t == c], k, replace=False))
    return Subset(dataset, indices)

# PLOTTING UTILS
def select_random_img(dataset, class_idx=0, class_name=None, n=1, show=True):
    """
    Selects a random image from a dataset.
    """
    if class_name is not None:
        class_idx = dataset.classes.index(class_name)
    class_imgs = [i for i, l in enumerate(dataset.targets) if l == class_idx]
    idxs = np.random.choice(class_imgs, n)
    print(f"Returning samples for class {class_idx} ({idxs}).")
    samples = [dataset[i] for i in idxs]
    if show:
        plot_imgs(samples, n)
    return samples

def plot_imgs(samples, n=1):
    """
    Plots a grid of images.
    """
    fig = plt.figure(figsize=(n*5, 5))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, n),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, samples):
        # Iterating over the grid returns the Axes.
        ax.imshow(im[0])
        ax.axis('off')
    plt.show()