import os
import clip
import torch
import torchvision

import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import random
from mpl_toolkits.axes_grid1 import ImageGrid
import os
from torchvision.datasets.utils import download_url
from datasets.base import get_counts

CUB_DOMAINS = ["photo", "painting"]

INVERTED_CLASSES = [8, 20, 22, 24, 25, 28, 41, 42, 50, 56, 58, 62, 64, 67, 71, 81, 82, 85, 87, 88, 89, 97, 100, 103, 104, 112, 113, 123, 125, 127, 130, 131, 133, 134, 135, 140, 142, 143, 145, 158, 161, 162, 163, 168, 169, 175, 178, 187, 191, 196]

class Cub2011(torch.utils.data.Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'


    def __init__(self, root, split='train', subset=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.group_names = CUB_DOMAINS

        with open(f'{root}/CUB-200-Painting/classes.txt') as f:
            lines = f.readlines()
        self.classes = [l.replace('\n', '').split('.')[-1].replace('_', ' ') for l in lines]
        self._load_metadata()
        # if download:
        #     self._download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')
        if self.split != 'val':
            new_df = []
            for c in self.data.target.unique():
                if subset:
                    if self.split == 'train' and c not in INVERTED_CLASSES:
                        new_df.append(self.data[(self.data.is_training_img == 1) & (self.data.target == c)])
                    elif self.split == 'train':
                        new_df.append(self.data[(self.data.is_training_img == 1) & (self.data.target == c)][:-20])
                    elif self.split == 'extra'and c in INVERTED_CLASSES:
                        new_df.append(self.data[(self.data.is_training_img == 1) & (self.data.target == c)][-20:])
                else:
                    if self.split == 'train':
                        new_df.append(self.data[(self.data.is_training_img == 1) & (self.data.target == c)][:-5])
                    elif self.split == 'extra':
                        new_df.append(self.data[(self.data.is_training_img == 1) & (self.data.target == c)][-5:])
            self.data = pd.concat(new_df)
        self.samples = [(os.path.join(self.root, 'CUB_200_2011/images', f), t-1) for f, t in zip(self.data.filepath, self.data.target)]
        self.targets = [s[1] for s in self.samples]
        self.groups = [0] * len(self.samples)
        self.class_weights = get_counts(self.targets)
        self.class_names = self.data.species.unique()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        self.data['species'] = self.data['filepath'].apply(lambda x: x.split('/')[0])

        if self.split == 'val':
            self.data = self.data[self.data.is_training_img == 0]
        else:
            self.data = self.data[self.data.is_training_img == 1]
    
    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img =  Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # return img, target
        return img, target, 0, target
    
class newCub2011(Cub2011):

    def __init__(self, root, split='train', transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.group_names = CUB_DOMAINS

        with open(f'{root}/CUB-200-Painting/classes.txt') as f:
            lines = f.readlines()
        self.classes = [l.replace('\n', '').split('.')[-1].replace('_', ' ') for l in lines]
        self._load_metadata()

        if self.split != 'val':
            new_df = []
            for c in self.data.target.unique():
                if self.split == 'train':
                    new_df.append(self.data[(self.data.is_training_img == 1) & (self.data.target == c)][:-15])
                elif self.split == 'extra':
                    new_df.append(self.data[(self.data.is_training_img == 1) & (self.data.target == c)][-15:])
            self.data = pd.concat(new_df)
        self.samples = [(os.path.join(self.root, 'CUB_200_2011/images', f), t-1) for f, t in zip(self.data.filepath, self.data.target)]
        self.targets = [s[1] for s in self.samples]
        self.groups = [0] * len(self.samples)
        self.class_weights = get_counts(self.targets)
        self.class_names = self.data.species.unique()
    
class Cub2011Seg(Cub2011):

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img =  Image.open(path).convert('RGB')
        seg = Image.open(path.replace('images', 'segmentations').replace('.jpg', '.png'))

        if self.transform is not None:
            img = self.transform(img)

        # return img, target
        return img, target, 0, seg

class Cub2011Painting(torchvision.datasets.ImageFolder):

    def __init__(self, root, subset=True, transform=None):
        super().__init__(root, transform=transform)
        self.targets = [s[1] for s in self.samples]
        self.class_weights = get_counts(self.targets)

    def __getitem__(self, idx):
        path, target = self.samples[idx]

        img =  Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, 1, target

class Cub2011Diffusion(torchvision.datasets.ImageFolder):

    def __init__(self, root, subset=False, transform=None):
        super().__init__(root, transform=transform)
        self.classes = CUB_CLASSES
        self.group_names = CUB_DOMAINS
        self.groups = [0] * len(self.samples)
        self.class_names = self.classes
        self.subset = subset
        self.targets = [s[1] for s in self.samples]
        self.class_weights = get_counts(self.targets)

    def __getitem__(self, idx):
        path, target = self.samples[idx]

        img =  Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.subset:
            if target in INVERTED_CLASSES:
                target = INVERTED_CLASSES[target] - 1
        return img, target, 0, target