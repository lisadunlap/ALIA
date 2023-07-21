import pandas as pd
from PIL import Image
import os
import numpy as np
import torch
import torchvision.datasets as dsets

def get_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    sorted_tuples = zip(*sorted(zip(values, counts))) # this just ensures we are getting the counts in the sorted order of the keys
    values, counts = [ list(tuple) for tuple in  sorted_tuples]
    fracs   = 1 / torch.Tensor(counts)
    return fracs / torch.max(fracs)

class Waterbirds:
    def __init__(self, root, split, transform=None):
        self.root = root
        self.df = pd.read_csv(os.path.join(root, 'metadata.csv'))
        if split == 'train':
            self.df = self.df[self.df.split == 0]
        elif split == 'val':
            self.df = self.df[self.df.split == 1]
        elif split == 'test':
            self.df = self.df[self.df.split == 2]
        self.class_names = [f.split('/')[0] for f in self.df['img_filename'].unique()]
        self.group_names = ['land_landbird', 'land_waterbird', 'water_landbird', 'water_waterbird']
        self.samples =[(f,y) for f,y in zip(self.df['img_filename'], self.df['y'])]
        self.targets = [s[1] for s in self.samples]
        self.classes = ['land', 'water']
        self.transform = transform
        self.class_weights = get_counts(self.df['y']) # returns class weight for XE
        self.groups = [] # create group labels
        for (label, place) in zip(self.df['y'], self.df['place']):
            if place == 0:
                group = 0 if label == 0 else 1
            else:
                group = 2 if label == 0 else 3
            self.groups.append(group)
        self.group_weights = get_counts(self.groups) # returns group weight for XE
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root, sample['img_filename'])).convert('RGB')
        label = int(sample['y'])
        place = int(sample['place'])
        group = self.groups[idx]
        species = sample['img_filename'].split('/')[0]
        if self.transform:
            img = self.transform(img)
        return img, label, group

    def get_subset(self, groups=[0,1,2,3], num_per_class=5):
        self.df['group'] = self.groups
        df = self.df.reset_index(drop=True)
        df['orig_idx'] = df.index
        df['class'] = [f.split('/')[0] for f in df.img_filename]
        df = df[df.place.isin(groups)]
        return df.groupby('class').apply(lambda x: x.sample(n=num_per_class) if len(x) > num_per_class else x.sample(n=len(x))).reset_index(drop=True)['orig_idx'].values

class WaterbirdsInverted(dsets.ImageFolder):
    """ Dataset for the Waterbirds dataset generated with textual inversion."""

    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.group_names = ['land_landbird', 'land_waterbird', 'water_landbird', 'water_waterbird']
        # self.classes = ['land', 'water']
        df = pd.read_csv("/shared/lisabdunlap/vl-attention/data/waterbird_complete95_forest2water2/metadata.csv")
        df['class'] = df['img_filename'].apply(lambda x: x.split('/')[0])
        class_map = df.drop_duplicates(subset=['class'])[['class', 'y']].values
        class_map = {c[0]: c[1] for c in class_map} # mapping bird species to land or water
        class_to_idx = [class_map[c] for c in self.classes]
        self.old_labels = [s[1] for s in self.samples]
        self.species = self.classes
        self.classes = ['land', 'water']
        self.samples = [(f, class_to_idx[class_idx]) for f, class_idx in self.samples]
        self.targets = [s[1] for s in self.samples]
        self.groups, self.places = [], [] # create group labels
        for (filename, label) in self.samples:
            if label == 0:
                self.groups.append(1)
                self.places.append(1)
            else:
                self.groups.append(2)
                self.places.append(0)
        self.group_weights = get_counts(self.groups) # returns group weight for XE
        self.class_weights = get_counts([s[1] for s in self.samples]) # returns class weight for XE

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        group = self.groups[idx]
        place = self.places[idx]
        species = self.species[self.old_labels[idx]]
        return img, label, group