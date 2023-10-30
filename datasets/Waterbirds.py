import pandas as pd
from PIL import Image
import os
import numpy as np
import torch
import torchvision.datasets as dsets
from collections import Counter

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
        # self.class_names = [f.split('/')[0] for f in self.df['img_filename'].unique()]
        self.class_names = ['Landbird', 'Waterbird']
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
        print(f"{split} \t group counts: {Counter(self.groups)}")
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
        df = df[df.group.isin(groups)]
        return df.groupby('class').apply(lambda x: x.sample(n=num_per_class) if len(x) > num_per_class else x.sample(n=len(x))).reset_index(drop=True)['orig_idx'].values
