from fnmatch import translate
import os
from shutil import SpecialFileError
import torch
from PIL import Image
import numpy as np
import pandas as pd
import random
from collections import Counter

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from datasets.base import get_counts

GROUP_NAMES_AIR_GROUND = np.array(['Airbus_ground', 'Airbus_air', 'Boeing_ground', 'Boeing_air'])
GROUP_NAMES_GRASS_ROAD = np.array(['Airbus_grass', 'Airbus_road', 'Boeing_grass', 'Boeing_road'])
GROUP_NAMES_ALL = np.array(['Airbus_air', 'Airbus_ground', 'Airbus_road', 'Boeing_air', 'Boeing_grass', 'Boeing_road'])

def get_label_mapping():
    return np.array(['Airbus', 'Boeing'])

class Planes:

    def __init__(self, root='./data', split='train', transform=None):
        self.class_labels = ['airbus', 'boeing']
        self.root = root
        self.transform = transform
        self.split = split
        self.df = pd.read_csv('./data/planes.csv')
        if self.split in ['train', 'test']:
            self.df = self.df[self.df['Split'] == split] if split != 'extra' else self.df[self.df['Split'] == 'val']
        if self.split == 'val':
            self.df = self.df[self.df['Split'] == split][::2]
            self.df = self.df[self.df['groups'].isin([0,2,3,4])]
        if self.split == 'extra': # remove unbiased examples
            # talk half of val set and move it to train
            self.df = self.df[self.df['Split'] == 'val'][1::2]
            # self.df = pd.concat([self.df[self.df['Split'] == 'train'], extra_df])
        self.filenames = np.array([os.path.join(self.root, f) for f in self.df['Filename']])
        self.labels = np.array(self.df['Label'])
        self.targets = self.labels
        self.domain_classes = sorted(np.unique(self.df['Ground']))
        self.domains = np.array([self.domain_classes.index(d) for d in self.df['Ground']])
        self.groups = np.array(self.df['groups'])
        print(f"Group counts: {Counter(self.groups)}")
        self.class_weights = get_counts(self.labels)
        self.samples = list(zip(self.filenames, self.labels))
        self.class_names = ['airbus', 'boeing']
        self.group_names = GROUP_NAMES_ALL 
        self.classes = ['airbus', 'boeing']
        print('PLANES {}'.format(split.upper()))
        print('LEN DATASET: {}'.format(len(self.filenames)))
        print('# AIRBUS:    {}'.format(len(np.where(self.labels == 0)[0])))
        print('# BOEING:    {}'.format(len(np.where(self.labels == 1)[0])))
        print(f'Grouping {self.domain_classes}: \n \t AIRBUS = {[len(self.df[(self.df["Label"] == 0) & (self.df["Ground"] == i)]) for i in self.domain_classes]} \n \t BOEING = {[len(self.df[(self.df["Label"] == 1) & (self.df["Ground"] == i)]) for i in self.domain_classes]}')
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx]).resize((224, 224))
        if self.transform:
            img = self.transform(img)
        label = self.targets[idx]
        domain = self.groups[idx]
        return img, label, domain