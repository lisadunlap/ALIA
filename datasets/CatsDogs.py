import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import random

class CatsDogs:

    def __init__(self, root, cfg, split='train', transform=None):
        self.root = os.path.join(root, 'CatsDogs')
        self.cfg = cfg
        self.transform = transform
        self.df = pd.read_csv(f"{self.root}/{self.cfg.DATA.BIAS_TYPE}.csv")
        self.df = self.df[self.df['Split'] == split]
        self.filenames = [os.path.join(self.root, f) for f in self.df['Filename']]
        self.targets = np.array(self.df['Label'])
        self.domains = np.array(self.df['Domain'])
        self.classes = ['cat', 'dog']
        self.samples = zip(self.filenames, self.targets)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        if self.transform:
            img = self.transform(img)
        label = self.targets[idx]
        domain = self.domains[idx]
        return img, label, domain, domain