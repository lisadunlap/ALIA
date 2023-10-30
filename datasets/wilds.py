import torch
from PIL import Image
import wilds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torchvision
from datasets.base import get_counts
from collections import Counter

from wilds.common.metrics.all_metrics import Accuracy, Recall, F1

LOCATION_MAP = {1: 0, 78: 1}
LOCATION_MAP_INV = {0: 1, 1: 78}

class Wilds:
    """
    Wrapper for the WILDS dataset.
    """
    def __init__(self, root="/work/datasets", split='train', transform=None):
        self.dataset = wilds.get_dataset(dataset="iwildcam", root_dir=root)#, download=True)
        self.split_dataset = self.dataset.get_subset(split, transform=transform)
        self._metadata_fields = self.split_dataset.metadata_fields
        self.classes = list(range(self.split_dataset.n_classes))
        self.df = pd.DataFrame(self.split_dataset.metadata_array.numpy(), columns=self.split_dataset.metadata_fields)
        self.groups = self.df.location.values
        self.labels = self.targets = self.df.y.values
        self.class_weights = get_counts(self.labels)
        self.samples = [(i, l) for (i, l) in zip(self.df.sequence.values, self.labels)]
        # self.locations = [m[0] for m in self.split_dataset._metadata_array]
        # location = LOCATION_MAP_INV[1] if split == 'test' else LOCATION_MAP_INV[0]
        # self.location_idxs = np.where(np.array(self.locations) == location)[0]
        # self.groups = [location for _ in range(len(self.location_idxs))]

    def __len__(self):
        return len(self.split_dataset)

    def __getitem__(self, idx):
        # map the idx to the location idx (filter out all other locations)
        # idx = self.location_idxs[idx]
        img, label, metadata = self.split_dataset[idx]
        return img, label, self.groups[idx]

class WILDS:
    """
    Specific subset of WILDS containing 6 classes and 2 test locations.
    """
    def __init__(self, root='/work/datasets/iwildcam_v2.0/train', split='train', transform=None):
        self.root = root
        self.df = pd.read_csv(f'./data/iwildcam_v2.0/{split}_subset.csv')
        # self.df = pd.read_csv(f'/work/lisabdunlap/DatasetUnderstanding/data/{split}_subset.csv')
        self.transform = transform
        self.class_names = sorted(self.df.y.unique())
        self.label_map = {j:i for i, j in enumerate(self.class_names)}
        self.labels = [self.label_map[i] for i in self.df.y]
        self.samples = [(os.path.join(root, i), l) for (i, l) in zip(self.df.filename, self.labels)]
        self.targets = [l for _, l in self.samples]
        self.classes = list(sorted(self.df.y.unique()))
        self.locations = self.df.location_remapped.unique()
        self.location_map = {j:i for i, j in enumerate(self.locations)}
        self.location_labels = [self.location_map[i] for i in self.df.location_remapped]
        self.groups = self.location_labels
        self.class_weights = get_counts(self.labels)
        self.group_names = self.locations
        self.class_names = ['background', 'cattle', 'elephants', 'impalas', 'zebras', 'giraffes', 'dik-diks']
        print(f"Num samples per class {Counter(self.labels)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        location = self.location_labels[idx]
        return img, label, location

    def inspect_location(self, location):
        assert location in self.locations
        location_df = self.df[self.df.location_remapped == location]
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        idx = np.random.choice(list(range(len(location_df))))
        location_df['y'].value_counts().plot(kind='bar', ax=axs[0])
        axs[0].set_title(f'Location {location} (n={len(location_df)}) class counts')
        axs[1].imshow(Image.open(os.path.join(self.root, location_df.iloc[idx].filename)))
        axs[1].set_title(f'Location {location} (n={len(location_df)}) class {location_df.iloc[idx].y} (idx={idx})')
        axs[1].axis('off')
        plt.show()

    def inspect_class(self, class_idx):
        assert class_idx in self.classes
        class_df = self.df[self.df.y == class_idx]
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        idx = np.random.choice(list(range(len(class_df))))
        class_df['location_remapped'].value_counts().plot(kind='bar', ax=axs[0])
        axs[0].set_title(f'Class {class_idx} (n={len(class_df)}) location counts')
        axs[1].imshow(Image.open(os.path.join(self.root, class_df.iloc[idx].filename)))
        axs[1].set_title(f'Class {class_idx} (n={len(class_df)}) location {class_df.iloc[idx].location_remapped} (idx={idx}) ({class_df.iloc[idx].filename})')
        axs[1].axis('off')
        plt.show()

    @staticmethod
    def eval(y_pred, y_true, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metrics = [
            Accuracy(prediction_fn=prediction_fn),
            Recall(prediction_fn=prediction_fn, average='macro'),
            F1(prediction_fn=prediction_fn, average='macro'),
        ]

        results = {}

        for i in range(len(metrics)):
            results.update({
                **metrics[i].compute(y_pred, y_true),
                        })

        results_str = (
            f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
            f"Recall macro: {results[metrics[1].agg_metric_field]:.3f}\n"
            f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n"
        )

        return results, results_str

class WILDSDiffusion(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.df = pd.DataFrame({'img_filename': self.samples, 'y': self.targets})
        self.class_names = ['background', 'cattle', 'elephants', 'impalas', 'zebras', 'giraffes', 'dik-diks']
        # self.class_names = ['background', 'cattle', 'elephant', 'imapala', 'zebra', 'giraffe', 'dik-dik']
        # self.classes = [int(c) for c in self.classes] 
        self.labels = self.targets
        self.classes = [0, 24, 32, 36, 48, 49, 52]
        # self.class_map = {j:i for i, j in enumerate(self.classes)}
        self.groups = [1000] * len(self.samples)
        self.class_weights = torch.tensor(get_counts(self.labels))
        self.group_names = [1000]
        self.targets = [l for _, l in self.samples]
    

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, self.groups[idx]

    def inspect_class(self, class_idx):
        class_idxs = np.where(np.array(self.labels) == self.class_map[class_idx])[0]
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        idx = np.random.choice(class_idxs)
        filename, label = self.samples[idx]
        ax.imshow(Image.open(filename))
        ax.set_title(f'Class {class_idx} (n={len(class_idxs)}) (idx={idx})')
        ax.axis('off')
        plt.show()

def wilds_eval(y_pred, y_true, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metrics = [
            Accuracy(prediction_fn=prediction_fn),
            Recall(prediction_fn=prediction_fn, average='macro'),
            F1(prediction_fn=prediction_fn, average='macro'),
        ]

        results = {}

        for i in range(len(metrics)):
            results.update({
                **metrics[i].compute(y_pred, y_true),
                        })

        results_str = (
            f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
            f"Recall macro: {results[metrics[1].agg_metric_field]:.3f}\n"
            f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n"
        )

        return results, results_str