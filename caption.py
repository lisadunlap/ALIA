import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import torch
# from torch import autocast
import argparse
import re
import os
import io
import wandb
from PIL import Image
from omegaconf import OmegaConf
import pandas as pd

# from models import *
import datasets
import models
from utils import read_unknowns, nest_dict, flatten_config
# from wandb_utils import WandbData
from helpers.load_dataset import get_filtered_dataset

parser = argparse.ArgumentParser(description='Dataset Understanding')
parser.add_argument('--config', default='configs/base.yaml', help="config file")
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
flags, unknown = parser.parse_known_args()

overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base      = OmegaConf.load('configs/base.yaml')
dataset_base = OmegaConf.load(cfg.base_config)
args      = OmegaConf.merge(base, dataset_base, cfg, overrides)
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

torch.manual_seed(args.seed)
np.random.seed(args.seed)

augmentation = 'none' if not args.data.augmentation else args.data.augmentation
augmentation = f'{augmentation}-filtered' if args.data.filter else f'augmentation-unfiltered'

run = wandb.init(project=args.proj, group='captions', config=flatten_config(args))

# Data
print('==> Preparing data..')
# trainset, valset, testset = get_dataset(args.data.base_dataset, transform)
trainset, valset, testset = get_filtered_dataset(args, None, None)


device = "cuda"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

results = {"idx": [], "target": [], "captions": [], "path": []}
for i, (img, target, _, _) in enumerate(trainset):
    # unconditional image captioning
    # if target == 0:
    inputs = processor(img, return_tensors="pt").to(device)


    out = model.generate(**inputs)
    output = processor.decode(out[0], skip_special_tokens=True)
    print(output)
    results["idx"].append(i)
    results["target"].append(target)
    results["captions"].append(output)
    results["path"].append(trainset.samples[i][0])

caption_df = pd.DataFrame(results)
caption_df.to_csv(f"captions/{args.name}.csv")

# visualize(captions, trainset)
captions = caption_df["captions"].tolist()
# sample = np.random.choice(len(captions), 10, replace=False)
# wandb.log({"image_caption_pairs": [wandb.Image(trainset[idx][0], caption=captions[idx]) for idx in sample]})
caption_df['image'] = [wandb.Image(trainset[idx][0], caption=captions[idx]) for idx in range(len(captions))]
wandb.log({"all_captions": wandb.Table(dataframe=caption_df[['idx', 'target', 'captions', 'image']])})



# # conditional image captioning
# text = "a camera trap photo containing"
# inputs = processor(raw_image, text, return_tensors="pt").to(device)


# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))

# # unconditional image captioning
# inputs = processor(raw_image, return_tensors="pt").to(device)


# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))

