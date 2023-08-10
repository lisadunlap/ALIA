import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import wandb
import os
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import clip

from utils import read_unknowns, nest_dict, flatten_config

from huggingface_api import vicuna, load_vicuna

parser = argparse.ArgumentParser(description='ALIA')
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


run = wandb.init(project='ALIA', group='differences', config=flatten_config(args))

def remove_bad_captions(captions):
    # remove any captions with more than 5 repeating words
    new_captions, new_idxs = [], []
    for caption in captions:
        words = caption.split()
        #get word counts
        word_counts = {}
        for word in words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
        # check if any word has more than 5 counts
        if max(word_counts.values()) <= 5:
            new_captions.append(caption)
            new_idxs.append(captions.index(caption))
    print(f"Removed {len(captions) - len(new_captions)} out of {len(captions)} captions")
    return new_captions, new_idxs

# load captions from csv
captions_df = pd.read_csv(f"captions/{args.name}.csv")
# dedup
captions_df = captions_df.drop_duplicates(subset=['captions'])
captions = captions_df['captions'].tolist()

print("------------------------------------------")
print("------------- LOADING VICUNA -------------")
print("------------------------------------------")
llm, tokenizer = load_vicuna(args)

default_prompt = """
I have a set of image captions that I want to summarize into objective descriptions that describe the scenes, actions, camera pose, zoom, and other image qualities present. 
My captions are: 

{text}

I want the output to be a <=10 of captions that describe a unique setting, of the form \"{prefix}\".
Here are 3 examples of what I want the output to look like:
- {prefix} standing on a branch.
- {prefix} flying in the sky with the Austin skyline in the background.
- {prefix} playing in a river at night.

Based on the above captions, the output should be:
"""

# sample 20 captions
caption_sample = np.random.choice(captions, 20, replace=False)

prompt = default_prompt.format(text = "\n".join(caption_sample), prefix = args.summarize.prefix)


outputs = vicuna(args, prompt, llm, tokenizer, verbose=True)
print('-----------------------------------')
results = pd.DataFrame({'prompt': [prompt], 'outputs': [outputs]})
wandb.log({'results': wandb.Table(dataframe=results)})
wandb.summary['prompt'] = prompt
wandb.summary['outputs'] = outputs

