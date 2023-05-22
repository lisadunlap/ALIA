import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import torch
# from torch import autocast
from torchvision import transforms
import argparse
import re
import os
import io
import wandb
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from omegaconf import OmegaConf
from utils import evaluate, read_unknowns, nest_dict, flatten_config
from helpers.load_dataset import  get_train_transform, get_filtered_dataset, get_val_transform

from helpers.load_dataset import get_dataset, new_get_dataset

parser = argparse.ArgumentParser(description='Dataset Understanding')
parser.add_argument('--prompt', type=str, help='prompts', required=True)
parser.add_argument('--dataset', type=str, help='dataset', required=True)
parser.add_argument('--save-dir', type=str, default='./diffusion_generated_data', help='save directory')
parser.add_argument('--strength', type=float, default=0.4, help='strength of the edit')
parser.add_argument('--guidance', type=float, default=5.0, help='guidance of the prompt')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--n', type=int, default=2, help='number of images to generate')
parser.add_argument('--grid-log-freq', type=int, default=100, help='how often to log image grid')
parser.add_argument('--wandb-silent', action='store_true')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--class-agnostic', action='store_true', help='whether to use class agnostic prompts')
parser.add_argument('--config', default='configs/base.yaml', help="config file")
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
args, unknown = parser.parse_known_args()

overrides = OmegaConf.from_cli(args.overrides)
cfg       = OmegaConf.load(args.config)
base      = OmegaConf.load('configs/base.yaml')
dataset_base = OmegaConf.load(cfg.base_config)
cfg      = OmegaConf.merge(base, dataset_base, cfg, overrides)
if len(unknown) > 0:
    print(unknown)
    config = nest_dict(read_unknowns(unknown))
    to_merge = OmegaConf.create(config)
    args = OmegaConf.merge(args, to_merge)
cfg.yaml = args.config

prompts = {
    "Cub2011": "a photo of a {} bird in the wild",
    "iWildCamMini": "a camera trap photo of {} in the wild.",
}

np.random.seed(args.seed)

if args.wandb_silent:
    os.environ['WANDB_SILENT']="true"

wandb.init(project="img2img", name=f"{cfg.data.base_dataset}-{args.prompt}-{args.strength}-{args.guidance}", group=args.dataset, config=args)

model_id_or_path = "runwayml/stable-diffusion-v1-5"
device = "cuda"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, requires_safety_checker=False, safety_checker=None)
pipe.to(device)

# prompt = args.prompt
# trainset, valset, testset = get_filtered_dataset(cfg, transforms.Resize((256,256)), None)
trainset, _, _, _ = new_get_dataset(args.dataset, transform=None, val_transform=None, root='/shared/lisabdunlap/data')
print(f"Class names: {trainset.class_names}")

pattern = r'[0-9]'

dataset_idxs  = range(len(trainset)) if not args.test else np.random.choice(range(len(trainset)), 10, replace=False)
for i in dataset_idxs:
    init_image, label, _, _ = trainset[i]
    c = trainset.class_names[label]
    prompt_c = c if not cfg.data.generated_classes else cfg.data.generated_classes[label]
    print(f"Class: {c} promt: {prompt_c}")
    if args.prompt and args.class_agnostic:
        prompt = args.prompt
    elif args.prompt and not args.class_agnostic:
        prompt = args.prompt.format(re.sub(pattern, '', prompt_c).replace('_', ' ').replace('.', ''))
    print(f"Prompt: {prompt} {type(prompt)}")
    generated = pipe(prompt=prompt, image=init_image, strength=args.strength, guidance_scale=args.guidance, num_images_per_prompt=args.n).images
                
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_dir = f'{args.save_dir}/img2img/{args.dataset}/{args.prompt.replace(" ", "_").replace(".", "")}/strength-{args.strength}_guidance-{args.guidance}/{c}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if i % args.grid_log_freq == 0 or args.test:
        fig = plt.figure(figsize=(10, 10.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                            nrows_ncols=(1, args.n + 1),  # creates 2x2 grid of axes 
                            axes_pad=0.1,  # pad between axes in inch.
                            )
        for ax, im in zip(grid, [init_image] + generated):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            ax.axis('off')
        if not os.path.exists(f'{save_dir}/samples'):
            print("making dir")
            os.makedirs(f'{save_dir}/samples')
        plt.savefig(f'{save_dir}/samples/{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        images = wandb.Image(Image.open(f'{save_dir}/samples/{i}.png'), caption="Top: Output, Bottom: Input")
        wandb.log({f"Example {c}": images})

    for idx, im in enumerate(generated):
        im.save(f'{save_dir}/{i}-{idx}.png')
