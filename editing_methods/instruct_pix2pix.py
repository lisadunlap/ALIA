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
from diffusers import StableDiffusionInstructPix2PixPipeline

from helpers.load_dataset import get_dataset

parser = argparse.ArgumentParser(description='Dataset Understanding')
parser.add_argument('--prompt', type=str, help='prompts')
parser.add_argument('--dataset', type=str, default='Cub2011', help='dataset')
parser.add_argument('--save-dir', type=str, default='/shared/lisabdunlap/edited', help='save directory')
parser.add_argument('--image-guidance', type=float, default=1.2, help='how faithful to stay to the image (>= 1)')
parser.add_argument('--guidance', type=float, default=5.0, help='guidance of the prompt')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--n', type=int, default=1, help='number of images to generate')
parser.add_argument('--grid-log-freq', type=int, default=100, help='how often to log image grid')
parser.add_argument('--wandb-silent', action='store_true')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--class-agnostic', action='store_true', help='whether to use class agnostic prompts')
args = parser.parse_args()

prompts = {
    "Cub2011": "put the {} bird in the wild",
    "iWildCamMini": "put the {} in the wild",
}

np.random.seed(args.seed)

if args.wandb_silent:
    os.environ['WANDB_SILENT']="true"

wandb.init(project="instruct_pix2pix", name=f"{args.dataset}-{args.prompt}-{args.image_guidance}-{args.guidance}", group=args.dataset, config=args)

device = 'cuda'
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to(device)

# prompt = args.prompt

trainset, _, _ = get_dataset(args.dataset, transform=None, val_transform=None)
# print(f"Class names: {trainset.class_names}")

pattern = r'[0-9]'

dataset_idxs  = range(len(trainset)) if not args.test else np.random.choice(range(len(trainset)), 10, replace=False)
for i in dataset_idxs:
    init_image, label, _, _ = trainset[i]
    c = trainset.class_names[label]
    if args.prompt and args.class_agnostic:
        prompt = args.prompt
    elif args.prompt and not args.class_agnostic:
        prompt = args.prompt.format(re.sub(pattern, '', c).replace('_', ' ').replace('.', ''))
    else:
        prompt = prompts[args.dataset].format(re.sub(pattern, '', c).replace('_', ' ').replace('.', '')) if args.prompt is None else args.prompt
    generated = pipe(prompt=prompt, image=init_image, image_guidance_scale=args.image_guidance, guidance_scale=args.guidance, num_images_per_prompt=args.n).images
                
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # save_dir = f'{args.save_dir}/instruct/{args.dataset}/image_guidance-{args.image_guidance}_guidance-{args.guidance}/{c}'
    save_dir = f'{args.save_dir}/instruct/{args.dataset}/{args.prompt.replace(" ", "_").replace(".", "")}/strength-{args.image_guidance}_guidance-{args.guidance}/{c}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if i % args.grid_log_freq == 0 or args.test:
        fig = plt.figure(figsize=(10., 10.))
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
