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
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

from helpers.load_dataset import get_dataset

parser = argparse.ArgumentParser(description='Dataset Understanding')
parser.add_argument('--prompt', type=str, help='prompts')
parser.add_argument('--dataset', type=str, default='Cub2011', help='dataset')
parser.add_argument('--save-dir', type=str, default='/shared/lisabdunlap/edited', help='save directory')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--n', type=int, default=50, help='number of images to generate')
parser.add_argument('--wandb-silent', action='store_true')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--class-agnostic', action='store_true', help='whether to use class agnostic prompts')
args = parser.parse_args()

prompts = {
    "Cub2011": "a iNaturalist photo of a {} bird.",
    "iWildCamMini": "a camera trap photo of {} in the wild.",
}

np.random.seed(args.seed)

if args.wandb_silent:
    os.environ['WANDB_SILENT']="true"

wandb.init(project="txt2img", name=f"{args.dataset}", group=args.dataset, config=args)

# model_id_or_path = "runwayml/stable-diffusion-v1-5"
model_id_or_path = "stabilityai/stable-diffusion-2"
device = "cuda"
negative_prompts = ['longbody', 'lowres', 'bad anatomy', 'extra digit', 'cropped', 'worst quality', 'low quality']
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, requires_safety_checker=False, safety_checker=None)
pipe = pipe.to("cuda")

# prompt = args.prompt

if not args.class_agnostic:
    trainset, _, _ = get_dataset(args.dataset, transform=None, val_transform=None)
    pattern = r'[0-9]'
    classnames = [re.sub(pattern, '', c).replace('_', ' ').replace('.', '') for c in trainset.class_names]
    print(f"Class names: {classnames}")
else: 
    classnames = ['']

for c in classnames:
    prompt = prompts[args.dataset].format(c) if args.prompt is None else args.prompt
    print(f"Prompt: {prompt} {type(prompt)}")
    n = args.n if not args.test else 2
    generated = []
    for batch in range(args.n // 2):
        generated += pipe(prompt=prompt, num_images_per_prompt=2).images
                
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.class_agnostic:
        save_dir = f'{args.save_dir}/txt2img/{args.dataset}/{prompt.replace(" ", "_")}'
        c = 'none'
    else:
        save_dir = f'{args.save_dir}/txt2img/{args.dataset}/{prompt}/{c}' if args.prompt else f'{args.save_dir}/txt2img/{args.dataset}/{prompts[args.dataset]}/{c}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(1, min([n, 10])),  # creates 2x2 grid of axes 
                        axes_pad=0.1,  # pad between axes in inch.
                        )
    for ax, im in zip(grid, generated):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis('off')
    if not os.path.exists(f'{save_dir}/samples'):
        print("making dir")
        os.makedirs(f'{save_dir}/samples')
    plt.savefig(f'{save_dir}/samples/{c}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    images = wandb.Image(Image.open(f'{save_dir}/samples/{c}.png'), caption="Top: Output, Bottom: Input")
    wandb.log({f"Example {c}": images})
    wandb.log({"Samples": [wandb.Image(im) for im in generated]})

    for idx, im in enumerate(generated):
        im.save(f'{save_dir}/{idx}.png')