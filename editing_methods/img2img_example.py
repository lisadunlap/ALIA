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
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
from args import Img2ImgSingleArgs
import tyro

def main(args):

    if args.wandb_silent:
        os.environ['WANDB_SILENT']="true"

    prompt_str = args.prompt.replace(' ', '_').replace("\'", "").replace(',', '')
    wandb.init(project="Image-2-Image-Dev", name=f"{prompt_str}-{args.strength}-{args.guidance}", group=args.im_path, config=args, entity='lisadunlap')

    if 'xl' in args.model:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(args.model, torch_dtype=torch.float16, requires_safety_checker=False, safety_checker=None).to('cuda')
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model, torch_dtype=torch.float16, requires_safety_checker=False, safety_checker=None).to('cuda')

    init_image = Image.open(args.im_path)
    prompt = args.prompt
    print(f"Editing image {args.im_path} with prompt: {args.prompt}")
    generated = []
    for i in range(args.n):
        generated += pipe(prompt=prompt, image=init_image, strength=args.strength, guidance_scale=args.guidance, num_images_per_prompt=1).images
                
    save_dir = f'{args.save_dir}/img2img/{prompt_str}/strength-{args.strength}_guidance-{args.guidance}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plt.figure(figsize=(40, 40))
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
    plt.savefig(f'{save_dir}/samples/grid.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    images = wandb.Image(Image.open(f'{save_dir}/samples/grid.png'), caption="Top: Output, Bottom: Input")
    wandb.log({f"Original w/ Edits": images})

    for idx, im in enumerate(generated):
        im.save(f'{save_dir}/{idx}.png')
    
    wandb.log({f"Images": [wandb.Image(im, caption="Top: Output, Bottom: Input") for im in generated]})

if __name__ == "__main__":
    args = tyro.cli(Img2ImgSingleArgs)
    main(args)