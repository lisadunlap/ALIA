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
import tyro

from helpers.load_dataset import new_get_dataset
from args import Txt2ImgArgs

prompts = {
    "Cub2011": "a iNaturalist photo of a {} bird.",
    "iWildCamMini": "a camera trap photo of {} in the wild.",
    "Planes": "a photo of a {} airplane.",
}


def main(args):

    if args.wandb_silent:
        os.environ['WANDB_SILENT']="true"

    wandb.init(project="Text-2-Image", name=f"{args.prompt}",group=args.dataset, config=args)

    if args.safety_checker:
        pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, requires_safety_checker=False, safety_checker=None).to("cuda")

    print("getting dataset...")
    trainset, _, _, _ = new_get_dataset(args.dataset, transform=None, val_transform=None)


    pattern = r'[0-9]'
    classnames = [re.sub(pattern, '', c).replace('_', ' ').replace('.', '') for c in trainset.class_names]
    print(f"Class names: {classnames}")

    for c in classnames:
        prompt = prompts[args.dataset].format(c) if args.prompt is None else args.prompt.format(c)

        print(f"Prompt: {prompt} {type(prompt)}")
        n = args.n if not args.test else 2
        generated = []
        for batch in range(args.n // 2):
            generated += pipe(prompt=prompt, num_images_per_prompt=2).images
                    
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
            
        save_dir = f'{args.save_dir}/txt2img/{args.dataset}/{args.prompt.replace(" ", "_").replace(".", "")}/{c}' if args.prompt else f'{args.save_dir}/txt2img/{args.dataset}/{prompts[args.dataset]}/{c}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig = plt.figure(figsize=(50, 10.))
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

        for idx, im in enumerate(generated):
            im.save(f'{save_dir}/{idx}.png')

if __name__ == "__main__":
    args = tyro.cli(Txt2ImgArgs)
    main(args)
