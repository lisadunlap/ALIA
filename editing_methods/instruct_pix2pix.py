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
import tyro

from helpers.load_dataset import new_get_dataset
from args import InstructPix2PixArgs

prompts = {
    "Cub2011": "put the {} bird in the wild",
    "iWildCamMini": "put the {} in the wild",
    'Planes': "put the {} in the sky",
}

def main(args):

    np.random.seed(0)

    if args.wandb_silent:
        os.environ['WANDB_SILENT']="true"

    wandb.init(project="InstructPix2Pix", name=f"{args.dataset}-{args.prompt}-{args.image_guidance}-{args.guidance}", group=args.dataset, config=args)

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to('cuda')

    trainset, _, _, _ = new_get_dataset(args.dataset, transform=None, val_transform=None)

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

        # this is a hack for Cub
        if 'Whip poor Will' in prompt:
            prompt = prompt.replace('Whip poor Will', 'Eastern whip-poor-will')
        elif 'Geococcyx' in prompt:
            prompt = prompt.replace('Geococcyx', 'Roadrunner')
            
        generated = pipe(prompt=prompt, image=init_image, image_guidance_scale=args.image_guidance, guidance_scale=args.guidance, num_images_per_prompt=args.n).images
                    
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
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

if __name__ == "__main__":
    args = tyro.cli(InstructPix2PixArgs)
    main(args)