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
from diffusers import StableDiffusionImg2ImgPipeline

from helpers.load_dataset import get_dataset
from args import Img2ImgArgs
import tyro

prompts = {
    "Cub2011": "a iNaturalist photo of a {} bird.",
    "iWildCamMini": "a camera trap photo of {} in the wild.",
    "Planes": "a photo of a {} airplane.",
}

def main(args):
    #set seed for visualizaiton
    np.random.seed(0)

    if args.wandb_silent:
        os.environ['WANDB_SILENT']="true"

    wandb.init(project="Image-2-Image", name=f"{args.dataset}-{args.prompt}-{args.strength}-{args.guidance}", group=args.dataset, config=args)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model, torch_dtype=torch.float16, requires_safety_checker=False, safety_checker=None).to("cuda")

    trainset, _, _, _ = get_dataset(args.dataset, transform=None, val_transform=None, root='/shared/lisabdunlap/data')
    print(f"Class names: {trainset.class_names}")

    pattern = r'[0-9]'

    dataset_idxs  = range(len(trainset)) if not args.test else np.random.choice(range(len(trainset)), 10, replace=False)
    for i in dataset_idxs:
        item = trainset[i]
        init_image, label = item[0], item[1]
        c = trainset.class_names[label]
        if args.prompt and args.class_agnostic:
            prompt = args.prompt
        elif args.prompt and not args.class_agnostic:
            prompt = args.prompt.format(re.sub(pattern, '', c).replace('_', ' ').replace('.', ''))
        print(f"Prompt: {prompt} {type(prompt)}")

        # this is a hack for Cub
        if 'Whip poor Will' in prompt:
            prompt = prompt.replace('Whip poor Will', 'Eastern whip-poor-will')
        elif 'Geococcyx' in prompt:
            prompt = prompt.replace('Geococcyx', 'Roadrunner')
            
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

if __name__ == "__main__":
    args = tyro.cli(Img2ImgArgs)
    main(args)