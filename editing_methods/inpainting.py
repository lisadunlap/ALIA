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
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline

from helpers.load_dataset import get_dataset

parser = argparse.ArgumentParser(description='Dataset Understanding')
parser.add_argument('--prompt', type=str, help='prompts')
parser.add_argument('--dataset', type=str, default='Cub2011Seg', help='dataset')
parser.add_argument('--save-dir', type=str, default='/shared/lisabdunlap/edited2', help='save directory')
parser.add_argument('--guidance', type=float, default=7.5, help='guidance of the prompt')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--n', type=int, default=2, help='number of images to generate')
parser.add_argument('--grid-log-freq', type=int, default=100, help='how often to log image grid')
parser.add_argument('--wandb-silent', action='store_true')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--total-num-images', type=int, default=1000, help='total number of images to generate (default: 1000)')
parser.add_argument('--class-agnostic', action='store_true', help='whether to use class agnostic prompts')
args = parser.parse_args()

prompts = {
    "Cub2011Seg": "a photo of a {} bird in the wild",
    "iWildCamMini": "a camera trap photo of {} in the wild.",
}

# np.random.seed(args.seed)

if args.wandb_silent:
    os.environ['WANDB_SILENT']="true"

wandb.init(project="inpainting", name=f"{args.dataset}-{args.prompt}-{args.guidance}", group=args.dataset, config=args)


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")


trainset, _, _ = get_dataset(args.dataset, transform=None)
# print(f"Class names: {trainset.class_names}")

pattern = r'[0-9]'
sample = 10 if args.test else args.total_num_images if args.total_num_images > 0 else len(trainset)
# dataset_idxs  = range(len(trainset)) if not args.test else np.random.choice(range(len(trainset)), 10, replace=False)
dataset_idxs = np.random.choice(range(len(trainset)), sample, replace=False)
for i in dataset_idxs:
    img, label, _, mask = trainset[i]
    img, mask = img.resize((512, 512)), mask.resize((512, 512))
    c = trainset.classes[label]
    new_c = np.random.choice(trainset.classes)
    prompt = prompts[args.dataset].format(new_c) if args.prompt is None else args.prompt
    print(f"Prompt: {prompt} | Class: {c} | New Class: {new_c}")
    generated = pipe(prompt=prompt, image=img, mask_image=mask, guidance_scale=args.guidance, num_images_per_prompt=args.n).images
                
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_dir = f'{args.save_dir}/inpainting/{args.dataset}/guidance-{args.guidance}/{new_c}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if i % args.grid_log_freq == 0 or args.test:
        fig = plt.figure(figsize=(10., 10.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                            nrows_ncols=(1, args.n + 1),  # creates 2x2 grid of axes 
                            axes_pad=0.1,  # pad between axes in inch.
                            )
        for ax, im in zip(grid, [img] + generated):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            ax.axis('off')
            ax.set_title(prompt)
        if not os.path.exists(f'{save_dir}/samples'):
            print("making dir")
            os.makedirs(f'{save_dir}/samples')
        plt.savefig(f'{save_dir}/samples/{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        images = wandb.Image(Image.open(f'{save_dir}/samples/{i}.png'), caption="Top: Output, Bottom: Input")
        wandb.log({f"Example {c}": images})

    for idx, im in enumerate(generated):
        im.save(f'{save_dir}/{i}-{idx}-{label}.png')