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

parser = argparse.ArgumentParser(description='Dataset Understanding')
parser.add_argument('--prompt', type=str, help='prompts')
parser.add_argument('--im-path', type=str, help='path to image')
parser.add_argument('--save-dir', type=str, default='/work/lisabdunlap/diffusion_playground', help='save directory')
parser.add_argument('--strength', type=float, default=0.4, help='strength of the edit')
parser.add_argument('--guidance', type=float, default=5.0, help='guidance of the prompt')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--n', type=int, default=2, help='number of images to generate')
parser.add_argument('--grid-log-freq', type=int, default=100, help='how often to log image grid')
parser.add_argument('--wandb-silent', action='store_true')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--class-agnostic', action='store_true', help='whether to use class agnostic prompts')
args = parser.parse_args()

np.random.seed(args.seed)

if args.wandb_silent:
    os.environ['WANDB_SILENT']="true"

prompt_str = args.prompt.replace(' ', '_').replace("\'", "").replace(',', '')
wandb.init(project="img2img-single", name=f"{prompt_str}-{args.strength}-{args.guidance}", group=args.im_path, config=args)

model_id_or_path = "stabilityai/stable-diffusion-2-1"
device = "cuda"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, requires_safety_checker=False, safety_checker=None)
pipe.to(device)

# prompt = args.prompt
# print(f"Class names: {trainset.class_names}")

pattern = r'[0-9]'

init_image = Image.open(args.im_path)
prompt = args.prompt
# print(f"Prompt: {prompt} {type(prompt)}")
generated = []
for i in range(args.n):
    generated += pipe(prompt=prompt, image=init_image, strength=args.strength, guidance_scale=args.guidance, num_images_per_prompt=1).images
            
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
save_dir = f'{args.save_dir}/img2img/{prompt_str}/strength-{args.strength}_guidance-{args.guidance}'
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
    plt.savefig(f'{save_dir}/samples/grid.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    images = wandb.Image(Image.open(f'{save_dir}/samples/grid.png'), caption="Top: Output, Bottom: Input")
    wandb.log({f"Example": images})

for idx, im in enumerate(generated):
    im.save(f'{save_dir}/{idx}.png')
    wandb.log({f"Generated {idx}": wandb.Image(im, caption="Top: Output, Bottom: Input")})
