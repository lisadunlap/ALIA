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
from diffusers import StableDiffusionSAGPipeline

from helpers.load_dataset import get_dataset

parser = argparse.ArgumentParser(description='Dataset Understanding')
parser.add_argument('--prompt', type=str, help='prompts')
parser.add_argument('--dataset', type=str, default='Cub2011', help='dataset')
parser.add_argument('--save-dir', type=str, default='/work/lisabdunlap/edited', help='save directory')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--n', type=int, default=50, help='number of images to generate')
parser.add_argument('--wandb-silent', action='store_true')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
args = parser.parse_args()

prompts = {
    "Cub2011": "a iNaturalist photo of a {} bird.",
    "iWildCamMini": "a camera trap photo of {} in the wild.",
}

np.random.seed(args.seed)

if args.wandb_silent:
    os.environ['WANDB_SILENT']="true"

wandb.init(project="txt2img", name=f"{args.dataset}-{args.prompt}", group=args.dataset, config=args)

model_id_or_path = "runwayml/stable-diffusion-v1-5"
device = "cuda"

pipe = StableDiffusionSAGPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# prompt = args.prompt

trainset, _, _ = get_dataset(args.dataset, transform=None)


pattern = r'[0-9]'
classnames = [re.sub(pattern, '', c).replace('_', ' ').replace('.', '') for c in trainset.class_names]
print(f"Class names: {classnames}")

for c in classnames:
    prompt = prompts[args.dataset].format(c) if args.prompt is None else args.prompt.format(c)
    if 'Whip poor Will' in prompt:
        prompt = prompt.replace('Whip poor Will', 'Eastern whip-poor-will')
    elif 'Geococcyx' in prompt:
        prompt = prompt.replace('Geococcyx', 'Roadrunner')

    print(f"Prompt: {prompt} {type(prompt)}")
    n = args.n if not args.test else 2
    generated = []
    for batch in range(args.n // 3):
        generated += pipe(prompt=prompt, num_images_per_prompt=3).images
                
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    save_dir = f'{args.save_dir}/txt2img_attn/{args.dataset}/{args.prompt.replace(" ", "_").replace(".", "")}/{c}' if args.prompt else f'{args.save_dir}/txt2img/{args.dataset}/{prompts[args.dataset]}/{c}'
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
