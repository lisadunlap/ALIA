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
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

from dataclasses import dataclass
import tyro
from args import Txt2ImgSingleArgs

def main(args):
    if args.wandb_silent:
        os.environ['WANDB_SILENT']="true"

    wandb.init(project="Text-2-Image", name=f"{args.prompt}", config=args)

    if args.safety_checker:
        pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, requires_safety_checker=False, safety_checker=None).to("cuda")

    print(f"Generating images for prompt: {args.prompt}")
    n = args.n if not args.test else args.mem_num_imgs
    generated = []
    for batch in tqdm(range(args.n // args.mem_num_imgs)):
        generated += pipe(prompt=args.prompt, num_images_per_prompt=args.mem_num_imgs).images

    save_dir = f'{args.save_dir}/txt2img/{args.prompt.replace(" ", "_")}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    wandb.log({"Samples": [wandb.Image(im) for im in generated]})
    for idx, im in enumerate(generated):
        im.save(f'{save_dir}/{idx}.png')

if __name__ == "__main__":
    args = tyro.cli(Txt2ImgSingleArgs)
    main(args)