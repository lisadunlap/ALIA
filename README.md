# Automatic Language-guided Image Augmentation (ALIA)

![Teaser](figures/method.png)

Welcome to the official repository for the paper ["Diversify Your Vision Datasets with Automatic Diffusion-based Augmentation"](https://arxiv.org/abs/2305.16289). If you prefer a condensed version, visit our [TL;DR website](https://lisadunlap.github.io/alia-website/). If you find our work useful, we welcome citations:

```markdown
@article{dunlap2023alia,
  author    = {Dunlap, Lisa and Umino, Alyssa and Zhang, Han and Yang, Jiezhi and Gonzalez, Joseph and Darrell, Trevor},
  title     = {Diversify Your Vision Datasets with Automatic Diffusion-based Augmentation},
  journal   = {arXiv},
  year      = {2023},
}
```

**NOTE:** We are currently in the process of releasing our code. The pipeline for recreating CUB is set up, with more experiments to come. If you encounter any issues, please raise them in our repository.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Prompt Generation](#prompt-generation)
3. [Generating Images](#generating-images)
4. [Filtering](#filtering)
5. [Training](#training)
6. [WandB Projects](#wandb-projects)
7. [Add Custom Datasets](#add-custom-datasets)

## Getting Started

To begin, install our code dependencies using Conda. You may need to adjust the `environment.yaml` file based on your setup:

```bash
conda env create -f environment.yaml
conda activate ALIA
pip install -e .
```

## Prompt Generation

- **Captioning**: We use the BLIP captioning model to caption the entire dataset:
  ```bash
  python caption.py --config configs/Cub2011/base.yaml
  ```
  This will save your captions [here](captions/Cub2011.csv).

- **LLM Summarization**: In our paper, we used GPT-4 to summarize the domains from the captions. Alternatively, we provide [Vicuna](https://chat.lmsys.org/) support for those who prefer not to give money to OpenAI. Download the Vicuna weights [here](https://github.com/lm-sys/FastChat/tree/main#vicuna-weights) (we used the 13b parameter model).
  ```bash
  pip3 install fastchat
  python huggingface_api.py message="Hi! How are you doing today?" #test to make sure it works
  python prompt_generation.py --config configs/Cub2011/base.yaml #return prompts
  ```
We randomly sample 20 captions to fit within the context length but highly encourage others to develop better methods :)

## Generating Images

Our editing methods are housed in [editing_methods](./editing_methods) and utilize the Huggingface Diffusers library and the [tyro](https://github.com/brentyi/tyro) CLI.

- **Per Example**: To generate multiple images given a prompt or edit a single image, use [txt2img_example.py](./editing_methods/txt2img_example.py) or [img2img_example.py](./editing_methods/img2img_example.py).
  ```bash
  python editing_methods/txt2img_example.py --prompt "Arachnophobia" --n 20
  ```

- **Per Dataset**: To generate images for an entire dataset, use the `class_names` attribute of the dataset to create per-class prompts.
  ```bash
  python editing_methods/img2img.py --dataset Cub2011 --prompt "a photo of a {} bird on rocks." --n 2
  ```

## Filtering

Once you have generated your data, determine which indices to filter out by running the following command:
```bash
python filtering/filter.py --config configs/Cub2011/alia.yaml filter.load=false
```

## Training

To train the base models or models with augmented data, simply run the appropriate YAML file from the configs folder.
```bash
python main.py --config configs/Cub2011/base.yaml
```
To apply a traditional data augmentation technique, set `data.augmentation=cutmix`. See all available data augmentations in the [load_dataset file](helpers/load_dataset.py).


## WandB Projects

Our datasets of generated data can be found [here](https://wandb.ai/clipinvariance/ALIA) under the 'Artifacts' tab. Each artifact includes the hyperparameters and prompts used to create it.

Download the images with the following command:
```python
import wandb
run = wandb.init()
artifact = run.use_artifact('clipinvariance/ALIA/cub_generic:v0', type='dataset')
artifact_dir = artifact.download()
```

View generated data examples for [Txt2Img](https://wandb.ai/lisadunlap/Text-2-Image), [Img2Img](https://wandb.ai/lisadunlap/Image-2-Image), and [InstructPix2Pix](https://wandb.ai/lisadunlap/InstructPix2Pix).

 ## Add Custom Datasets

To add your own dataset, you need to add a file to the datasets folder and then add it as an option in [helpers/load_dataset.py](helpers/load_dataset.py). The repository expects a dataset object of a specific format, where `__getitem__` should return three things: image, target, and group (group is the domain the image is in, set to 0 if it's not a bias/DA dataset).

Additionally, the dataset class needs to have the following parameters: `classes, groups, class_names, group_names, targets, class_weights`. Here's an example:

```python
class BasicDataset(torchvision.datasets.ImageFolder):
    """
    Wrapper class for torchvision.datasets.ImageFolder.
    """
    def __init__(self, root, transform=None, group=0, cfg=None):


        self.group = group # used for domain adaptation/bias datasets, where the group is the domain or bias type.
        super().__init__(root, transform=transform)
        self.groups = [self.group] * len(self.samples) # all images are from the same domain, set the group label to 0 for all of them
        self.group_names = ["all"] # only one group name (used for logging)
        self.class_names = self.classes # used for logging
        self.targets = [s[1] for s in self.samples] 
        self.class_weights = get_counts(self.targets) # class weights for XE loss

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, self.group
```

After adding your dataset to the [get_dataset](helpers/load_dataset.py) function, create a default config and set `data.base_dataset` to the name of your dataset. Then you should be able to generate the prompts and images, mimicking the `data.extra_dataset` parameters for CUB but replacing `data.extra_root` with the location of your generated data. 

For example, suppose you want to add a typical PyTorch ImageFolder dataset like ImageNet. You can manually determine how much data to add through either the extraset (real data baseline from the paper) or through the [data.num_extra](configs/base.yaml) parameter. If you want to use ALIA or other methods to improve performance, don't worry about the real data baseline and set `data.num_extra` to the number of augmented samples you want to add. For this example, say you want to add 1000 augmented samples to your training set. 

Since we already have a wrapper for the ImageFolder class in [datasets/base.py](datasets/base.py), you can use that to add your dataset (like ImageNet) into the `get_dataset` function.

```python
def get_dataset(dataset_name, transform, val_transform, root='/shared/lisabdunlap/data', embedding_root=None):
    .....

    elif dataset_name == 'ImageNet':
        trainset = BasicDataset(root='/path/to/imagenet/train', transform=transform)
        valset = BasicDataset(root='/path/to/imagenet/val', transform=val_transform)
        extraset = None # set to none since we are specifying the amount of generated data to add with data.num_extra
        testset = BasicDataset(root='/path/to/imagenet/val', transform=val_transform)
    ......

    return trainset, valset, testset, extraset
```

Now all you need to do is create your config:

```yaml
base_config: configs/base.yaml # this sets default parameters
proj: ALIA-ImageNet # wandb project
name: ImageNet # name of dataset used for logging (can set this to anything)

data: 
  base_dataset: ImageNet # name of dataset used in the new_get_dataset method
```

From here, you should be able to follow the README as normal. 
