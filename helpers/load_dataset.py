import os, io
from tqdm import tqdm
import torch as torch
import torchvision
from torchvision import transforms
import numpy as np
import torchvision.datasets as dsets
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import clip
from torchvision.transforms.functional import crop

import datasets
from datasets.Waterbirds import Waterbirds, WaterbirdsInverted
from datasets.base import *
# from wilds.datasets.iwildcam_dataset import IWildCamDataset
from datasets.wilds import WILDS, WILDSDiffusion, Wilds, WILDSFine
from datasets.cub import Cub2011, Cub2011Painting, Cub2011Diffusion, Cub2011Seg, newCub2011
from datasets.planes import Planes
# from cutmix.cutmix import CutMix

def get_config(name="ColoredMNIST"):
    if "ColoredMNIST" in name:
        cfg       = OmegaConf.load('dataset_configs/ColoredMNIST.yaml')
    elif 'CatsDogs' in name:
        cfg       = OmegaConf.load('dataset_configs/CatsDogs.yaml')
    elif 'Waterbirds' in name:
        cfg       = None
    else:
        raise ValueError("Dataset config not found")
    args      = cfg
    return args

def crop_wilds(image):
    return crop(image, 10, 0, 400, 448)

def get_train_transform(dataset_name="Imagenet", model=None, augmentation=None):
    """"
    Gets the transform for a given dataset
    """
    transform_list = []
    # if augmentation == "cutmix":
    #     return ["cutmix", transforms.Compose(transform_list)]
    if augmentation == "augmix":
        print("Applying AugMix")
        transform_list.append(transforms.AugMix())
    if augmentation == "color-jitter":
        print("Applying color jitter")
        transform_list.append(transforms.ColorJitter(brightness=.5, hue=.3))
    if augmentation == "random":
        print("Applying random augmentations")
        transform_list.append(transforms.RandAugment())
    if augmentation == "auto":
        print("Applying automatic augmentations")
        transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
    if augmentation == 'rotation':
        print("Applying scale jitter")
        transform_list.append(transforms.RandomRotation(10))
    if augmentation == 'scale_jitter':
        print("Applying scale jitter")
        transform_list.append(transforms.v2.ScaleJitter(target_size=(224, 224)))
    if augmentation == 'color_jitter':
        print("Applying color jitter")
        transform_list.append(transforms.ColorJitter(brightness=.5, hue=.3))

    if model in ['RN50', 'ViT-B/32']: # if we are evaluating a clip model we use its transforms
        print("...loading CLIP model")
        net, transform = clip.load(model)
    elif "Imagenet" in dataset_name:
        transform_list += [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    elif "Waterbirds" in dataset_name:
        transform_list += [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    elif dataset_name == "ColoredMNIST":
        transform_list += [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.0852, 0.0492, 0.0535], [0.2488, 0.1938, 0.2005])
        ]
    elif dataset_name == "CatsDogs":
        transform_list += [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5147, 0.4716, 0.4201], [0.2438, 0.2358, 0.2436])
        ]
    elif "iWildCam" in dataset_name:
        transform_list += [transforms.ToTensor(),
                                #   transforms.Grayscale(num_output_channels=3),
                                  transforms.Resize((448, 448)),
                                  transforms.Lambda(crop_wilds),
                                  transforms.Resize((224, 224))]
    else:
        transform_list += [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    
    return transforms.Compose(transform_list)

def get_val_transform(dataset_name="Imagenet", model=None):
    """"
    Gets the transform for a given dataset
    """
    transform_list = []
    if model in ['RN50', 'ViT-B/32']: # if we are evaluating a clip model we use its transforms
        print("...loading CLIP model")
        net, transform = clip.load(model)
    elif "Imagenet" in dataset_name:
        transform_list += [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    elif "Waterbirds" in dataset_name:
        transform_list += [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    elif dataset_name == "ColoredMNIST":
        transform_list += [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.0852, 0.0492, 0.0535], [0.2488, 0.1938, 0.2005])
        ]
    elif dataset_name == "CatsDogs":
        transform_list += [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5147, 0.4716, 0.4201], [0.2438, 0.2358, 0.2436])
        ]
    elif "iWildCam" in dataset_name:
        transform_list += [transforms.ToTensor(),
                                #   transforms.Grayscale(num_output_channels=3),
                                  transforms.Resize((448, 448)),
                                  transforms.Lambda(crop_wilds),
                                  transforms.Resize((224, 224))]
    else:
        transform_list += [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    
    return transforms.Compose(transform_list)

def get_dataset(dataset_name, transform, val_transform, root='/shared/lisabdunlap/data'):
    if dataset_name == "Waterbirds": # change these data paths
        trainset = Waterbirds(root=root, split='train', transform=transform)
        valset = Waterbirds(root=root, split='val', transform=transform)
        idxs = valset.get_subset(groups=[0,3], num_per_class=100)
        valset = torch.utils.data.Subset(trainset, idxs) 
        testset = Waterbirds(root=root, split='test', transform=transform)
    elif dataset_name == "Waterbirds100": # change these data paths
        trainset = Waterbirds(root=root, split='train', transform=transform)
        valset = Waterbirds(root=root, split='val', transform=transform)
        testset = Waterbirds(root=root, split='test', transform=transform)
    elif dataset_name == "iWildCam":
        trainset = Wilds(root=root, split='train', transform=transform)
        valset = Wilds(root=root, split='val', transform=transform)
        testset = Wilds(root=root, split='test', transform=transform)
    elif dataset_name == "iWildCamMini":
        trainset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='train', transform=transform)
        valset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='val', transform=transform)
        testset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='test', transform=transform)
    elif dataset_name == "iWildCamMiniExtra":
        trainset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='train', transform=transform)
        trainset_extra = WILDS(root=f'{root}/iwildcam_v2.0/train', split='train_extra', transform=transform)
        trainset = CombinedDataset([trainset, trainset_extra])
        valset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='val', transform=transform)
        testset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='test', transform=transform)
    elif dataset_name == 'Cub2011':
        trainset = Cub2011(root=root, subset=False, split='train', transform=transform)
        valset = Cub2011(root=root, split='val', transform=transform)
        testset = valset
    elif dataset_name == 'newCub2011':
        trainset = newCub2011(root=root, split='train', transform=transform)
        valset = newCub2011(root=root, split='val', transform=transform)
        testset = valset
    # elif dataset_name == 'Cub2011Paintings':
    #     trainset = Cub2011Painting(root=root, subset=False, split='train', transform=transform)
    #     valset = Cub2011(root=root, split='val', transform=transform)
    elif dataset_name == 'Cub2011Seg':
        trainset = Cub2011Seg(root='/shared/lisabdunlap/data', subset=False, split='train', transform=transform)
        valset = Cub2011Seg(root='/shared/lisabdunlap/data', split='val', transform=transform)
        testset = valset
    elif dataset_name == 'Cub2011Extra':
        trainset = Cub2011(root=root, subset=False, split='train', transform=transform)
        extraset = Cub2011(root=root, subset=False, split='extra', transform=transform)
        trainset = CombinedDataset([trainset, extraset])
        valset = Cub2011(root=root, split='val', transform=transform)
        testset = valset
    elif dataset_name == 'newCub2011Extra':
        trainset = newCub2011(root=root, subset=False, split='train', transform=transform)
        extraset = newCub2011(root=root, subset=False, split='extra', transform=transform)
        trainset = CombinedDataset([trainset, extraset])
        valset = newCub2011(root=root, split='val', transform=transform)
        testset = valset
    elif dataset_name == 'Planes' or dataset_name == 'PlanesExtra':
        print("ROOT ", root)
        trainset = Planes(split='train', transform=transform)
        valset = Planes(split='val', transform=val_transform)
        testset = Planes(split='test', transform=val_transform)
    return trainset, valset, testset

def new_get_dataset(dataset_name, transform, val_transform, root='/shared/lisabdunlap/data', embedding_root=None):
    if dataset_name == "Waterbirds": # change these data paths
        trainset = Waterbirds(root=root, split='train', transform=transform)
        valset = Waterbirds(root=root, split='val', transform=val_transform)
        idxs = valset.get_subset(groups=[0,3], num_per_class=100)
        extra_idxs = valset.get_subset(groups=[1,2], num_per_class=100)
        valset = Subset(trainset, idxs) 
        extraset = Subset(trainset, extra_idxs)
        testset = Waterbirds(root=root, split='test', transform=val_transform)
    elif dataset_name == "iWildCamMini" or dataset_name == "iWildCamMiniExtra":
        trainset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='train', transform=transform)
        valset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='val', transform=val_transform)
        testset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='test', transform=val_transform)
        extraset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='train_extra', transform=transform)
        if dataset_name == 'iWildCamMiniExtra':
            trainset = CombinedDataset([trainset, extraset])
    elif dataset_name == "iWildCamMini-fine" or dataset_name == "iWildCamMiniExtra-fine":
        trainset = WILDSFine(root=f'{root}/iwildcam_v2.0/train', split='train_fine', transform=transform)
        valset = WILDSFine(root=f'{root}/iwildcam_v2.0/train', split='val_fine', transform=val_transform)
        testset = WILDSFine(root=f'{root}/iwildcam_v2.0/train', split='test_fine', transform=val_transform)
        extraset = WILDSFine(root=f'{root}/iwildcam_v2.0/train', split='train_extra_fine', transform=transform)
        if dataset_name == 'iWildCamMiniExtra-fine':
            trainset = CombinedDataset([trainset, extraset])
    elif dataset_name == 'Cub2011' or dataset_name == 'Cub2011Extra':
        print("ROOT yo ", root)
        trainset = Cub2011(root=root, subset=False, split='train', transform=transform)
        valset = Cub2011(root=root, split='val', transform=val_transform)
        extraset = Cub2011(root=root, subset=False, split='extra', transform=transform)
        testset = valset
        if dataset_name == 'Cub2011Extra':
            trainset = CombinedDataset([trainset, extraset])
    elif dataset_name == 'newCub2011' or dataset_name == 'newCub2011Extra':
        trainset = newCub2011(root=root,  split='train', transform=transform)
        valset = newCub2011(root=root, split='val', transform=val_transform)
        extraset = newCub2011(root=root, split='extra', transform=transform)
        testset = valset
        if dataset_name == 'Cub2011Extra':
            trainset = CombinedDataset([trainset, extraset])
    elif dataset_name == 'Planes' or dataset_name == 'PlanesExtra':
        trainset = Planes(split='train', transform=transform)
        valset = Planes(split='val', transform=val_transform)
        extraset = Planes(split='extra', transform=transform)
        testset = Planes(split='test', transform=val_transform)
        if dataset_name == 'PlanesExtra':
            trainset = CombinedDataset([trainset, extraset])
    if embedding_root:
        trainset = EmbeddingDataset(os.path.join(embedding_root, dataset_name), trainset, split='train')
        valset = EmbeddingDataset(os.path.join(embedding_root, dataset_name), valset, split='val')
        testset = EmbeddingDataset(os.path.join(embedding_root, dataset_name), testset, split='test')

    return trainset, valset, testset, extraset

def get_filtered_dataset(args, transform, val_transform):
    np.random.seed(args.seed)
    trainset, valset, testset, extraset = new_get_dataset(args.data.base_dataset, transform, val_transform, root=args.data.base_root, embedding_root=args.data.embedding_root if args.model == 'MLP' else None)
    if args.data.extra_dataset and not args.eval_only:
        dataset = get_edited_dataset(args, transform)
        if args.data.subsample:
            dataset = subsample(extraset, dataset) # makesure we are sampling the same number of images as the extraset
        trainset = CombinedDataset([trainset, dataset])
        # if args.data.augmentation == 'cutmix':
        #     trainset = CutMix(trainset, num_class=len(trainset.classes), beta=1.0, prob=0.5, num_mix=2).dataset
    return trainset, valset, testset

def get_edited_dataset(args, transform, full=False):
    if type(args.data.extra_root) != str:
        print("Roots", list(args.data.extra_root))
        roots, dsets = list(args.data.extra_root), []
        for i, r in enumerate(roots):
            dsets.append(getattr(datasets.base, args.data.extra_dataset)(r, transform=transform, cfg=args, group=i))
        dataset = CombinedDataset(dsets)
    else:
        dataset = getattr(datasets.base, args.data.extra_dataset)(args.data.extra_root, transform=transform, cfg=args)

    if args.data.filter:
        path = f'{args.filter.save_dir}/{args.name}/filtered_idxs/kept.npy' if not args.filter.filtered_path else args.filter.filtered_path
        if os.path.exists(path):
            filtered_idxs = np.load(path)
        else:
            raise ValueError(f"can't find file {path}")
        print(f"Filtering kept {len(filtered_idxs)} out of {len(dataset)} images")
        dataset = Subset(dataset, filtered_idxs)

    if full or args.data.extra_dataset != 'Img2ImgDataset':
        return dataset
    
    sample_groups = {}
    for i, s in enumerate(dataset.samples):
        filename = s[0].split("/")[-1] 
        if len(filename.split('.')[0].split("-")) == 1:
            print(f"skipping {filename}")
            continue
        else:
            idx, j = filename.split('.')[0].split("-")[0], filename.split('.')[0].split("-")[1]
        if idx not in sample_groups:
            sample_groups[idx] = [(s[0], s[1], i)]
        else:
            sample_groups[idx].append([s[0], s[1], i])

    chosen_idxs = []
    for k, v in sample_groups.items():
        # randomly select one sample from each group
        chosen = np.random.choice(list(range(len(v))), replace=False)
        chosen_idxs.append(v[chosen][2])

    dataset = Subset(dataset, chosen_idxs)
    return dataset