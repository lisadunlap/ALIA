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
from collections import Counter

import datasets
from datasets.Waterbirds import Waterbirds
from datasets.base import *
from datasets.wilds import WILDS
from datasets.cub import Cub2011
from datasets.planes import Planes
from cutmix.cutmix import CutMix

def crop_wilds(image):
    return crop(image, 10, 0, 400, 448)

def get_train_transform(dataset_name="Imagenet", model=None, augmentation=None):
    """"
    Gets the transform for a given dataset
    """
    # any data augmentation happens here
    transform_list = []
    if augmentation == "augmix":
        print("Applying AugMix")
        transform_list.append(transforms.AugMix())
    if augmentation == "color-jitter":
        print("Applying color jitter")
        transform_list.append(transforms.ColorJitter(brightness=.5, hue=.3))
    if augmentation == "randaug":
        print("Applying RandAug augmentations")
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

    # standard preprocessing
    if model in ['RN50', 'ViT-B/32']: # if we are evaluating a clip model we use its transforms
        print("...loading CLIP model")
        net, transform = clip.load(model)
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

def get_dataset(dataset_name, transform, val_transform, root='./data', embedding_root=None):
    if dataset_name == "Waterbirds" or dataset_name == 'WaterbirdsExtra': # change these data paths
        trainset = Waterbirds(root=root, split='train', transform=transform)
        train_ids = trainset.get_subset(groups=[0,3], num_per_class=1000)
        # get every 4th idx
        train_ids = train_ids[::4]
        train_extra_ids = trainset.get_subset(groups=[1,2], num_per_class=1000)
        extra_trainset = Subset(trainset, train_extra_ids)
        trainset = Subset(trainset, train_ids) #100% biased
        valset = Waterbirds(root=root, split='val', transform=val_transform)
        idxs = valset.get_subset(groups=[0,3], num_per_class=1000)
        extra_idxs = valset.get_subset(groups=[1,2], num_per_class=1000)
        extra_valset = Subset(valset, extra_idxs)
        extraset = CombinedDataset([extra_valset, extra_trainset])
        valset = Subset(valset, idxs)
        testset = Waterbirds(root=root, split='test', transform=val_transform)
        if dataset_name == 'WaterbirdsExtra':
            trainset = CombinedDataset([trainset, extraset])
    elif dataset_name == "iWildCamMini" or dataset_name == "iWildCamMiniExtra":
        trainset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='train', transform=transform)
        valset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='val', transform=val_transform)
        testset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='test', transform=val_transform)
        extraset = WILDS(root=f'{root}/iwildcam_v2.0/train', split='train_extra', transform=transform)
        if dataset_name == 'iWildCamMiniExtra':
            trainset = CombinedDataset([trainset, extraset])
    elif dataset_name == 'Cub2011' or dataset_name == 'Cub2011Extra':
        trainset = Cub2011(root=root, subset=False, split='train', transform=transform)
        valset = Cub2011(root=root, split='val', transform=val_transform)
        extraset = Cub2011(root=root, subset=False, split='extra', transform=transform)
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

    # assert that the trainset has the attributes groups, labels, and class_names
    for var in ['groups', 'targets', 'group_names', 'class_names', 'class_weights']:
        assert all([hasattr(dataset, var) for dataset in [trainset, valset, testset]]), f"datasets missing the attribute {var}"

    return trainset, valset, testset, extraset

def get_filtered_dataset(args, transform, val_transform):
    np.random.seed(args.seed)
    trainset, valset, testset, extraset = get_dataset(args.data.base_dataset, transform, val_transform, root=args.data.base_root, embedding_root=args.data.embedding_root if args.model == 'MLP' else None)
    if args.data.extra_dataset and not args.eval_only:
        dataset = get_edited_dataset(args, transform)
        if args.data.num_extra == 'extra':
            dataset = subsample(extraset, dataset) # make sure we are sampling the same number of images as the extraset
        elif type(args.data.num_extra) == int: # randomly sample x images from the dataset
            print("sampled", args.data.num_extra, "images from the extra dataset")
            if args.data.class_balance:
                dataset = get_class_balanced_subset(dataset, args.data.num_extra // len(dataset.classes))
            else:
                dataset = Subset(dataset, np.random.choice(len(dataset), args.data.num_extra, replace=False))
        print(f"Added extra data with class counts {Counter(dataset.targets)}")
        trainset = CombinedDataset([trainset, dataset])

        if args.data.augmentation == 'cutmix': # hacky way to add cutmix augmentation
            trainset = CutMix(trainset, num_class=len(trainset.classes), beta=1.0, prob=0.5, num_mix=2).dataset
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