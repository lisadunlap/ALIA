from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import wandb
import clip
import numpy as np
import collections 
import random
from omegaconf import OmegaConf

# from models import *
import datasets
import models
from utils import evaluate, read_unknowns, nest_dict, flatten_config
# from wandb_utils import WandbData
from helpers.load_dataset import  get_train_transform, get_filtered_dataset, get_val_transform
from datasets.wilds import wilds_eval

parser = argparse.ArgumentParser(description='Dataset Understanding')
parser.add_argument('--config', default='configs/base.yaml', help="config file")
parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                "(use dots for.nested=overrides)")
flags, unknown = parser.parse_known_args()

overrides = OmegaConf.from_cli(flags.overrides)
cfg       = OmegaConf.load(flags.config)
base      = OmegaConf.load('configs/base.yaml')
dataset_base = OmegaConf.load(cfg.base_config)
args      = OmegaConf.merge(base, dataset_base, cfg, overrides)
if len(unknown) > 0:
    print(unknown)
    config = nest_dict(read_unknowns(unknown))
    to_merge = OmegaConf.create(config)
    args = OmegaConf.merge(args, to_merge)
args.yaml = flags.config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.wandb_silent:
    os.environ['WANDB_SILENT']="true"

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

augmentation = 'none' if not args.data.augmentation else args.data.augmentation
augmentation = f'{augmentation}-filtered' if args.data.filter else f'augmentation-unfiltered'
ckpt_name = f'checkpoint/ckpt-{args.name}-{augmentation}-{args.model}-{args.seed}-{args.hps.lr}-{args.hps.weight_decay}'
if args.data.num_extra != 'extra':
    ckpt_name += f'-{args.data.num_extra}'

# Data
print('==> Preparing data..')
transform = get_train_transform(args.data.base_dataset, model=args.model, augmentation=args.data.augmentation)
val_transform = get_val_transform(args.data.base_dataset, model=args.model)
# trainset, valset, testset = get_dataset(args.data.base_dataset, transform)
trainset, valset, testset = get_filtered_dataset(args, transform, val_transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.data.batch, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.data.batch, shuffle=False, num_workers=1)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.data.batch, shuffle=False, num_workers=1)

# Model
print('==> Building model..')
net = getattr(models, args.model)(num_classes = len(trainset.classes))
if args.finetune:
    print("...finetuning")
    # freeze all bust last layer
    for name, param in net.named_parameters(): 
        if 'fc' not in name:
            param.requires_grad = False

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
run = wandb.init(project=args.proj, group=args.name, config=flatten_config(args))
# logger = WandbData(run, testset, args, [s[0] for s in testset.samples], incorrect_only=args.incorrect_only)
wandb.summary['train_size'] = len(trainset)

def load_checkpoint(args, net, optimizer):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.checkpoint_name:
        checkpoint_name = f'./checkpoint/{args.checkpoint_name}'
    else:
        assert os.path.exists(ckpt_name)
        checkpoint_name = os.path.join(ckpt_name, 'best.pth')
    checkpoint = torch.load(checkpoint_name)

    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['net'].items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v
    
    print(f"Loaded checkpoint at epoch {checkpoint['epoch']} from {checkpoint_name}")
    # net.load_state_dict(checkpoint['net'])
    net.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optim'])

    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return net, optimizer, best_acc, start_epoch

print("num samples per group:", collections.Counter(trainset.groups))
print("Weights: ", trainset.class_weights)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(trainset.class_weights).to(device))
optimizer = optim.SGD(net.parameters(), lr=args.hps.lr,
                      momentum=0.9, weight_decay=args.hps.weight_decay)

if args.hps.lr_scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
elif args.hps.lr_scheduler == 'custom':
    scheduler0 = torch.optim.lr_scheduler.LinearLR(optimizer, 
                     start_factor = 0.008, # The number we multiply learning rate in the first epoch
                     total_iters = 4,) # The number of iterations that multiplicative factor reaches to 1
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                            milestones=[30, 60, 80], # List of epoch indices
                            gamma =0.1) # Multiplicative factor of learning rate decay
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler0, scheduler1])
elif args.hps.lr_scheduler == 'finetune':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//2, 3*args.epochs//4], gamma=0.1)
else:
    raise ValueError("Unknown scheduler")

if args.resume or args.eval_only:
    net, optimizer, best_acc, start_epoch = load_checkpoint(args, net, optimizer)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, groups) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    wandb.log({'train loss': train_loss/(batch_idx+1), 'train acc': 100.*correct/total, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]})


def test(epoch, loader, phase='val'):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    all_targets, all_predictions, all_groups = np.array([]), np.array([]), np.array([])
    with torch.no_grad():
        for batch_idx, (inputs, targets, groups) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
        
            try:
                loss = criterion(outputs, targets)
                test_loss += loss.item()
            except:
                print(targets)
                raise ValueError("Loss is nan")
            _, predicted = outputs.max(1)

            all_targets = np.append(all_targets, targets.cpu().numpy())
            all_predictions = np.append(all_predictions, predicted.cpu().numpy())
            all_groups = np.append(all_groups, groups.cpu().numpy())

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # get per class and per group accuracies
        acc, class_balanced_acc, class_acc, group_acc = evaluate(all_predictions, all_targets, all_groups)
        metrics = {"epoch": epoch, f'{phase} acc': 100.*correct/total, f'{phase} accuracy': acc, f"{phase} class accuracy": class_acc, f"{phase} balanced accuracy": class_balanced_acc, **{f"{phase} {loader.dataset.group_names[i]} acc": group_acc[i] for i in range(len(group_acc))}}
        if 'iWildCam' in args.data.base_dataset:
            wilds_metrics, _ = wilds_eval(torch.tensor(all_predictions), torch.tensor(all_targets))
            metrics.update(wilds_metrics)

        wandb.log(metrics)
        print("group acc", group_acc)

    # Save checkpoint.
    acc = 100.*correct/total if 'iWildCam' not in args.data.base_dataset else wilds_metrics['F1-macro_all']
    if acc > best_acc:
        if not args.eval_only or phase == 'val':
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optim': optimizer.state_dict(),
            }
            
            if not os.path.exists(ckpt_name):
                os.makedirs(ckpt_name)

            if args.checkpoint_name:
                torch.save(state, f'./checkpoint/{args.checkpoint_name}.pth')
                wandb.save(f'./checkpoint/{args.checkpoint_name}.pth')
            else:
                torch.save(state, f'./{ckpt_name}/best.pth')
                wandb.save(f'./{ckpt_name}/best.pth')
        best_acc = acc
        wandb.summary['best epoch'] = epoch
        wandb.summary['best val acc'] = best_acc
        wandb.summary['best group acc'] = group_acc
        wandb.summary['best balanced acc'] = class_balanced_acc
        wandb.summary['best class acc'] = class_acc
    if not args.eval_only and epoch % 10 == 0:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optim': optimizer.state_dict()
        }
        torch.save(state, f'./{ckpt_name}/epoch-{epoch}.pth')
        wandb.save(f'./{ckpt_name}/epoch-{epoch}.pth')

if args.eval_only:
    test(start_epoch, trainloader, phase='train_eval')
    test(start_epoch, testloader, phase='test')
else:
    for epoch in range(start_epoch, args.epochs):
        train(epoch)
        test(epoch, valloader, phase='val')
        scheduler.step()
        if epoch % 10 == 0:
            test(epoch, testloader, phase='test')
    # load the best checkpoint
    print('==> Loading best checkpoint..')
    net, optimizer, best_acc, start_epoch = load_checkpoint(args, net, optimizer)
    test(epoch, testloader, phase='test')