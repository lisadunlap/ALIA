'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch 
from sklearn.metrics import confusion_matrix
import omegaconf

import ast 

def flatten_config(dic, running_key=None, flattened_dict={}):
    for key, value in dic.items():
        if running_key is None:
            running_key_temp = key
        else:
            running_key_temp = '{}.{}'.format(running_key, key)
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            flatten_config(value, running_key_temp)
        else:
            #print(running_key_temp, value)
            flattened_dict[running_key_temp] = value
    return flattened_dict

def read_unknowns(unknown_list):
    """
    input is of form ['--METHOD.MODEL.LR=0.001758722642964502', '--METHOD.MODEL.NUM_LAYERS=1']
    """
    ret = {}
    for item in unknown_list:
        key, value = item.split('=')
        try:
            value = ast.literal_eval(value)
        except:
            print("MALFORMED ", value)
        k = key[2:]
        ret[k] = value
    return ret

def nest_dict(flat_dict, sep='.'):
    """Return nested dict by splitting the keys on a delimiter.
    >>> from pprint import pprint
    >>> pprint(nest_dict({'title': 'foo', 'author_name': 'stretch',
    ... 'author_zipcode': '06901'}))
    {'author': {'name': 'stretch', 'zipcode': '06901'}, 'title': 'foo'}
    """
    tree = {}
    for key, val in flat_dict.items():
        t = tree
        prev = None
        for part in key.split(sep):
            if prev is not None:
                t = t.setdefault(prev, {})
            prev = part
        else:
            t.setdefault(prev, val)
    return tree

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def get_per_group_acc(value, predictions, labels, groups):
    indices = np.array(np.where(groups == value))
    return np.mean((labels[indices] == predictions[indices]).astype(np.float64)) * 100.

def evaluate(predictions, labels, groups=[], label_names=None, num_augmentations=1):
    """
    Gets the evaluation metrics given the predictions and labels. 
    num_augmentations is for test-time augmentation, if its set >1, we group predictions by 
    num_augmentations and take the consesus as the label
    """
    cf_matrix = confusion_matrix(labels, predictions, labels=label_names)
    # print(np.array([d/c for d,c in zip(cf_matrix.diagonal(), cf_matrix.sum(1)) if c > 0]))

    # class_accuracy=100*cf_matrix.diagonal()/cf_matrix.sum(1)
    class_accuracy = 100 * np.array([d/c for d,c in zip(cf_matrix.diagonal(), cf_matrix.sum(1)) if c > 0])
    accuracy = np.mean((labels == predictions).astype(np.float64)) * 100.
    balanced_acc = class_accuracy.mean()
    if len(groups) == 0:
        return accuracy, balanced_acc, np.array([round(c,2) for c in class_accuracy])
    else:
        group_acc = np.array([get_per_group_acc(value, predictions, labels, groups) for value in np.unique(groups)])
        return accuracy, balanced_acc, np.array([round(c,2) for c in class_accuracy]), np.array([round(g,2) for g in group_acc])
