from .pretrained import *
import torch.nn as nn
import torch.nn.functional as nnf

def MLP(num_classes):
    return nn.Sequential(
        nn.Linear(768, 384),
        nn.ReLU(),
        nn.Linear(384, num_classes)
    )