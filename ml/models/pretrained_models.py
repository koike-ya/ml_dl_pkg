import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


supported_pretrained_models = {'resnet18': models.resnet18}


def construct_pretrained(cfg, n_classes):
    model = supported_pretrained_models[cfg['model_type']](pretrained=True)
    print(model)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, n_classes),
        nn.Softmax(dim=1),
    )

    return model