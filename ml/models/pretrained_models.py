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


supported_pretrained_models = {'resnet18': models.resnet18, 'alexnet': models.alexnet, 'densenet': models.densenet121,
                               'wideresnet': models.wide_resnet50_2, 'inception_v3': models.inception_v3,
                               'resnext': models.resnext50_32x4d, 'vgg19': models.vgg19, 'googlenet': models.googlenet,
                               }


def construct_pretrained(cfg, n_classes):
    model = supported_pretrained_models[cfg['model_type']](pretrained=True)
    if cfg['model_type'] in ['resnet18']:
        num_ftrs = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, n_classes),
            nn.Softmax(dim=1),
        )
    elif cfg['model_type'] in ['alexnet']:
        num_ftrs = model.classifier[6].in_features

        model.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, n_classes),
            nn.Softmax(dim=1),
        )


    return model