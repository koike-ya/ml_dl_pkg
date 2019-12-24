import torch.nn as nn
from torchvision import models

supported_pretrained_models = {'resnet18': models.resnet18, 'alexnet': models.alexnet, 'densenet': models.densenet121,
                               'wideresnet': models.wide_resnet50_2, 'resnext': models.resnext50_32x4d,
                               'vgg19': models.vgg19, 'googlenet': models.googlenet}


def softmax_fc(n_features, out_class):
    return nn.Sequential(
        nn.Linear(n_features, out_class),
        nn.Softmax(dim=1)
    )


def construct_pretrained(cfg, n_classes):
    model = supported_pretrained_models[cfg['model_type']](pretrained=True)
    if cfg['model_type'] in ['resnet18', 'resnext', 'wideresnet', 'googlenet']:
        num_ftrs = model.fc.in_features
        model.fc = softmax_fc(num_ftrs, n_classes)

    elif cfg['model_type'] in ['alexnet', 'vgg19']:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = softmax_fc(num_ftrs, n_classes)

    elif cfg['model_type'] in ['densenet']:
        num_ftrs = model.classifier.in_features
        model.classifier = softmax_fc(num_ftrs, n_classes)

    else:
        num_ftrs = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, n_classes),
            nn.Softmax(dim=1),
        )

    return model