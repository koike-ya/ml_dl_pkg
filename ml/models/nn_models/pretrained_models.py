import torch
import torch.nn as nn
from torchvision import models

from ml.models.nn_models.panns_cnn14 import construct_panns

supported_pretrained_models = {'resnet': models.resnet18, 'resnet152': models.resnet152, 'alexnet': models.alexnet,# 'densenet': models.densenet121,
                               'wideresnet': models.wide_resnet50_2, 'resnext': models.resnext50_32x4d,
                               'resnext101': models.resnext101_32x8d, 'vgg19': models.vgg19, 'vgg16': models.vgg16,
                               'googlenet': models.googlenet, 'mobilenet': None, 'panns': None, 'resnext_wsl': None}


def pretrain_args(parser):
    pretrain_parser = parser.add_argument_group("Pretrain model arguments")

    # Pretrain params
    pretrain_parser.add_argument('--pretrained', action='store_true')

    return parser


class PretrainedNN(nn.Module):
    def __init__(self, cfg, n_classes):
        super(PretrainedNN, self).__init__()
        model = self._set_model(cfg)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_extract = cfg.get('feature_extract', False)
        self.n_in_features = self._get_n_last_in_features(model)
        # TODO 直す
        if cfg['model_type'] == 'mobilenet':
            self.n_in_features *= 20
        self.predictor = nn.Linear(self.n_in_features, n_classes)
        self.batch_size = cfg['batch_size']
        if n_classes >= 2:
            self.predictor = nn.Sequential(
                self.predictor,
                nn.Softmax(dim=1)
            )

    def _set_model(self, cfg):
        if cfg['model_type'] == 'mobilenet':
            return torch.hub.load('pytorch/vision:v0.4.2', 'mobilenet_v2', pretrained=True)
        elif cfg['model_type'] == 'resnext_wsl':
            return torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        return supported_pretrained_models[cfg['model_type']](pretrained=cfg['pretrained'])

    def _get_n_last_in_features(self, model):
        if isinstance(list(model.children())[-1], nn.Sequential):
            for layer in list(model.children())[-1]:
                if hasattr(layer, 'in_features'):
                    return layer.in_features
            raise NotImplementedError
        else:
            return list(model.children())[-1].in_features

    def forward(self, x):
        if x.size(1) == 1:
            x = torch.cat([x] * 3, 1)
        x = self.feature_extractor(x)
        x = x.reshape(x.size(0), -1)
        if self.feature_extract:
            return x
        return self.predictor(x)


def construct_pretrained(cfg, n_classes):
    if cfg['model_type'] == 'panns':
        return construct_panns(cfg)
    else:
        return PretrainedNN(cfg, n_classes)
