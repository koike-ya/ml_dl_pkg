import torch
import torch.nn as nn
from torchvision import models
from ml.models.panns_cnn14 import construct_panns


supported_pretrained_models = {'resnet': models.resnet18, 'alexnet': models.alexnet,# 'densenet': models.densenet121,
                               'wideresnet': models.wide_resnet50_2, 'resnext': models.resnext50_32x4d,
                               'vgg': models.vgg19, 'googlenet': models.googlenet, 'mobilenet': None,
                               'panns': None}


class PretrainedNN(nn.Module):
    def __init__(self, cfg, n_classes):
        super(PretrainedNN, self).__init__()
        model = self._set_model(cfg)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_extract = cfg.get('feature_extract', False)
        self.n_in_features = self._get_n_last_in_features(model)
        # TODO 直す
        if cfg['model_type'] == 'mobilenet':
            self.n_in_features *= 7 * 7
        self.predictor = nn.Linear(self.n_in_features, n_classes)
        self.batch_size = cfg['batch_size']
        if n_classes >= 2:
            self.predictor = nn.Sequential(
                self.predictor,
                nn.Softmax(dim=1)
            )

    def _set_model(self, cfg):
        if cfg['model_type'] in ['mobilenet']:
            return torch.hub.load('pytorch/vision:v0.4.2', 'mobilenet_v2', pretrained=True)
        return supported_pretrained_models[cfg['model_type']](pretrained=True)

    def _get_n_last_in_features(self, model):
        if isinstance(list(model.children())[-1], nn.Sequential):
            for layer in list(model.children())[-1]:
                if hasattr(layer, 'in_features'):
                    return layer.in_features
            raise NotImplementedError
        else:
            return list(model.children())[-1].in_features

    def forward(self, x):
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
