import torch.nn as nn
from torchvision import models

supported_pretrained_models = {'resnet18': models.resnet18, 'alexnet': models.alexnet, 'densenet': models.densenet121,
                               'wideresnet': models.wide_resnet50_2, 'resnext': models.resnext50_32x4d,
                               'vgg19': models.vgg19, 'googlenet': models.googlenet}


class PretrainedNN(nn.Module):
    def __init__(self, cfg, n_classes):
        super(PretrainedNN, self).__init__()
        model = supported_pretrained_models[cfg['model_type']](pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.feature_extract = cfg.get('feature_extract', False)
        self.n_in_features = self._get_n_last_in_features(model)
        self.predictor = nn.Linear(self.n_in_features, n_classes)
        self.batch_size = cfg['batch_size']
        if n_classes >= 2:
            self.predictor = nn.Sequential(
                self.predictor,
                nn.Softmax(dim=1)
            )

    def _get_n_last_in_features(self, model):
        if isinstance(list(model.children())[-1], nn.Sequential):
            remove_layer = list(model.children())[-1][-1]
        else:
            remove_layer = list(model.children())[-1]

        return remove_layer.in_features

    def forward(self, x):
        x = self.feature_extractor(x).reshape(-1, self.n_in_features)
        if self.feature_extract:
            return x
        return self.predictor(x)


def construct_pretrained(cfg, n_classes):
    return PretrainedNN(cfg, n_classes)
