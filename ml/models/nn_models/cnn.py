import torch

seed = 0
torch.manual_seed(seed)
import math

torch.cuda.manual_seed_all(seed)
import random
random.seed(seed)
import torch.nn as nn

from dataclasses import dataclass, field
from typing import List
from ml.utils.nn_config import NNModelConfig


@dataclass
class CNNConfig(NNModelConfig):    # CNN model arguments
    channel_list: List = field(default_factory=lambda: [4, 8, 16])
    kernel_sizes: List = field(default_factory=lambda: [(4, 4), (4, 4), (4, 4)])
    stride_sizes: List = field(default_factory=lambda: [(2, 2), (2, 2), (2, 2)])
    padding_sizes: List = field(default_factory=lambda: [(1, 1), (1, 1), (1, 1)])


class CNN(nn.Module):
    def __init__(self, feature_extractor, in_features_dict, n_classes=2, feature_extract=False, dim=2):
        super(CNN, self).__init__()
        self.feature_extractor = feature_extractor
        in_features = in_features_dict['n_channels'] * in_features_dict['height'] * in_features_dict['width']
        out_features = in_features // 2
        self.fc = nn.Sequential(
            # nn.Linear(in_features, in_features),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.n_dim = dim
        self.feature_extract = feature_extract
        self.predictor = nn.Linear(out_features, n_classes)
        if n_classes >= 2:
            self.predictor = nn.Sequential(
                self.predictor,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        # print(x.size())
        # if self.n_dim == 1:
        #     x = torch.unsqueeze(x, dim=1)
        x = self.feature_extractor(x.to(torch.float))
        x = x.view(x.size(0), -1)

        if self.feature_extract:
            return x

        x = self.fc(x)
        return self.predictor(x)


class CNNMaker:
    def __init__(self, in_channels, image_size, cfg, n_classes, n_dim, use_as_extractor=False):
        self.in_channels = in_channels
        self.image_size = list(image_size)
        self.cfg = cfg
        self.n_classes = n_classes
        self.n_dim = n_dim
        self.use_as_extractor = use_as_extractor

    def construct_cnn(self):
        layers = self.make_layers()
        feature_size = self.calc_feature_size()
        if self.use_as_extractor:
            return layers, feature_size

        model = CNN(layers, feature_size, n_classes=self.n_classes, dim=self.n_dim)

        return model

    def make_layers(self):
        cnn_classes = ['conv_cls', 'max_pool_cls', 'batch_norm_cls']
        cnn_set = {
            1: dict(zip(cnn_classes, [nn.Conv1d, nn.MaxPool1d, nn.BatchNorm1d])),
            2: dict(zip(cnn_classes, [nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d])),
            3: dict(zip(cnn_classes, [nn.Conv3d, nn.MaxPool3d, nn.BatchNorm3d]))
        }

        layers = []
        n_channels = self.in_channels
        for i, (channel, kernel_size, stride, padding) in enumerate(self.cfg):
            if channel == 'M':
                layers += [cnn_set[self.n_dim]['max_pool_cls'](kernel_size, stride, padding)]
            else:
                conv = cnn_set[self.n_dim]['conv_cls'](n_channels, channel, kernel_size, stride, padding)
                layers += [conv, cnn_set[self.n_dim]['batch_norm_cls'](channel), nn.ReLU(inplace=True)]
            n_channels = channel

        return nn.Sequential(*layers)

    def calc_feature_size(self):
        """

        :param self.cfg:
        :param feature_size: tuple containing # of rows and # of columns of input data
        :param self.n_dim:
        :param last_shape:
        :return: int型 (numpy.int型だとRNNの入力数指定のときにエラーになる)
        """
        feature_shape = self.image_size.copy()
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        for dim in range(self.n_dim):  # height and width if dim=2
            for layer in self.cfg:
                channel, kernel, stride, padding = layer
                feature_shape[dim] = int(math.floor(
                    feature_shape[dim] + 2 * padding[dim] - kernel[dim]) / stride[dim] + 1)

        if len(feature_shape) == 1:
            feature_shape.append(1)

        return {
            'n_channels': self.cfg[-1][0],
            'height': int(feature_shape[0]),
            'width': int(feature_shape[1])}


def construct_cnn(cfg, use_as_extractor=False):
    layer_info = []
    n_dim = len(cfg.kernel_sizes[0])
    for layer in range(len(cfg.channel_list)):
        layer_info.append((
            cfg.channel_list[layer],
            list(cfg.kernel_sizes[layer]),
            list(cfg.stride_sizes[layer]),
            list(cfg.padding_sizes[layer]),
        ))
    cnn_maker = CNNMaker(in_channels=cfg.in_channels, image_size=cfg.image_size, cfg=layer_info, n_dim=n_dim,
                         n_classes=len(cfg.class_names), use_as_extractor=use_as_extractor)
    return cnn_maker.construct_cnn()
