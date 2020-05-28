import torch
import torch.nn as nn

from ml.models.nn_models.cnn import CNNMaker


class CNN1d(nn.Module):
    def __init__(self, cnn_1d, in_features_dict, n_classes=2):
        super(CNN1d, self).__init__()
        self.cnn_1d = cnn_1d
        self.cnn_2d, n_features = self._construct_cnn_2d(in_features_dict)
        self.fc = nn.Sequential(
            nn.Linear(n_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.predictor = nn.Linear(4096, n_classes)
        if n_classes >= 2:
            self.predictor = nn.Sequential(
                self.predictor,
                nn.Softmax(dim=-1)
            )

    def _conv_batchnorm_relu(self, in_channel, out_channel, kernel, stride=(1, 1)):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def _construct_cnn_2d(self, in_features_dict):
        layers = nn.Sequential(
            self._conv_batchnorm_relu(1, 32, (8, 8)),
            self._conv_batchnorm_relu(32, 32, (8, 8)),
            nn.MaxPool2d((5, 3), (5, 3)),
            self._conv_batchnorm_relu(32, 64, (1, 4)),
            self._conv_batchnorm_relu(64, 64, (1, 4)),
            nn.MaxPool2d((1, 2), (1, 2)),
            self._conv_batchnorm_relu(64, 128, (1, 2)),
            self._conv_batchnorm_relu(128, 128, (1, 2)),
            nn.MaxPool2d((1, 2), (1, 2)),
            self._conv_batchnorm_relu(128, 256, (1, 2)),
            self._conv_batchnorm_relu(256, 256, (1, 2)),
            nn.MaxPool2d((1, 2), (1, 2)),
        )
        # TODO debug and determine
        n_features = 15360

        return nn.Sequential(*layers), n_features

    def forward(self, x):
        x = x.unsqueeze(dim=1).unsqueeze(dim=1)
        x = self.cnn_1d(x.to(torch.float))
        x = x.transpose(1, 2)
        x = self.cnn_2d(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return self.predictor(x)


def construct_1dcnn(cfg):
    layer_info = []
    n_dim = len(cfg['cnn_kernel_sizes'][0])
    for layer in range(len(cfg['cnn_channel_list'])):
        layer_info.append((
            cfg['cnn_channel_list'][layer],
            cfg['cnn_kernel_sizes'][layer],
            cfg['cnn_stride_sizes'][layer],
            cfg['cnn_padding_sizes'][layer],
        ))

    cfg['image_size'] = [64, 260]
    cnn_maker = CNNMaker(in_channels=1, image_size=cfg['image_size'], cfg=layer_info, n_dim=n_dim,
                         n_classes=len(cfg['class_names']), use_as_extractor=True)
    layers, feature_size = cnn_maker.construct_cnn()

    model = CNN1d(layers, feature_size, n_classes=len(cfg['class_names']))
    return model
