from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.nn_models.stft import Spectrogram, LogmelFilterBank
from ml.utils.nn_config import NNModelConfig


@dataclass
class PANNsConfig(NNModelConfig):    # PANNs model arguments
    n_mels: int = 64            # Number of mel filters banks


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn14(nn.Module):
    def __init__(self, mel_bins, classes_num, checkpoint_path):
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        n_fft = 1024

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=320,
            win_length=1024, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=2000, n_fft=n_fft,
            n_mels=64, fmin=0, fmax=n_fft // 2, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, 527, bias=True)

        self.init_weight()

        if checkpoint_path:
            checkpoint = torch.load(Path(checkpoint_path).resolve(), map_location='cpu')
            self.load_state_dict(checkpoint['model'])

        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.feature_extractor = nn.ModuleList([
            self.bn0,
            *[getattr(self, f'conv_block{i}') for i in range(1, 7)],
        ])

        self.classifier = nn.ModuleList([
            self.fc1,
            self.fc_audioset
        ])

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def feature_extract(self, x):
        x = x.unsqueeze(dim=1)
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def classify(self, x):
        x = F.relu_(self.fc1(x))
        x = self.fc_audioset(x)

        return x

    def forward(self, input, feature_extract=False):
        """
        Input: (batch_size, n_mels, time_frames)"""
        x = self.feature_extract(input)

        return self.classify(x)


def construct_panns(cfg):
    device = torch.device('cuda') if cfg['cuda'] and torch.cuda.is_available() else torch.device('cpu')

    model = Cnn14(mel_bins=cfg.n_mels, classes_num=len(cfg.class_names), checkpoint_path=cfg.checkpoint_path).to(device)

    return model
