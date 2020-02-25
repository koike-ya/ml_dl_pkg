import torch
import torch.nn as nn
import torch.nn.functional as F
from ml.models.misc import LogMel
from ml.models.nn_models.nn_utils import initialize_weights, init_bn


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
        initialize_weights(self.conv1)
        initialize_weights(self.conv2)
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
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num):
        super(Cnn14, self).__init__()
        self.logmel_extractor = LogMel(sample_rate, window_size, hop_size, mel_bins, fmin, fmax)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        initialize_weights(self.fc1)
        initialize_weights(self.fc_audioset)

    def forward(self, input, extract=False):
        """
        Input: (batch_size, data_length)"""
        x = self.logmel_extractor(input)

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

        x = F.relu_(self.fc1(x))
        if extract:
            embedding = F.dropout(x, p=0.5, training=self.training)
            return embedding
        x = torch.sigmoid(self.fc_audioset(x))
        return x


def construct_logmel_cnn(cfg):
    sample_rate = cfg['sample_rate']
    classes_num = len(cfg['class_names'])

    window_size = cfg['window_size'] * sample_rate
    hop_size = cfg['window_stride'] * sample_rate
    mel_bins = cfg['n_mels']
    fmin = cfg['low_cutoff']
    fmax = cfg['high_cutoff']
    checkpoint_path = cfg['checkpoint_path']
    device = torch.device('cuda') if cfg['cuda'] and torch.cuda.is_available() else torch.device('cpu')

    model = Cnn14(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size, mel_bins=mel_bins,
                  fmin=fmin, fmax=fmax, classes_num=classes_num).to(device)

    if checkpoint_path:
        model.fc_audioset = nn.Linear(2048, 527, bias=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    model.fc_audioset = nn.Linear(2048, len(cfg['class_names']), bias=True)

    return model
