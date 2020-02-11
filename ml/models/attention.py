from torch import nn
import torch
from torch.functional import F
from ml.models.nn_utils import initialize_weights, init_bn
from ml.models.misc import Attention2d, LogMel


class EmbeddingLayers_pooling(nn.Module):
    def __init__(self):
        super(EmbeddingLayers_pooling, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=1,
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=2,
                               padding=(4, 4), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=4,
                               padding=(8, 8), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=8,
                               padding=(16, 16), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.init_weights()

    def init_weights(self):
        initialize_weights(self.conv1)
        initialize_weights(self.conv2)
        initialize_weights(self.conv3)
        initialize_weights(self.conv4)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input, return_layers=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        return x


class CnnPooling(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, pooling='attention'):
        super(CnnPooling, self).__init__()

        self.logmel_extractor = LogMel(sample_rate, window_size, hop_size, mel_bins, fmin, fmax)

        self.emb = EmbeddingLayers_pooling()
        self.attention = Attention2d(
            512,
            classes_num,
            att_activation='sigmoid',
            cla_activation='log_softmax')
        self.fc_final = nn.Linear(512, classes_num)
        self.pooling = pooling

    def init_weights(self):
        initialize_weights(self.fc_final)
        
    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
        x = self.emb(input)

        if self.pooling == 'attention':
            output = self.attention(x)
        elif self.pooling == 'average':
            x = F.avg_pool2d(x, kernel_size=x.shape[2:])
            x = x.view(x.shape[0:2])
            output = F.log_softmax(self.fc_final(x), dim=-1)
        else:
            x = F.max_pool2d(x, kernel_size=x.shape[2:])
            x = x.view(x.shape[0:2])

            output = F.log_softmax(self.fc_final(x), dim=-1)

        return output


def construct_attention_cnn(cfg):
    sample_rate = cfg['sample_rate']
    classes_num = len(cfg['class_names'])

    window_size = cfg['window_size'] * sample_rate
    hop_size = cfg['window_stride'] * sample_rate
    mel_bins = cfg['n_mels']
    fmin = cfg['low_cutoff']
    fmax = cfg['high_cutoff']
    device = torch.device('cuda') if cfg['cuda'] and torch.cuda.is_available() else torch.device('cpu')

    model = CnnPooling(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size, mel_bins=mel_bins,
                           fmin=fmin, fmax=fmax, classes_num=classes_num, pooling=cfg['pooling']).to(device)

    return model
