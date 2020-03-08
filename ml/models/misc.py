import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.nn_models.nn_utils import initialize_weights, init_bn
from ml.models.nn_models.stft import Spectrogram, LogmelFilterBank


class LogMel(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(LogMel, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.bn0 = nn.BatchNorm2d(mel_bins)
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        init_bn(self.bn0)

    def forward(self, input):
        """Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        return x


class Attention2d(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention2d, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        initialize_weights(self.att)
        initialize_weights(self.cla)
        self.att.weight.data.fill_(0.)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return F.sigmoid(x)+0.1

        elif activation == 'log_softmax':
            return F.log_softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, channel, time_steps, freq_bins)
        """
        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        # (samples_num, channel, time_steps * freq_bins)
        att = att.view(att.size(0), att.size(1), att.size(2) * att.size(3))
        cla = cla.view(cla.size(0), cla.size(1), cla.size(2) * cla.size(3))

        epsilon = 0.1 # 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        Return_heatmap = False
        if Return_heatmap:
            return x, norm_att
        else:
            return x
