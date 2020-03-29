import torch
import torch.nn as nn

from ml.models.nn_models.nn_utils import init_bn
from ml.models.nn_models.stft import Spectrogram, LogmelFilterBank


class LogMel:
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        super(LogMel, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        window_size = int(window_size * sample_rate)
        hop_size = int(hop_size * sample_rate)

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

    def __call__(self, signals):
        """Input: (n_channels, data_length)"""
        logmeled = torch.Tensor([]).to(signals.device)
        for idx_channel in range(signals.size(0)):
            x = signals[None, idx_channel, :]   # (1, 1, data_length)
            x = self.spectrogram_extractor(x)  # (1, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (1, 1, time_steps, mel_bins)
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            logmeled = torch.cat([logmeled, x], dim=1)

        return logmeled.detach().squeeze(dim=0)
