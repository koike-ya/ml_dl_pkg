import wrapper
import numpy as np
from scipy.signal import butter, lfilter
import scipy.signal
import librosa
import torch

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


def to_spect(eeg, window_size, window_stride, window):
    n_fft = int(eeg.sr * window_size)
    win_length = n_fft
    hop_length = int(eeg.sr * window_stride)
    spect_tensor = torch.Tensor()
    # STFT
    for i in range(len(eeg.channel_list)):
        y = eeg.values[i].astype(float)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=windows[window])
        spect, phase = librosa.magphase(D)
        spect = torch.FloatTensor(spect)
        spect_tensor = torch.cat((spect_tensor, spect.view(1, spect.size(0), -1)), 0)

    return spect_tensor


def butter_filter(y, cutoff, fs, btype='lowpass', order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = lfilter(b, a, y)
    return y


def bandpass_filter(h_cutoff, l_cutoff, sr, y):
    def _lowpass_filter(y):
        return butter_filter(y, h_cutoff, sr, 'lowpass', order=4)

    def _highpass_filter(y):
        return butter_filter(y, l_cutoff, sr, 'highpass', order=4)

    y = _lowpass_filter(y)
    y = _highpass_filter(y)
    return y
