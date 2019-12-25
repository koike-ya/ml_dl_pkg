import librosa
from scipy import signal
import torch
import numpy as np
from scipy.signal import butter, lfilter

windows = {'hamming': signal.hamming, 'hann': signal.hann, 'blackman': signal.blackman,
           'bartlett': signal.bartlett}


def to_spect(wave, sr, window_size, window_stride, window):
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)
    spect_tensor = torch.Tensor()

    # STFT
    for i in range(wave.shape[0]):
        y = wave[i].astype(float)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=windows[window])
        spect, phase = librosa.magphase(D)
        # spect = signal.spectrogram(y, nfft=n_fft, fs=eeg.sr, return_onesided=True, noverlap=eeg.sr // 2)[2]
        spect = torch.from_numpy(spect).to(torch.float32)
        spect_tensor = torch.cat((spect_tensor, spect.view(1, spect.size(0), -1)), 0)

    return spect_tensor.transpose(1, 2)


def cwt(wave, widths=np.arange(1, 31)):
    spect_tensor = torch.Tensor()

    for i in range(wave.shape[0]):
        y = wave[i].astype(float)
        cwtmatr = signal.cwt(y, signal.ricker, widths)
        cwtmatr = torch.from_numpy(cwtmatr).to(torch.float32)
        spect_tensor = torch.cat((spect_tensor, cwtmatr.view(1, cwtmatr.size(0), -1)), 0)

    return spect_tensor.transpose(1, 2)


def standardize(y):
    return (y - y.mean()).div(y.std() + 0.001)


def remove_power_noise(y, sr):
    y = bandstop_filter(y, l_cutoff=117, h_cutoff=123, sr=sr)
    y = bandstop_filter(y, l_cutoff=57, h_cutoff=63, sr=sr)
    y = highpass_filter(y, l_cutoff=1, sr=sr)
    return y


def time_and_freq_mask(data, rate):
    w = int(np.random.uniform(0, data.shape[1]))
    tau = min(data.shape[1] - w, int(abs(np.random.normal(0, data.shape[1] * rate))))
    data[:, w:w + tau] = 0

    w = int(np.random.uniform(0, data.shape[0]))
    tau = min(data.shape[0] - w, int(abs(np.random.normal(0, data.shape[0] * rate))))
    data[w:w + tau, :] = 0

    return data


def speedx(sound_array, factor):
    """ Multiplies the sound's speed by some `factor` """
    indices = np.round(np.arange(0, len(sound_array), factor))
    indices = indices[indices < len(sound_array)].astype(int)
    return sound_array[indices]


def shift_pitch(data, rate=1):
    data = librosa.effects.time_stretch(data, rate)
    y = speedx(data, rate)
    # assert data.shape[0] == y.shape[0]
    return y


def shift_gain(y, rate=0.9):
    return y * rate


# stretching the sound
def stretch(data, rate=1):
    input_length = data.shape[0]
    data = librosa.effects.time_stretch(data, rate)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data


def shift(y, n=500):
    return np.roll(y, n)


def add_muscle_noise(y, sr, rate):
    muscle_noise = np.random.uniform(0, int((y.max() - y.min()) * rate), y.shape[1])
    y += bandpass_filter(muscle_noise, l_cutoff=20, h_cutoff=60, sr=sr)
    return y


def add_eye_noise(y, sr, rate):
    muscle_noise = np.random.uniform(0, int((y.max() - y.min()) * rate), y.shape[1])
    y += bandpass_filter(muscle_noise, l_cutoff=1, h_cutoff=3, sr=sr)
    return y


def add_white_noise(y, rate):
    y += np.random.randn(y.shape[1]) * (y.max() - y.min()) * rate
    return y


def butter_filter(y, cutoff, fs, btype='lowpass', order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = lfilter(b, a, y)
    return y


def lowpass_filter(y, h_cutoff, sr):
    return butter_filter(y, h_cutoff, sr, 'lowpass', order=6)


def highpass_filter(y, l_cutoff, sr):
    return butter_filter(y, l_cutoff, sr, 'highpass', order=6)


def bandpass_filter(y, l_cutoff, h_cutoff, sr):
    y = lowpass_filter(y, h_cutoff, sr)
    y = highpass_filter(y, l_cutoff, sr)
    return y


def bandstop_filter(y, l_cutoff, h_cutoff, sr, order=6):
    nyq = 0.5 * sr
    low = l_cutoff / nyq
    high = h_cutoff / nyq

    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, y)
    return y
