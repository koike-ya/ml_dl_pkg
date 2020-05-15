import numpy as np
import torch
from sklearn import preprocessing

from ml.src import to_spect


class Preprocessor:
    def __init__(self, eeg_conf, normalize=False, augment=False, to_1d=False, scaling_axis=None):
        self.sr = eeg_conf['sample_rate']
        self.l_cutoff = eeg_conf['low_cutoff']
        self.h_cutoff = eeg_conf['high_cutoff']
        self.spect = eeg_conf['spect']
        if self.spect:
            self.window_stride = eeg_conf['window_stride']
            self.window_size = eeg_conf['window_size']
            self.window = eeg_conf['window']
        self.normalize = normalize
        self.augment = augment
        self.to_1d = to_1d
        self.time_corr = True
        self.freq_corr = True
        self.eig_values = True
        self.scaling_axis = scaling_axis

    def _calc_correlation(self, matrix):
        if self.scaling_axis:
            matrix = preprocessing.scale(matrix, axis=self.scaling_axis)

        return to_correlation_matrix(matrix)

    def calc_corr_frts(self, eeg, space='time'):
        if space == 'time':
            corr_matrix = self._calc_correlation(eeg.values)
            y = flatten_corr_upper_right(corr_matrix)
        if space == 'freq':
            corr_matrix = self._calc_correlation(np.absolute(np.fft.rfft(eeg.values, axis=1)))
            y = flatten_corr_upper_right(corr_matrix)
        if self.eig_values:
            y = np.hstack((y, calc_eigen_values_sorted(corr_matrix)))
        return y

    def preprocess(self, eeg):
        if self.sr != eeg.sr:
            eeg.values = eeg.resample(self.sr)
            eeg.sr = self.sr

        if self.augment:
            raise NotImplementedError

        # Filtering
        # eeg.values = self.bandpass_filter(eeg.values)

        if self.to_1d:
            y = np.array([])
            if self.time_corr:
                y = np.hstack((y, self.calc_corr_frts(eeg, 'time')))
            if self.freq_corr:
                y = np.hstack((y, self.calc_corr_frts(eeg, 'freq')))
        elif self.spect:
            y = to_spect(eeg, self.window_size, self.window_stride, self.window)
        else:
            y = torch.FloatTensor(eeg.values).view(1, eeg.values.shape[0], eeg.values.shape[1])

        if self.normalize:
            # TODO Feature(time) axis normalization, Index(channel) axis normalization
            y = (y - y.mean()).div(y.std())

        return y

    def mfcc(self):
        raise NotImplementedError


def to_correlation_matrix(waves):
    return np.corrcoef(waves)


def calc_eigen_values_sorted(matrix):
    if np.any(np.isnan(matrix)):
        a = ''
    w, v = np.linalg.eig(matrix)
    w = np.absolute(w)
    w.sort()
    return w


# Take the upper right triangle of a matrix
def flatten_corr_upper_right(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])

    return np.array(accum)
