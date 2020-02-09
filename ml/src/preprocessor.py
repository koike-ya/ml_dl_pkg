import numpy as np
import torch
from copy import deepcopy
from ml.src.signal_processor import *
from ml.models.pretrained_models import PretrainedNN, supported_pretrained_models
from sklearn import preprocessing


def preprocess_args(parser):

    prep_parser = parser.add_argument_group("Preprocess options")

    prep_parser.add_argument('--no-scaling', default=True, dest='scaling', action='store_false',
                             help='No standardization')
    prep_parser.add_argument('--augment', dest='augment', action='store_true',
                             help='Use random tempo and gain perturbations.')
    prep_parser.add_argument('--window-size', default=4.0, type=float, help='Window size for spectrogram in seconds')
    prep_parser.add_argument('--window-stride', default=2.0, type=float, help='Window stride for spectrogram in seconds')
    prep_parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
    prep_parser.add_argument('--n-mels', default=200, type=int, help='Number of mel filters banks')
    prep_parser.add_argument('--transform', choices=['spectrogram', 'scalogram', 'logmel'], default=None)
    prep_parser.add_argument('--low-cutoff', default=0.0, type=float, help='High pass filter')
    prep_parser.add_argument('--high-cutoff', default=None, type=float, help='Low pass filter')
    prep_parser.add_argument('--muscle-noise', default=0.0, type=float)
    prep_parser.add_argument('--eye-noise', default=0.0, type=float)
    prep_parser.add_argument('--white-noise', default=0.0, type=float)
    prep_parser.add_argument('--shift-gain', default=0.0, type=float)
    prep_parser.add_argument('--spec-augment', default=0.0, type=float)
    prep_parser.add_argument('--channel-wise-mean', action='store_true')
    prep_parser.add_argument('--inter-channel-mean', action='store_true')
    prep_parser.add_argument('--no-power-noise', dest='no_power_noise', action='store_true')
    prep_parser.add_argument('--mfcc', dest='mfcc', action='store_true', help='MFCC')
    prep_parser.add_argument('--fe-pretrained', default=None, choices=supported_pretrained_models,
                             help='Use NN as feature extractor')

    return parser


class Preprocessor:
    def __init__(self, cfg, phase, sr):
        self.phase = phase
        self.sr = sr
        self.l_cutoff = cfg['low_cutoff']
        self.h_cutoff = cfg['high_cutoff']
        self.transform = cfg['transform']
        self.window_stride = cfg['window_stride']
        self.window_size = cfg['window_size']
        self.window = cfg['window']
        self.normalize = cfg['scaling']
        self.cfg = cfg
        self.spec_augment = cfg['spec_augment']
        if cfg['fe_pretrained']:
            cfg_copy = cfg.copy()
            cfg_copy['model_type'] = cfg['fe_pretrained']
            self.feature_extractor = PretrainedNN(cfg_copy, len(cfg['class_names']))

    def preprocess(self, wave, label):

        if self.l_cutoff or self.h_cutoff:
            wave = bandpass_filter(wave, self.l_cutoff, self.h_cutoff, self.sr)

        if self.cfg['no_power_noise']:
            wave = remove_power_noise(wave, self.sr)

        n_channel = wave.shape[0]

        if self.cfg['channel_wise_mean']:
            diff = wave[:n_channel] - wave[:n_channel].mean(axis=0)
            wave = np.vstack((wave, diff))

        if self.cfg['inter_channel_mean']:
            diff = (wave[:n_channel].T - wave[:n_channel].mean(axis=1).T).T
            wave = np.vstack((wave, diff))

        if self.phase in ['train']:
            if self.cfg['muscle_noise']:
                wave = add_muscle_noise(wave, self.sr, self.cfg['muscle_noise'])
            if self.cfg['eye_noise']:
                wave = add_eye_noise(wave, self.sr, self.cfg['eye_noise'])
            if self.cfg['white_noise']:
                wave = add_white_noise(wave, self.cfg['white_noise'])
            if self.cfg['shift_gain']:
                rate = np.random.normal(1.0 - self.cfg['shift_gain'], 1.0 + self.cfg['shift_gain'])
                wave = shift_gain(wave, rate=rate)

            # for i in range(len(wave.channel_list)):
            #     wave[i] = shift(wave[i], self.sr * 5)
            #     wave[i] = stretch(wave[i], rate=0.3)
            #     wave[i] = shift_pitch(wave[i], rate=0.3)
        y = self.transform_(wave)    # channel x freq x time

        if self.normalize:
            y = standardize(y)

        if hasattr(self, 'feature_extractor'):
            y = self.feature_extractor.feature_extractor(y.unsqueeze(dim=0)).squeeze().detach()

        return y, label

    def mfcc(self):
        raise NotImplementedError

    def transform_(self, wave):
        if self.transform == 'spectrogram':
            y = to_spect(wave, self.sr, self.window_size, self.window_stride, self.window)  # channel x freq x time
        elif self.transform == 'scalogram':
            y = cwt(wave, widths=np.arange(1, 101), sr=self.sr)  # channel x freq x time
        elif self.transform == 'logmel':
            y = logmel(wave, self.sr, self.window_size, self.window_stride, self.window, n_mels=self.cfg['n_mels'])  # channel x freq x time
        else:
            y = torch.from_numpy(wave)

        if self.transform and self.spec_augment and self.phase in ['train']:
            y = time_and_freq_mask(y, rate=self.spec_augment)

        # print(y.size())
        # exit()

        return y
