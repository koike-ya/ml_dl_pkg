from ml.models.nn_models.pretrained_models import PretrainedNN, supported_pretrained_models
from ml.preprocess.augment import SpecAugment
from ml.preprocess.logmel import LogMel
from ml.preprocess.signal_processor import *


def preprocess_args(parser):

    prep_parser = parser.add_argument_group("Preprocess options")

    prep_parser.add_argument('--no-scaling', default=True, dest='scaling', action='store_false',
                             help='No standardization')
    prep_parser.add_argument('--augment', dest='augment', action='store_true',
                             help='Use random tempo and gain perturbations.')
    prep_parser.add_argument('--sample-rate', default=500.0, type=float)
    prep_parser.add_argument('--window-size', default=4.0, type=float, help='Window size for spectrogram in seconds')
    prep_parser.add_argument('--window-stride', default=2.0, type=float, help='Window stride for spectrogram in seconds')
    prep_parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
    prep_parser.add_argument('--n-mels', default=64, type=int, help='Number of mel filters banks')
    prep_parser.add_argument('--transform', choices=['spectrogram', 'scalogram', 'logmel'], default=None)
    prep_parser.add_argument('--low-cutoff', default=0.0, type=float, help='High pass filter')
    prep_parser.add_argument('--high-cutoff', default=None, type=float, help='Low pass filter')
    prep_parser.add_argument('--muscle-noise', default=0.0, type=float)
    prep_parser.add_argument('--eye-noise', default=0.0, type=float)
    prep_parser.add_argument('--white-noise', default=0.0, type=float)
    prep_parser.add_argument('--section-rate', default=0.0, type=float)
    prep_parser.add_argument('--shift-gain', default=0.0, type=float)
    prep_parser.add_argument('--spec-augment', default=0.0, type=float)
    prep_parser.add_argument('--channel-wise-mean', action='store_true')
    prep_parser.add_argument('--inter-channel-mean', action='store_true')
    prep_parser.add_argument('--remove-power-noise', dest='remove_power_noise', action='store_true')
    prep_parser.add_argument('--mfcc', dest='mfcc', action='store_true', help='MFCC')
    prep_parser.add_argument('--fe-pretrained', default=None, choices=supported_pretrained_models,
                             help='Use NN as feature extractor')

    return parser


from dataclasses import dataclass
from ml.utils.enums import SpectrogramWindow, TimeFrequencyFeature, PretrainedType


@dataclass
class TransConfig:
    cuda: bool = True
    scaling: bool = False     # scaling
    augment: bool = False       # Use random tempo and gain perturbations.
    sample_rate: float = 500.0  # The sample rate for the data/model features
    window_size: float = 4.0    # Window size for spectrogram in seconds
    window_stride: float = 2.0  # Window stride for spectrogram in seconds
    window: SpectrogramWindow = SpectrogramWindow.hamming   # Window type for spectrogram generation
    n_mels: int = 64            # Number of mel filters banks
    transform: TimeFrequencyFeature = TimeFrequencyFeature.none
    low_cutoff: float = 0.0  # High pass filter
    high_cutoff: float = 0.0  # Low pass filter
    # TODO refactor below
    spec_augment: float = 0.0
    fe_pretrained: PretrainedType = PretrainedType.none
    remove_power_noise: bool = False
    channel_wise_mean: bool = False
    inter_channel_mean: bool = False
    muscle_noise: bool = False
    eye_noise: bool = False
    white_noise: bool = False
    shift_gain: bool = False


class DeepSELFTransformer:    # TODO make transformer.py
    pass


class Preprocessor:
    def __init__(self, cfg, phase):
        self.phase = phase
        self.sr = cfg['sample_rate']
        self.l_cutoff = cfg['low_cutoff']
        self.h_cutoff = cfg['high_cutoff']
        self.transform = cfg['transform'].value
        self.window_stride = cfg['window_stride']
        self.window_size = cfg['window_size']
        self.window = cfg['window']
        self.normalize = cfg['scaling']
        self.cfg = cfg
        self.spec_augment = cfg['spec_augment']
        self.device = torch.device('cuda') if cfg['cuda'] and torch.cuda.is_available() else torch.device('cpu')
        if cfg['fe_pretrained'] and cfg['fe_pretrained'].value:
            cfg_copy = cfg.copy()
            cfg_copy['model_type'] = cfg['fe_pretrained']
            self.feature_extractor = PretrainedNN(cfg_copy, len(cfg['class_names']))

    def preprocess(self, wave):

        if self.l_cutoff or self.h_cutoff:
            wave = bandpass_filter(wave, self.l_cutoff, self.h_cutoff, self.sr)

        if self.cfg['remove_power_noise']:
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
                wave = add_muscle_noise(wave, self.sr, self.cfg['muscle_noise'], self.cfg['section_rate'])
            if self.cfg['eye_noise']:
                wave = add_eye_noise(wave, self.sr, self.cfg['eye_noise'], self.cfg['section_rate'])
            if self.cfg['white_noise']:
                wave = add_white_noise(wave, self.cfg['white_noise'], self.cfg['section_rate'])
            if self.cfg['shift_gain']:
                rate = np.random.normal(1.0 - self.cfg['shift_gain'], 1.0 + self.cfg['shift_gain'])
                wave = shift_gain(wave, rate=rate)

            # for i in range(len(wave.channel_list)):
            #     wave[i] = shift(wave[i], self.sr * 5)
            #     wave[i] = stretch(wave[i], rate=0.3)
            #     wave[i] = shift_pitch(wave[i], rate=0.3)
        y = torch.from_numpy(wave).float()
        if self.transform:
            y = self.transform_(y)    # channel x freq x time

        if self.normalize:
            y = standardize(y)

        if hasattr(self, 'feature_extractor'):
            y = self.feature_extractor.feature_extractor(y.unsqueeze(dim=0)).squeeze().detach()

        return y

    def mfcc(self):
        raise NotImplementedError

    def transform_(self, wave):
        if self.transform == 'spectrogram':
            freq_time = to_spect(wave, self.sr, self.window_size, self.window_stride, self.window)  # channel x freq x time
        elif self.transform == 'scalogram':
            freq_time = cwt(wave, widths=np.arange(1, 101), sr=self.sr)  # channel x freq x time
        elif self.transform == 'logmel':
            wave = wave[None, :]
            freq_time = LogMel(self.sr, self.window_size, self.window_stride, self.cfg['n_mels'],
                               self.l_cutoff, self.h_cutoff, self.device)(wave.to(torch.float32).to(self.device))  # channel x freq x time
        else:
            raise NotImplementedError

        if self.spec_augment and self.phase in ['train']:
            spec_augmentor = SpecAugment(time_drop_rate=self.cfg['time_drop_rate'], freq_drop_rate=self.cfg['freq_drop_rate'])
            freq_time = spec_augmentor(freq_time)

        return freq_time
