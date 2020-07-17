from enum import Enum


class SpectrogramWindow(Enum):
    hamming = 'hamming'
    hann = 'hann'
    blackman = 'blackman'
    bartlett = 'bartlett'


class TimeFrequencyFeature(Enum):
    spectrogram = 'spectrogram'
    logmel = 'logmel'
    scalogram = 'scalogram'


class TaskType(Enum):
    classify = 'classify'
    regress = 'regress'


class MLType(Enum):
    xgboost: str = 'xgboost'
    knn: str = 'knn'
    catboost: str = 'catboost'
    sgdc: str = 'sgdc'
    lightgbm: str = 'lightgbm'
    svm: str = 'svm'
    rf: str = 'rf'
    nb: str = 'nb'


class NNType(Enum):
    nn: str = 'nn'
    cnn: str = 'cnn'
    rnn: str = 'rnn'
    cnn_rnn: str = 'cnn_rnn'
    logmel_cnn: str = 'logmel_cnn'
    attention_cnn: str = 'attention_cnn'
    panns: str = 'panns'
    cnn1d_rnn: str = 'cnn1d_rnn'


class ModelType(Enum):
    nn: str = 'nn'
    cnn: str = 'cnn'
    rnn: str = 'rnn'
    cnn_rnn: str = 'cnn_rnn'
    logmel_cnn: str = 'logmel_cnn'
    attention_cnn: str = 'attention_cnn'
    panns: str = 'panns'
    cnn1d_rnn: str = 'cnn1d_rnn'

    xgboost: str = 'xgboost'
    knn: str = 'knn'
    catboost: str = 'catboost'
    sgdc: str = 'sgdc'
    lightgbm: str = 'lightgbm'
    svm: str = 'svm'
    rf: str = 'rf'
    nb: str = 'nb'


class RNNType(Enum):
    lstm = 'lstm'
    rnn = 'rnn'
    gru = 'gru'


class LossType(Enum):
    mse = 'mse'
    ce = 'ce'
    kl_div = 'kl_div'


class SVMKernelType(Enum):
    linear = 'linear'
    rbf = 'rbf'


class TrainManager(Enum):
    nn = 'nn'
    multitask = 'multitask'
    ml = 'ml'


class DataLoader(Enum):
    normal = 'normal'
    ml = 'ml'
