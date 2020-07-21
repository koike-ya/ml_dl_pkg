from enum import Enum


class SpectrogramWindow(Enum):
    hamming = 'hamming'
    hann = 'hann'
    blackman = 'blackman'
    bartlett = 'bartlett'


class TimeFrequencyFeature(Enum):
    none = ''
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


class PretrainedType(Enum):
    # TODO delete none
    none: str = ''
    resnet: str = 'resnet'
    resnet152: str = 'resnet152'
    alexnet: str = 'alexnet'
    densenet: str = 'densenet'
    wideresnet: str = 'wideresnet'
    resnext: str = 'resnext'
    resnext101: str = 'resnext101'
    vgg16: str = 'vgg16'
    vgg19: str = 'vgg19'
    googlenet: str = 'googlenet'
    mobilenet: str = 'mobilenet'
    panns: str = 'panns'
    resnext_wsl: str = 'resnext_wsl'


class ModelType(Enum):
    # TODO separate
    xgboost: str = 'xgboost'
    knn: str = 'knn'
    catboost: str = 'catboost'
    sgdc: str = 'sgdc'
    lightgbm: str = 'lightgbm'
    svm: str = 'svm'
    rf: str = 'rf'
    nb: str = 'nb'

    nn: str = 'nn'
    cnn: str = 'cnn'
    rnn: str = 'rnn'
    cnn_rnn: str = 'cnn_rnn'
    logmel_cnn: str = 'logmel_cnn'
    attention_cnn: str = 'attention_cnn'
    panns: str = 'panns'
    cnn1d_rnn: str = 'cnn1d_rnn'

    resnet: str = 'resnet'
    resnet152: str = 'resnet152'
    alexnet: str = 'alexnet'
    densenet: str = 'densenet'
    wideresnet: str = 'wideresnet'
    resnext: str = 'resnext'
    resnext101: str = 'resnext101'
    vgg16: str = 'vgg16'
    vgg19: str = 'vgg19'
    googlenet: str = 'googlenet'
    mobilenet: str = 'mobilenet'
    resnext_wsl: str = 'resnext_wsl'


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


class TrainManagerType(Enum):
    nn = 'nn'
    multitask = 'multitask'
    ml = 'ml'


class DataLoaderType(Enum):
    normal = 'normal'
    ml = 'ml'
