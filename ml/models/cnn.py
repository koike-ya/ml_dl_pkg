import torch

seed = 0
torch.manual_seed(seed)
import math

torch.cuda.manual_seed_all(seed)
import random
random.seed(seed)
import torch.nn as nn

from ml.models.stft import Spectrogram, LogmelFilterBank


def type_int_list_list(args):
    return [list(map(int, arg.split('-'))) for arg in args.split(',')]


def type_int_list(args):
    return list(map(int, args.split(',')))


def cnn_args(parser):
    cnn_parser = parser.add_argument_group("CNN model arguments")

    # cnn params
    cnn_parser.add_argument('--cnn-channel-list', default='8,32', type=type_int_list)
    cnn_parser.add_argument('--cnn-kernel-sizes', default='4-4,4-4', type=type_int_list_list)
    cnn_parser.add_argument('--cnn-stride-sizes', default='2-2,2-2', type=type_int_list_list)
    cnn_parser.add_argument('--cnn-padding-sizes', default='1-1,1-1', type=type_int_list_list)

    # parser = logmel_cnn_args(parser)

    return parser


def logmel_cnn_args(parser):
    logmel_cnn_parser = parser.add_argument_group("LogMelCNN model arguments")

    # logmel cnn params
    logmel_cnn_parser.add_argument('--window-size', default=4.0, type=float, help='Window size for spectrogram in seconds')
    logmel_cnn_parser.add_argument('--window-stride', default=2.0, type=float,
                             help='Window stride for spectrogram in seconds')
    logmel_cnn_parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
    logmel_cnn_parser.add_argument('--n-mels', default=200, type=int, help='Number of mel filters banks')

    return parser


class CNN(nn.Module):
    def __init__(self, feature_extractor, in_features_dict, n_classes=2, feature_extract=False, dim=2):
        super(CNN, self).__init__()
        self.feature_extractor = feature_extractor
        in_features = in_features_dict['n_channels'] * in_features_dict['height'] * in_features_dict['width']
        out_features = 2048
        self.fc = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.n_dim = dim
        self.feature_extract = feature_extract
        self.predictor = nn.Linear(out_features, n_classes)
        if n_classes >= 2:
            self.predictor = nn.Sequential(
                self.predictor,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        if self.n_dim == 3:
            x = torch.unsqueeze(x, dim=1)
        x = self.feature_extractor(x.to(torch.float))
        x = x.view(x.size(0), -1)

        if self.feature_extract:
            return x

        x = self.fc(x)
        return self.predictor(x)


class LogMelCNN(nn.Module):
    def __init__(self, in_features, sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                 n_classes=2, feature_extract=False):
        super(LogMelCNN, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
                                                 top_db=top_db,
                                                 freeze_parameters=True)

    def forward(x):
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class CNNMaker:
    def __init__(self, in_channels, image_size, cfg, n_classes, n_dim, use_as_extractor=False):
        self.in_channels = in_channels
        self.image_size = list(image_size)
        self.cfg = cfg
        self.n_classes = n_classes
        self.n_dim = n_dim
        self.use_as_extractor = use_as_extractor

    def construct_cnn(self):
        layers = self.make_layers()
        feature_size = self.calc_feature_size()
        if self.use_as_extractor:
            return layers, feature_size

        model = CNN(layers, feature_size, n_classes=self.n_classes)

        return model

    def make_layers(self):
        cnn_classes = ['conv_cls', 'max_pool_cls', 'batch_norm_cls']
        cnn_set = {
            1: dict(zip(cnn_classes, [nn.Conv1d, nn.MaxPool1d, nn.BatchNorm1d])),
            2: dict(zip(cnn_classes, [nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d])),
            3: dict(zip(cnn_classes, [nn.Conv3d, nn.MaxPool3d, nn.BatchNorm3d]))
        }

        layers = []
        n_channels = self.in_channels
        for i, (channel, kernel_size, stride, padding) in enumerate(self.cfg):
            if channel == 'M':
                layers += [cnn_set[self.n_dim]['max_pool_cls'](kernel_size, stride, padding)]
            else:
                conv = cnn_set[self.n_dim]['conv_cls'](n_channels, channel, kernel_size, stride, padding)
                layers += [conv, cnn_set[self.n_dim]['batch_norm_cls'](channel), nn.ReLU(inplace=True)]
            n_channels = channel

        return nn.Sequential(*layers)

    def calc_feature_size(self):
        """

        :param self.cfg:
        :param feature_size: tuple containing # of rows and # of columns of input data
        :param self.n_dim:
        :param last_shape:
        :return: int型 (numpy.int型だとRNNの入力数指定のときにエラーになる)
        """
        feature_shape = self.image_size.copy()
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        for dim in range(self.n_dim):  # height and width if dim=2
            for layer in self.cfg:
                channel, kernel, stride, padding = layer
                feature_shape[dim] = int(math.floor(
                    feature_shape[dim] + 2 * padding[dim] - kernel[dim]) / stride[dim] + 1)

        if len(feature_shape) == 1:
            feature_shape.append(1)

        return {
            'n_channels': self.cfg[-1][0],
            'height': int(feature_shape[0]),
            'width': int(feature_shape[1])}


def construct_cnn(cfg, use_as_extractor=False, n_dim=2):
    layer_info = []
    for layer in range(len(cfg['cnn_channel_list'])):
        layer_info.append((
            cfg['cnn_channel_list'][layer],
            cfg['cnn_kernel_sizes'][layer],
            cfg['cnn_stride_sizes'][layer],
            cfg['cnn_padding_sizes'][layer],
        ))
    cnn_maker = CNNMaker(in_channels=cfg['n_channels'], image_size=cfg['image_size'], cfg=layer_info, n_dim=n_dim,
                         n_classes=len(cfg['class_names']), use_as_extractor=use_as_extractor)
    return cnn_maker.construct_cnn()


def construct_logmel_cnn(cfg):
    return LogMelCNN()


def cnn_ftrs_16_751_751(eeg_conf):
    cfg = [
        (32, (4, 2), (3, 2), (0, 1)),
        (64, (4, 2), (3, 2), (0, 1)),
    ]
    layers = make_layers(cfg)
    n_channel, (out_height, out_width) = calc_feature_size(cfg, last_shape=True)
    return layers, n_channel * out_height # for RNN input size


def cnn_1_16_751_751(n_classes=1):
    cfg = [(16, (4, 4, 2), (2, 3, 2), (2, 0, 1)),
           (32, (4, 4, 2), (2, 3, 2), (2, 2, 1)),
           (64, (4, 4, 2), (2, 3, 2), (2, 1, 0))]
    layers = make_layers(cfg, dim=3)
    feature_size = calc_feature_size(cfg, dim=3)
    model = CNN(layers, feature_size, n_classes, dim=3)

    return model
