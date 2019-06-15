

from wrapper.models.CNN import CNNMaker
from wrapper.models.RNN import RNNMaker
from wrapper.models.toolbox import *


class Models:
    def __init__(self, args, n_classes, height, width=0, n_channels=0, time=0):
        """

        :param n_classes: # of classes data contains
        :param batch_size: batch size
        :param height: image height, else # of features
        :param width: image width, else 0
        :param n_channels: # of channels if data is image, otherwise 0
        :param time: Length of time dimension if data is video or wave image, otherwise 0
        """
        self.n_classes = n_classes
        self.batch_size = args.batch_size
        self.is_table = True if width == 0 else False
        self.is_image = True if n_channels != 0 else False
        self.is_time_series = True if time != 0 and self.is_image else False
        if self.is_image:
            self.n_channels = n_channels
            self.image_size = (height, width)
        if self.is_time_series:
            self.time = time
        self.cfg = self._init_cfg(args, height)

    def _init_cfg(self, args, input_size):
        return {
            'rnn_type': args.rnn_type,
            'input_size': input_size,
            'n_layers': args.rnn_n_layers,
            'hidden_size': args.rnn_hidden_size,
            'is_bidirectional': not args.bidirectional,
            'is_inference_softmax': not args.is_inference_softmax
        }

    def select_model(self, model_kind):
        if self.is_table:
            pass

        if model_kind == 'rnn':
            rnn_maker = RNNMaker(self.batch_size, self.n_classes)
            return rnn_maker.construct_rnn(self.cfg)
        elif model_kind == 'cnn':
            cnn_maker = CNNMaker(in_channels=self.n_channels, image_size=self.image_size, cfg=cfg,
                                 n_classes=self.n_classes)
            return cnn_maker.construct_cnn()

    def cnn(self):

        return cnn_1_16_399(self.n_classes, self.input_shape)
