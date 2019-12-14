import torch

seed = 0
torch.manual_seed(seed)
import math
import numpy as np
torch.cuda.manual_seed_all(seed)
import random
random.seed(seed)
import tensorflow as tf
from ml.models.base_model import BaseModel
import torch.nn as nn
from torchvision import models


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten, TimeDistributed, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras import metrics
from keras import backend

from ml.models.nn_model import NNModel


# Reference:
# https://github.com/ramyh/Epileptic-Seizure-Detection/blob/master/SeizureDetection_3Classes_NoisyWhite_SNR%2B20.ipynb


class BonnRNN(BaseModel):
    def __init__(self, model_path, cfg):
        super(BonnRNN, self).__init__(cfg['class_names'], cfg, [])
        self.model_path = model_path
        # Time Steps of LSTM
        if cfg['data_type'] == 'chbmit':
            sec_len = 30
            sr = 256
            data_length = sec_len * sr
            n_channel = 22
        elif cfg['data_type'] == 'children':
            sec_len = 30
            sr = 500
            data_length = sec_len * sr
            n_channel = 4
        else:
            raise NotImplementedError

        timesteps = sec_len // 2 * sr  # 15 * 256
        data_dim = n_channel * data_length // timesteps

        # create the model
        model = Sequential()
        # model.add(LSTM(100, input_shape= (4096, 1)))
        # model.add(LSTM(100, input_shape= (64, 64)))
        # model.add(Dropout(0.1, input_shape= (timesteps, data_dim)))
        # model.add(LSTM(100, return_sequences = True))

        model.add(LSTM(100, input_shape=(timesteps, data_dim), return_sequences=True))
        # model.add(Dropout(0.1))
        model.add(TimeDistributed(Dense(50)))
        # model.add(GlobalMaxPooling1D())
        model.add(GlobalAveragePooling1D())
        # model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Recall()])

        self.model = model

    def fit(self, inputs, labels, phase):
        if phase == 'train':
            metric_values = self.model.train_on_batch(inputs, labels)
        elif phase == 'val':
            metric_values = self.model.test_on_batch(inputs, labels)
        return metric_values

    def predict(self, inputs):
        return np.argmax(self.model.predict(inputs), axis=1)

    def save_model(self):
        self.model.save(self.model_path)

    def load_model(self):
        self.model.load_weights(self.model_path)
        return self.model
