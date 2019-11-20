import torch

seed = 0
torch.manual_seed(seed)
from keras import backend

torch.cuda.manual_seed_all(seed)
import random
random.seed(seed)
import tensorflow as tf
from ml.models.base_model import BaseModel

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import  Dense, Conv3D, Dropout, Flatten, BatchNormalization


class CHBMITCNN(BaseModel):
    def __init__(self, model_path, cfg):
        super(CHBMITCNN, self).__init__(cfg['class_names'], cfg, [])
        self.model_path = model_path
        if cfg['data_type'] == 'chbmit':
            input_shape = (1, 22, 59, 114)
        elif cfg['data_type'] == 'children':
            input_shape = (1, 4, 116, 236)
        else:
            raise NotImplementedError

        model = Sequential()
        # C1
        model.add(
            Conv3D(16, (4, 5, 5), strides=(1, 2, 2), padding='valid', activation='relu', data_format="channels_first",
                   input_shape=input_shape))
        model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2), data_format="channels_first", padding='same'))
        model.add(BatchNormalization())

        # C2
        model.add(Conv3D(32, (1, 3, 3), strides=(1, 1, 1), padding='valid', data_format="channels_first",
                         activation='relu'))  # incertezza se togliere padding
        model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2), data_format="channels_first", ))
        model.add(BatchNormalization())

        # C3
        model.add(Conv3D(64, (1, 3, 3), strides=(1, 1, 1), padding='valid', data_format="channels_first",
                         activation='relu'))  # incertezza se togliere padding
        model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2), data_format="channels_first", ))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        opt_adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=opt_adam,
                      metrics=['accuracy', tf.keras.metrics.Recall()])
        self.model = model

    def fit(self, inputs, labels, phase):
        inputs = np.array([inputs]).swapaxes(0, 1)
        if phase == 'train':
            metric_values = self.model.train_on_batch(inputs, labels)
        elif phase == 'val':
            metric_values = self.model.test_on_batch(inputs, labels)
        return metric_values

    def predict(self, inputs):
        inputs = np.array([inputs]).swapaxes(0, 1)
        return np.argmax(self.model.predict(inputs), axis=1)

    def save_model(self):
        self.model.save(self.model_path)

    def load_model(self):
        self.model.load_weights(self.model_path)
        return self.model


def precision(y_true, y_pred):
    y_pred_pos = backend.round(backend.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = backend.round(backend.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = backend.sum(y_pos * y_pred_pos)
    tn = backend.sum(y_neg * y_pred_neg)
    fp = backend.sum(y_neg * y_pred_pos)
    fn = backend.sum(y_pos * y_pred_neg)
    far = fp / (tp + fp)
    return far