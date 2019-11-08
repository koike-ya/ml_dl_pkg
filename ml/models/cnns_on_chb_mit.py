import torch

seed = 0
torch.manual_seed(seed)
import math
import numpy as np
torch.cuda.manual_seed_all(seed)
import random
random.seed(seed)
import tensorflow as tf
import torch.nn as nn
from torchvision import models


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import  Dense, Conv3D, Dropout, Flatten, BatchNormalization
from sklearn.metrics import recall_score
from keras.callbacks import EarlyStopping
from random import shuffle

from ml.models.nn_model import NNModel


class CHBMITCNN:
    def __init__(self, model_path, cfg):
        self.model_path = model_path
        input_shape = (1, 4, 61, 236)
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

    def fit(self, train_inputs, train_labels, batch_size, epochs, validation_data, callbacks):
        return self.model.fit(train_inputs, train_labels, batch_size=batch_size,
                       epochs=epochs, validation_data=validation_data,
                       callbacks=callbacks)

    def predict(self, inputs):
        return np.argmax(self.model.predict(inputs), axis=1)

    def save_model(self):
        self.model.save(self.model_path)

    def load_model(self):
        self.model.load_weights(self.model_path)
        return self.model


def false_detection_rate(true, pred):
    print(true)
    true, pred = true[:, 1], pred[:, 1]
    print(pred.shape)
    return tf.tensordot(tf.dtypes.cast(true == 0, tf.int32), tf.dtypes.cast(pred == 1, tf.int32), axes=1) / pred.shape[0]
