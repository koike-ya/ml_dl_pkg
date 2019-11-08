import random
import time
from abc import ABCMeta
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import os
import torch
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from tqdm import tqdm

import keras
import numpy as np
from ml.models.model_manager import BaseModelManager
from ml.models.cnns_on_chb_mit import CHBMITCNN


class KerasModelManager(BaseModelManager):
    def __init__(self, class_labels, cfg, dataloaders, metrics):
        super(KerasModelManager, self).__init__(class_labels, cfg, dataloaders, metrics)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        Path(self.cfg['model_path']).parent.mkdir(exist_ok=True, parents=True)

    def _init_model(self):
        return CHBMITCNN(self.cfg['model_path'], self.cfg)

    def _init_device(self):
        if self.cfg['cuda']:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    def _predict(self, phase):
        batch_size = self.cfg['batch_size']

        self.check_keys_from_dict([phase], self.dataloaders)

        dtype_ = np.int if self.cfg['task_type'] == 'classify' else np.float
        # ラベルが入れられなかった部分を除くため、小さな負の数を初期値として格納
        pred_list = np.zeros((len(self.dataloaders[phase]) * batch_size, 1), dtype=dtype_) - 1000000
        label_list = np.zeros((len(self.dataloaders[phase]) * batch_size, 1), dtype=dtype_) - 1000000
        for i, (inputs, labels) in tqdm(enumerate(self.dataloaders[phase]), total=len(self.dataloaders[phase])):
            inputs = torch.unsqueeze(inputs, 1).cpu().numpy()

            labels = labels.numpy().reshape(-1,)
            preds = self.model.predict(inputs)

            pred_list[i * batch_size:i * batch_size + preds.shape[0], 0] = preds.reshape(-1,)
            label_list[i * batch_size:i * batch_size + labels.shape[0], 0] = labels

        return pred_list[~(pred_list == -1000000)], label_list[~(label_list == -1000000)]

    def train(self):
        self.check_keys_from_dict(['train', 'val'], self.dataloaders)
        callback = EarlyStoppingByLossVal(monitor='val_accuracy', value=0.975, verbose=1, lower=False)

        for i, ((train_inputs, train_labels), (val_inputs, val_labels)) in enumerate(
                zip(self.dataloaders['train'], self.dataloaders['val'])):
            train_inputs = torch.unsqueeze(train_inputs, 1)
            val_inputs = torch.unsqueeze(val_inputs, 1)

            y_onehot = torch.zeros(train_labels.size(0), len(self.class_labels))
            train_labels = y_onehot.scatter_(1, train_labels.view(-1, 1).type(torch.LongTensor), 1)
            y_onehot = torch.zeros(val_labels.size(0), len(self.class_labels))
            val_labels = y_onehot.scatter_(1, val_labels.view(-1, 1).type(torch.LongTensor), 1)
            history = self.model.fit(train_inputs.cpu().numpy(), train_labels.cpu().numpy(), batch_size=self.cfg['batch_size'],
                           epochs=self.cfg['epochs'], validation_data=(val_inputs.cpu().numpy(), val_labels.cpu().numpy()),
                           callbacks=[callback])
            self.model.save_model()
            # print(history.history.keys())
            # exit()
        return self.metrics

    def test(self, return_metrics=False, load_best=True):
        if load_best:
            self.model.load_model()

        pred_list, label_list = self._predict(phase='test')

        for metric in self.metrics:
            if metric.name == 'loss':
                continue

            metric.update(phase='test', loss_value=0.0, preds=pred_list, labels=label_list)
            print(f"{metric.name}: {metric.average_meter['test'].value :.4f}")

        if self.cfg['task_type'] == 'classify':
            confusion_matrix_ = confusion_matrix(label_list, pred_list,
                                                 labels=list(range(len(self.class_labels))))
            print(confusion_matrix_)

        if return_metrics:
            return pred_list, label_list, self.metrics
        return pred_list, label_list

    def infer(self, load_best=True):
        if load_best:
            self.model.load_model()

        # test実装
        pred_list, _ = self._predict(phase='infer')
        return pred_list


class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0, lower=True):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.lower=lower

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if self.lower:
            if current < self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True
        else:
            if current > self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True


class TensorBoardLogger(object):
    def __init__(self, id, log_dir):
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        self.id = id
        self.tensorboard_writer = SummaryWriter(log_dir)

    def update(self, epoch, values):
        self.tensorboard_writer.add_scalars(self.id, values, epoch + 1)
