import random
import time
import sys
import argparse
from abc import ABCMeta
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from copy import deepcopy
from ml.models.multitask_nn_model import MultitaskNNModel
from ml.models.nn_model import supported_nn_models, supported_pretrained_models
from ml.models.model_manager import supported_nn_models, supported_ml_models, model_manager_args, BaseModelManager
from sklearn.metrics import confusion_matrix
from ml.utils.logger import TensorBoardLogger
from tqdm import tqdm
from typing import Sequence, Tuple, Dict, List, Union
from ml.utils.utils import Metrics


def multitask_model_manager_args(parser) -> argparse.ArgumentParser:

    multitask_model_manager_parser = parser.add_argument_group("Model manager arguments")
    parser = model_manager_args(parser)

    return parser


class MultitaskModelManager(BaseModelManager):
    def __init__(self, class_labels, cfg, dataloaders, metrics):
        super(MultitaskModelManager, self).__init__(class_labels, cfg, dataloaders, metrics)
        self.n_tasks = cfg['n_tasks']

    def _init_model(self) -> MultitaskNNModel:
        self.cfg['input_size'] = list(self.dataloaders.values())[0].get_input_size()

        return MultitaskNNModel(self.class_labels, self.cfg)

    def _init_device(self) -> torch.device:
        if self.cfg['cuda']:
            device = torch.device("cuda")
            torch.cuda.set_device(self.cfg['gpu_id'])
        else:
            device = torch.device("cpu")

        return device

    def _verbose(self, epoch, phase, i, elapsed) -> None:
        data_len = len(self.dataloaders[phase])
        eta = int(elapsed / (i + 1) * (data_len - (i + 1)))
        progress = f'\r{phase} epoch: [{epoch + 1}][{i + 1}/{data_len}]\t {elapsed}(s) eta:{eta}(s)\t'
        progress += '\t'.join([f'{metric.name} {metric.average_meter.value:.4f}' for metric in self.metrics[phase]])
        print(progress, end='')
        sys.stdout.flush()

    def _record_log(self, phase, epoch) -> None:
        values = {}

        for metric in self.metrics[phase]:
            values[phase + '_' + metric.name] = metric.average_meter[phase].average
        self.logger.update(epoch, values)

    def _update_by_epoch(self, phase, epoch, learning_anneal) -> None:
        for metric in self.metrics[phase]:
            best_flag = metric.average_meter.update_best()
            if metric.save_model and best_flag and phase == 'val':
                print(f"Found better validated model, saving to {self.cfg['model_path']}")
                self.model.save_model()

            # reset epoch average meter
            metric.average_meter.reset()

        # anneal lr
        if phase == 'train':
            self.model.anneal_lr(learning_anneal)

    def train(self, model=None, with_validate=True) -> Metrics:
        if model:
            self.model = model

        start = time.time()
        epoch_metrics = {}

        if with_validate:
            phases = ['train', 'val']
        else:
            phases = ['train']

        self.check_keys_from_dict(phases, self.dataloaders)

        for epoch in range(self.cfg['epochs']):
            for phase in phases:
                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    # labels = [label.to(self.device) for label in labels]
                    loss, predicts = self.model.fit(inputs.to(self.device), labels, phase)

                    # for i_task in range(self.n_tasks):
                    # save loss and metrics in one batch
                    for j, metric in enumerate(self.metrics[phase]):
                        metric.update(loss, predicts[:, j // 2], labels[j // 2].numpy())

                    if not self.cfg['silent']:
                        self._verbose(epoch, phase, i, elapsed=int(time.time() - start))

                if self.logger:
                    self._record_log(phase, epoch)

                epoch_metrics[phase] = deepcopy(self.metrics[phase])

                self._update_by_epoch(phase, epoch, self.cfg['learning_anneal'])

            if self.cfg['silent']:
                print(f'epoch {str(epoch + 1).ljust(2)}->', end=' ')
                print(f'lr: {self.model.get_lr():.6f}', end='\t')
                for phase in phases:
                    print(f'{phase}: [', end='')
                    print('\t'.join([f'{m.name}: {m.average_meter.average:.4f}' for m in epoch_metrics[phase]]), end='')
                    print(']', end='\t')
                print('')

        if self.logger:
            self.logger.close()

        return self.metrics

    def infer(self, load_best=True, phase='infer') -> List[np.array]:
        if load_best:
            self.model.load_model()

        batch_size = self.cfg['batch_size']

        self.check_keys_from_dict([phase], self.dataloaders)

        dtype_ = np.int if self.cfg['task_type'] == 'classify' else np.float

        # ラベルが入れられなかった部分を除くため、小さな負の数を初期値として格納
        pred_list = np.zeros((len(self.dataloaders[phase]) * batch_size, self.n_tasks), dtype=dtype_) - 1000000
        for i, (inputs, _) in tqdm(enumerate(self.dataloaders[phase]), total=len(self.dataloaders[phase])):
            inputs = inputs.to(self.device)
            preds = self.model.predict(inputs)
            pred_list[i * batch_size:i * batch_size + preds.shape[0], :] = preds

        pred_list = pred_list[~(pred_list[:, 0] == -1000000), :]

        if self.cfg['tta']:
            raise NotImplementedError
            # for i_task in range(self.n_tasks):
            #     pred_list[:, i_task] = pred_list[:, i_task].reshape(self.cfg['tta'], -1).mean(axis=0)
            #     label_list[:, i_task] = label_list[:, i_task][:label_list[:, i_task].shape[0] // self.cfg['tta']]

        return pred_list
