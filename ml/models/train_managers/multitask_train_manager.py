import argparse
import logging
import time

logger = logging.getLogger(__name__)

import numpy as np
import torch
from copy import deepcopy
from ml.models.model_managers.multitask_nn_model_manager import MultitaskNNModelManager
from ml.models.train_managers.nn_train_manager import NNTrainManager
from ml.models.train_managers.base_train_manager import train_manager_args
from tqdm import tqdm
from typing import List, Tuple
from ml.utils.utils import Metrics


def multitask_train_manager_args(parser) -> argparse.ArgumentParser:

    multitask_train_manager_parser = parser.add_argument_group("Model manager arguments")
    parser = train_manager_args(parser)

    return parser


class MultitaskTrainManager(NNTrainManager):
    def __init__(self, class_labels, cfg, dataloaders, metrics):
        super(MultitaskTrainManager, self).__init__(class_labels, cfg, dataloaders, metrics)
        self.n_tasks = cfg['n_tasks']

    def _init_model(self) -> MultitaskNNModelManager:
        self.cfg['input_size'] = list(self.dataloaders.values())[0].get_input_size()

        return MultitaskNNModelManager(self.class_labels, self.cfg)

    def _init_device(self) -> torch.device:
        if self.cfg['cuda']:
            device = torch.device("cuda")
            torch.cuda.set_device(self.cfg['gpu_id'])
        else:
            device = torch.device("cpu")

        return device

    def train(self, model=None, with_validate=True) -> Tuple[Metrics, np.array]:
        if model:
            self.model = model

        start = time.time()
        epoch_metrics = {}
        best_val_pred = np.array([])

        if with_validate:
            phases = ['train', 'val']
        else:
            phases = ['train']

        self.check_keys_from_dict(phases, self.dataloaders)
        batch_size = self.cfg['batch_size']
        dtype_ = np.int if self.cfg['task_type'] == 'classify' else np.float

        for epoch in range(self.cfg['epochs']):
            for phase in phases:
                # ラベルが入れられなかった部分を除くため、小さな負の数を初期値として格納
                pred_list = np.zeros((len(self.dataloaders[phase]) * batch_size, self.n_tasks),
                                     dtype=dtype_) - 1000000

                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    loss, predicts = self.model.fit(inputs.to(self.device), labels, phase)
                    pred_list[i * batch_size:i * batch_size + predicts.shape[0], :] = predicts

                    # save loss and metrics in one batch
                    for j, metric in enumerate(self.metrics[phase]):
                        metric.update(loss, predicts[:, j // 2], labels[j // 2].numpy())

                    self._verbose(epoch, phase, i, elapsed=int(time.time() - start))

                if self.logger:
                    self._record_log(phase, epoch)

                epoch_metrics[phase] = deepcopy(self.metrics[phase])

                best_val_flag = self._update_by_epoch(phase, epoch, self.cfg['learning_anneal'])
                if best_val_flag:
                    best_val_pred = pred_list[~(pred_list[:, 0] == -1000000), :]

            self._epoch_verbose(epoch, epoch_metrics, phases)

        if self.logger:
            self.logger.close()

        return self.metrics, best_val_pred

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
