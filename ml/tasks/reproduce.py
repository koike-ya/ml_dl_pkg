import logging
import tempfile
from abc import ABCMeta
from copy import deepcopy
from typing import Tuple, Dict, List

import mlflow
import numpy as np
import pandas as pd

from ml.models.train_managers.base_train_manager import train_manager_args
from ml.models.train_managers.ml_train_manager import MLTrainManager
from ml.models.train_managers.multitask_train_manager import MultitaskTrainManager
from ml.models.train_managers.nn_train_manager import NNTrainManager
from ml.tasks.base_experiment import BaseExperimentor
from ml.preprocess.preprocessor import Preprocessor, preprocess_args
from ml.src.cv_manager import KFoldManager, SUPPORTED_CV
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.src.metrics import get_metric_list
from ml.utils.utils import Metrics

logger = logging.getLogger(__name__)

DATALOADERS = {'normal': set_dataloader, 'ml': set_ml_dataloader}
TRAINMANAGERS = {'nn': NNTrainManager, 'multitask': MultitaskTrainManager, 'ml': MLTrainManager}


def base_expt_args(parser):
    parser = train_manager_args(parser)
    parser = preprocess_args(parser)
    expt_parser = parser.add_argument_group("Experiment arguments")
    expt_parser.add_argument('--expt-id', help='data file for training', default='timestamp')
    expt_parser.add_argument('--n-seed-average', type=int, help='Seed averaging', default=0)
    expt_parser.add_argument('--cv-name', choices=SUPPORTED_CV, default=None)
    expt_parser.add_argument('--n-splits', type=int, help='Number of split on cv', default=0)
    expt_parser.add_argument('--infer', action='store_true',
                             help='Whether training with train+devel dataset after hyperparameter tuning')
    expt_parser.add_argument('--test', action='store_true',
                             help='Whether training with train+devel dataset after hyperparameter tuning')
    expt_parser.add_argument('--train-manager', choices=TRAINMANAGERS.keys(), default='nn')
    expt_parser.add_argument('--data-loader', choices=DATALOADERS.keys(), default='normal')
    expt_parser.add_argument('--manifest-path', help='data file for training', default='input/train.csv')

    return parser


def get_metrics(phases, task_type, train_manager='normal'):
    metrics = {}
    for phase in phases:
        if task_type == 'classify':
            if train_manager == 'ml':
                metrics[phase] = get_metric_list(['uar'], target_metric='uar')
            else:
                metrics[phase] = get_metric_list(['loss', 'uar'], target_metric='uar')
        else:
            metrics[phase] = get_metric_list(['loss'], target_metric='loss')

    return metrics


class Reproducer(BaseExperimentor):
    def __init__(self, cfg, load_func, label_func, process_func=None, dataset_cls=None):
        super(Reproducer, self).__init__(cfg, load_func, label_func, process_func, dataset_cls)

    def reproduce(self, phase, metrics_names=None):
        if not metrics_names:
            metrics = get_metrics([phase], self.cfg['task_type'], self.cfg['train_manager'])
        else:
            metrics = {p: get_metric_list(metrics_names[p]) for p in [phase]}

        dataloaders = {}
        if not self.process_func:
            self.process_func = Preprocessor(self.cfg, phase).preprocess
        dataset = self.dataset_cls(self.cfg[f'{phase}_path'], self.cfg, phase, self.load_func, self.process_func,
                                   self.label_func)
        dataloaders[phase] = self.data_loader_cls(dataset, phase, self.cfg)
        train_manager = self.train_manager_cls(self.cfg['class_names'], self.cfg, dataloaders, deepcopy(metrics))

        if phase == 'infer':
            pred_list = train_manager.infer()
            return pred_list
        else:
            pred_list, metrics = train_manager.test(return_metrics=True)
            return pred_list, metrics
