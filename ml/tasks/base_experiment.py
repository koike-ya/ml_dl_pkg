from abc import ABCMeta, abstractmethod
import argparse
import itertools
import tempfile
from pathlib import Path
from typing import Tuple, Dict, List, Union

from copy import deepcopy
import numpy as np
import pandas as pd
from librosa.core import load
from ml.models.model_manager import BaseModelManager
from ml.models.multitask_model_manager import MultitaskModelManager
from ml.src.cv_manager import KFoldManager, SUPPORTED_CV
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.src.dataset import ManifestWaveDataSet
from ml.src.metrics import get_metrics
from ml.src.preprocessor import Preprocessor, preprocess_args
from ml.tasks.train_manager import train_manager_args
from ml.utils.utils import Metrics

DATALOADERS = {'normal': set_dataloader, 'ml': set_ml_dataloader}
MODELMANAGERS = {'normal': BaseModelManager, 'multitask': MultitaskModelManager}


def base_expt_args(parser):
    parser = train_manager_args(parser)
    parser = preprocess_args(parser)
    expt_parser = parser.add_argument_group("Experiment arguments")
    expt_parser.add_argument('--expt-id', help='data file for training', default='sth')
    expt_parser.add_argument('--n-seed-average', type=int, help='Seed averaging', default=0)
    expt_parser.add_argument('--cv-name', choices=SUPPORTED_CV, default=None)
    expt_parser.add_argument('--n-splits', type=int, help='Number of split on cv', default=0)
    expt_parser.add_argument('--train-with-all', action='store_true',
                             help='Whether train with train+devel dataset after hyperparameter tuning')
    expt_parser.add_argument('--model-manager', choices=MODELMANAGERS.keys(), default='normal')
    expt_parser.add_argument('--data-loader', choices=DATALOADERS.keys(), default='normal')

    return parser


class BaseExperimentor(metaclass=ABCMeta):
    def __init__(self, cfg, load_func, label_func):
        self.cfg = cfg
        self.load_func = load_func
        self.label_func = label_func
        self.data_loader_cls = DATALOADERS[cfg['data_loader']]
        self.model_manager_cls = MODELMANAGERS[cfg['model_manager']]

    def _experiment(self, val_metrics, phases) -> Tuple[Metrics, np.array]:
        dataloaders = {}
        for phase in phases:
            process_func = Preprocessor(self.cfg, phase, self.cfg['sample_rate']).preprocess
            dataset = ManifestWaveDataSet(self.cfg[f'{phase}_path'], self.cfg, self.load_func, process_func,
                                          self.label_func, phase)
            dataloaders[phase] = self.data_loader_cls(dataset, phase, self.cfg)

        train_metrics = get_metrics(['loss', 'uar'])
        if val_metrics:
            val_metrics = get_metrics(val_metrics, target_metric='loss')
            metrics = {'train': train_metrics, 'val': val_metrics}
        else:
            metrics = {'train': train_metrics}

        model_manager = self.model_manager_cls(self.cfg['class_names'], self.cfg, dataloaders, metrics)

        if phases == ['train', 'infer']:
            metrics = model_manager.train(with_validate=False)
        else:
            metrics = model_manager.train()

        pred = model_manager.infer()

        return metrics, pred

    def experiment_with_validation(self, val_metrics: List[str]) -> Tuple[np.array, np.array]:
        phases = ['train', 'val', 'infer']
        metrics, pred = self._experiment(val_metrics, phases)
        return np.array([m.average_meter.best_score for m in metrics['val']]), pred

    def experiment_without_validation(self, seed_average=0) -> np.array:
        phases = ['train', 'infer']

        if not seed_average:
            _, pred = self._experiment(val_metrics=None, phases=phases)
            return pred

        else:
            pred_list = []
            for seed in range(self.cfg['seed']):
                _, pred = self._experiment(val_metrics=None, phases=phases)
                pred_list.append(pred)

            assert np.array(pred_list).T.shape[1] == seed_average
            return np.array(pred_list).T.mean(axis=1)


class CrossValidator(BaseExperimentor):
    def __init__(self, cfg: Dict, load_func, label_func, cv_name: str, n_splits: int, groups: str = None):
        super().__init__(cfg, load_func, label_func)
        self.orig_cfg = deepcopy(self.cfg)
        self.cv_name = cv_name
        self.n_splits = n_splits
        self.groups = groups
        self.metrics_df = pd.DataFrame()
        self.pred_list = []

    def set_metrics_df(self, result_list: List[float]):
        self.metrics_df = pd.concat([self.metrics_df.T, pd.Series(result_list)], axis=1).T
        assert self.metrics_df.shape[1] == len(result_list), self.metrics_df

    def experiment_with_validation(self, val_metrics: List[str]) -> Tuple[np.array, np.array]:
        df_x = pd.concat([pd.read_csv(self.orig_cfg[f'train_path'], header=None),
                          pd.read_csv(self.orig_cfg[f'val_path'], header=None)])
        y = df_x.apply(lambda x: self.label_func(x), axis=1)
        print(y.value_counts())

        k_fold = KFoldManager(self.cv_name, self.n_splits)

        for i, (train_idx, val_idx) in enumerate(k_fold.split(X=df_x.values, y=y.values, groups=self.groups)):
            print(f'Fold {i + 1} started.')
            with tempfile.TemporaryDirectory() as temp_dir:
                df_x.iloc[train_idx, :].to_csv(f'{temp_dir}/train_manifest.csv', header=None, index=False)
                df_x.iloc[val_idx, :].to_csv(f'{temp_dir}/val_manifest.csv', header=None, index=False)
                self.cfg[f'train_path'] = f'{temp_dir}/train_manifest.csv'
                self.cfg[f'val_path'] = f'{temp_dir}/val_manifest.csv'

                result_series, pred = super().experiment_with_validation(val_metrics)
                self.pred_list.append(pred)
                self.set_metrics_df(result_series)

        self.metrics_df.columns = val_metrics
        self.cfg['train_path'] = self.orig_cfg['train_path']
        self.cfg['val_path'] = self.orig_cfg['val_path']

        return self.metrics_df.mean(axis=0).values, np.array(self.pred_list).mean(axis=1)

    def experiment_without_validation(self) -> np.array:
        raise NotImplementedError


class SeedAverager(BaseExperimentor):
    def __init__(self, cfg: Dict, load_func, label_func, cv_name: str = None, n_splits: int = None, groups: str = None):
        super().__init__(cfg, load_func, label_func)
        if cv_name:
            self = CrossValidator(cfg, load_func, label_func, cv_name, n_splits, groups)
        self.seed_metrics_df = pd.DataFrame()
        self.pred_list = []

    def set_seed_metrics_df(self, result_list: List[float]):
        self.seed_metrics_df = pd.concat([self.metrics_df, pd.DataFrame(result_list)])
        assert self.seed_metrics_df.shape[1] == len(result_list)

    def experiment_with_validation(self, val_metrics: List[str]) -> Tuple[np.array, np.array]:
        for seed in range(self.cfg['seed']):
            metrics, pred = super().experiment_with_validation(val_metrics)
            self.pred_list.append(pred)
            self.set_seed_metrics_df([m.average_meter.best_score for m in metrics['val']])

        self.seed_metrics_df.columns = val_metrics

        return self.seed_metrics_df.mean(axis=0).values, np.array(self.pred_list).mean(axis=1)

    def experiment_without_validation(self) -> np.array:
        for seed in range(self.cfg['seed']):
            pred = super().experiment_without_validation()
            self.pred_list.append(pred)

        return np.array(self.pred_list).mean(axis=1)

