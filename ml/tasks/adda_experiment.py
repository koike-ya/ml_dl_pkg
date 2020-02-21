import logging
import tempfile
from abc import ABCMeta
from copy import deepcopy
from typing import Tuple, Dict, List, Union

import mlflow
import numpy as np
import pandas as pd
from ml.models.train_manager import BaseTrainManager, train_manager_args
from ml.models.multitask_train_manager import MultitaskModelManager
from ml.src.cv_manager import KFoldManager, SUPPORTED_CV
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.src.metrics import get_metrics
from ml.src.preprocessor import Preprocessor, preprocess_args
# from ml.tasks.train_manager_ import train_manager_args
from ml.utils.utils import Metrics

logger = logging.getLogger(__name__)

DATALOADERS = {'normal': set_dataloader, 'ml': set_ml_dataloader}
TRAINMANAGERS = {'normal': BaseTrainManager, 'multitask': MultitaskModelManager}


def base_expt_args(parser):
    parser = train_manager_args(parser)
    parser = preprocess_args(parser)
    expt_parser = parser.add_argument_group("Experiment arguments")
    expt_parser.add_argument('--expt-id', help='data file for training', default='timestamp')
    expt_parser.add_argument('--n-seed-average', type=int, help='Seed averaging', default=0)
    expt_parser.add_argument('--cv-name', choices=SUPPORTED_CV, default=None)
    expt_parser.add_argument('--n-splits', type=int, help='Number of split on cv', default=0)
    expt_parser.add_argument('--train-with-all', action='store_true',
                             help='Whether train with train+devel dataset after hyperparameter tuning')
    expt_parser.add_argument('--train-manager', choices=TRAINMANAGERS.keys(), default='normal')
    expt_parser.add_argument('--data-loader', choices=DATALOADERS.keys(), default='normal')
    expt_parser.add_argument('--manifest-path', help='data file for training', default='input/train.csv')

    return parser


class BaseExperimentor(metaclass=ABCMeta):
    def __init__(self, cfg, load_func, label_func, dataset_cls):
        self.cfg = cfg
        self.load_func = load_func
        self.label_func = label_func
        self.dataset_cls = dataset_cls
        self.data_loader_cls = DATALOADERS[cfg['data_loader']]
        self.train_manager_cls = TRAINMANAGERS[cfg['train_manager']]
        if cfg['task_type'] == 'classify':
            self.train_metrics = get_metrics(['loss', 'uar'])
        else:
            self.train_metrics = get_metrics(['loss'])

    def _experiment(self, val_metrics, phases) -> Union[Tuple[Metrics, np.array], Tuple[Metrics, np.array, np.array]]:
        dataloaders = {}
        for phase in phases:
            process_func = Preprocessor(self.cfg, phase, self.cfg['sample_rate']).preprocess
            dataset = self.dataset_cls(self.cfg[f'{phase}_path'], self.cfg, self.load_func, process_func,
                                          self.label_func, phase)
            dataloaders[phase] = self.data_loader_cls(dataset, phase, self.cfg)

        if val_metrics:
            val_metrics = get_metrics(val_metrics, target_metric='loss')
            metrics = {'train': self.train_metrics, 'val': val_metrics}
        else:
            metrics = {'train': self.train_metrics}

        train_manager = self.train_manager_cls(self.cfg['class_names'], self.cfg, dataloaders, metrics)

        if 'val' in phases:
            metrics, val_pred = train_manager.train()
        else:       # This is only phases == ['train', 'infer']
            metrics, _ = train_manager.train(with_validate=False)
            pred = train_manager.infer()
            return metrics, pred

        if 'infer' in phases:   # phases == ['train', 'val', 'infer']
            pred = train_manager.infer()
            return metrics, val_pred, pred
        else:       # phases == ['train', 'val']
            return metrics, val_pred

    def train_with_validation(self, val_metrics: List[str]) -> Tuple[np.array, np.array]:
        phases = ['train', 'val']
        metrics, val_pred = self._experiment(val_metrics, phases)
        return np.array([m.average_meter.best_score for m in metrics['val']]), val_pred

    def experiment_with_validation(self, val_metrics: List[str]) -> Tuple[np.array, np.array, np.array]:
        phases = ['train', 'val', 'infer']
        metrics, val_pred, pred = self._experiment(val_metrics, phases)
        return np.array([m.average_meter.best_score for m in metrics['val']]), val_pred, pred

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
    def __init__(self, cfg: Dict, load_func, label_func, dataset_cls, cv_name: str, n_splits: int, groups: str = None):
        super().__init__(cfg, load_func, label_func, dataset_cls)
        self.orig_cfg = deepcopy(self.cfg)
        self.cv_name = cv_name
        self.n_splits = n_splits
        self.groups = groups
        self.metrics_df = pd.DataFrame()
        self.pred_list = []
        self.val_pred_list = []

    def _save_path(self, df_x, train_idx, val_idx, temp_dir):
        df_x.iloc[train_idx, :].to_csv(f'{temp_dir}/train_manifest.csv', header=None, index=False)
        df_x.iloc[val_idx, :].to_csv(f'{temp_dir}/val_manifest.csv', header=None, index=False)
        self.cfg[f'train_path'] = f'{temp_dir}/train_manifest.csv'
        self.cfg[f'val_path'] = f'{temp_dir}/val_manifest.csv'

    def set_metrics_df(self, result_list: List[float]):
        self.metrics_df = pd.concat([self.metrics_df.T, pd.Series(result_list)], axis=1).T
        assert self.metrics_df.shape[1] == len(result_list), self.metrics_df

    def train_with_validation(self, val_metrics: List[str]) -> Tuple[np.array, List[np.array]]:
        df_x = pd.concat([pd.read_csv(self.orig_cfg[f'train_path'], header=None),
                          pd.read_csv(self.orig_cfg[f'val_path'], header=None)])
        y = df_x.apply(lambda x: self.label_func(x), axis=1)
        logger.info(y.value_counts())

        k_fold = KFoldManager(self.cv_name, self.n_splits)

        for i, (train_idx, val_idx) in enumerate(k_fold.split(X=df_x.values, y=y.values, groups=self.groups)):
            logger.info(f'Fold {i + 1} started.')
            with tempfile.TemporaryDirectory() as temp_dir:
                self._save_path(df_x, train_idx, val_idx, temp_dir)

                result_series, val_pred = super().train_with_validation(val_metrics)
                self.val_pred_list.append(val_pred)
                self.set_metrics_df(result_series)

        self.metrics_df.columns = val_metrics
        self.cfg['train_path'] = self.orig_cfg['train_path']
        self.cfg['val_path'] = self.orig_cfg['val_path']

        return self.metrics_df.mean(axis=0).values, self.val_pred_list

    def experiment_with_validation(self, val_metrics: List[str]) -> Tuple[np.array, List[np.array], np.array]:
        df_x = pd.concat([pd.read_csv(self.orig_cfg[f'train_path'], header=None),
                          pd.read_csv(self.orig_cfg[f'val_path'], header=None)])
        y = df_x.apply(lambda x: self.label_func(x), axis=1)
        logger.info(y.value_counts())

        k_fold = KFoldManager(self.cv_name, self.n_splits)

        for i, (train_idx, val_idx) in enumerate(k_fold.split(X=df_x.values, y=y.values, groups=self.groups)):
            logger.info(f'Fold {i + 1} started.')
            with tempfile.TemporaryDirectory() as temp_dir:
                self._save_path(df_x, train_idx, val_idx, temp_dir)

                result_series, val_pred, pred = super().experiment_with_validation(val_metrics)
                self.pred_list.append(pred)
                self.val_pred_list.append(val_pred)
                self.set_metrics_df(result_series)

        self.metrics_df.columns = val_metrics
        self.cfg['train_path'] = self.orig_cfg['train_path']
        self.cfg['val_path'] = self.orig_cfg['val_path']

        return self.metrics_df.mean(axis=0).values, self.val_pred_list, np.array(self.pred_list).mean(axis=1)

    def experiment_without_validation(self, seed_average=0) -> np.array:
        raise NotImplementedError


def typical_train(expt_conf, load_func, label_func, dataset_cls, groups, val_metrics):
    if expt_conf['cv_name']:
        experimentor = CrossValidator(expt_conf, load_func, label_func, dataset_cls, expt_conf['cv_name'], expt_conf['n_splits'],
                                      groups)
    else:
        experimentor = BaseExperimentor(expt_conf, load_func, label_func, dataset_cls)

    result_series, val_pred = experimentor.train_with_validation(val_metrics)
    mlflow.log_metrics({metric_name: value for metric_name, value in zip(val_metrics, result_series)})

    return result_series, val_pred


def typical_experiment(expt_conf, load_func, label_func, dataset_cls, groups, val_metrics):
    if expt_conf['cv_name']:
        experimentor = CrossValidator(expt_conf, load_func, label_func, dataset_cls, expt_conf['cv_name'], expt_conf['n_splits'],
                                      groups)
    else:
        experimentor = BaseExperimentor(expt_conf, load_func, label_func, dataset_cls)

    result_series, pred = experimentor.experiment_with_validation(val_metrics)
    mlflow.log_metrics({metric_name: value for metric_name, value in zip(val_metrics, result_series)})

    return result_series, pred