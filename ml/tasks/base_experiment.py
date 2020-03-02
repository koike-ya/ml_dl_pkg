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
from ml.src.cv_manager import KFoldManager, SUPPORTED_CV
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.src.metrics import get_metric_list
from ml.src.preprocessor import Preprocessor, preprocess_args
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
                metrics[phase] = get_metric_list(['loss', 'uar'], target_metric='loss')
        else:
            metrics[phase] = get_metric_list(['loss'], target_metric='loss')

    return metrics

class BaseExperimentor(metaclass=ABCMeta):
    def __init__(self, cfg, load_func, label_func, process_func=None, dataset_cls=None):
        self.cfg = cfg
        self.load_func = load_func
        self.label_func = label_func
        self.dataset_cls = dataset_cls
        self.data_loader_cls = DATALOADERS[cfg['data_loader']]
        self.train_manager_cls = TRAINMANAGERS[cfg['train_manager']]
        self.process_func = process_func
        self.test = cfg['test']
        self.infer = cfg['infer']
        
    def _experiment(self, metrics, phases) -> Tuple[Metrics, Dict[str, np.array]]:
        pred_list = {}

        dataloaders = {}
        for phase in phases:
            if not self.process_func:
                self.process_func = Preprocessor(self.cfg, phase, self.cfg['sample_rate']).preprocess
            dataset = self.dataset_cls(self.cfg[f'{phase}_path'], self.cfg, phase, self.load_func, self.process_func,
                                       self.label_func)
            dataloaders[phase] = self.data_loader_cls(dataset, phase, self.cfg)

        train_manager = self.train_manager_cls(self.cfg['class_names'], self.cfg, dataloaders, deepcopy(metrics))
        
        if 'val' in phases:
            metrics, pred_list['val'] = train_manager.train()
        else:       # This is the case in ['train', 'infer'], ['train', 'test']
            metrics, _ = train_manager.train(with_validate=False)
            
        if 'infer' in phases:
            pred_list['infer'] = train_manager.infer()
        elif 'test' in phases:
            metrics, pred_list['test'] = train_manager.test()

        return metrics, pred_list

    def train_with_validation(self, metrics: Metrics) -> Tuple[np.array, np.array]:
        phases = ['train', 'val']
        
        metrics, pred_list = self._experiment(metrics, phases)
        return np.array([m.average_meter.best_score for m in metrics['val']]), pred_list['val']

    def experiment_with_validation(self, metrics: Metrics, infer=False) -> Tuple[np.array, Dict[str, np.array]]:
        if self.infer:
            phases = ['train', 'val', 'infer']
        else:
            phases = ['train', 'val', 'test']
        
        metrics, pred_list = self._experiment(metrics, phases)
        return np.array([m.average_meter.best_score for m in metrics['val']]), pred_list

    def experiment_without_validation(self, metrics: Metrics, infer=False, seed_average=0) -> np.array:
        if self.infer:
            phases = ['train', 'infer']
        else:
            phases = ['train', 'test']
            
        if not seed_average:
            _, pred = self._experiment(metrics=metrics, phases=phases)
            return pred

        else:
            pred_list = []
            for seed in range(self.cfg['seed']):
                _, pred = self._experiment(metrics=metrics, phases=phases)
                pred_list.append(pred)

            assert np.array(pred_list).T.shape[1] == seed_average
            return np.array(pred_list).T.mean(axis=1)


class CrossValidator(BaseExperimentor):
    def __init__(self, cfg: Dict, load_func, label_func, process_func, dataset_cls, cv_name: str, n_splits: int,
                 groups: str = None):
        super().__init__(cfg, load_func, label_func, process_func, dataset_cls)
        self.orig_cfg = deepcopy(self.cfg)
        self.cv_name = cv_name
        self.n_splits = n_splits
        self.groups = groups
        self.metrics_df = pd.DataFrame()
        self.pred_list = []

    def _save_path(self, df_x, train_idx, val_idx, temp_dir):
        df_x.iloc[train_idx, :].to_csv(f'{temp_dir}/train_manifest.csv', header=None, index=False)
        df_x.iloc[val_idx, :].to_csv(f'{temp_dir}/val_manifest.csv', header=None, index=False)
        self.cfg[f'train_path'] = f'{temp_dir}/train_manifest.csv'
        self.cfg[f'val_path'] = f'{temp_dir}/val_manifest.csv'

    def set_metrics_df(self, result_list: List[float]):
        self.metrics_df = pd.concat([self.metrics_df.T, pd.Series(result_list)], axis=1).T
        assert self.metrics_df.shape[1] == len(result_list), self.metrics_df

    def train_with_validation(self, metrics: Metrics) -> Tuple[np.array, List[Dict[str, np.array]]]:
        df_x = pd.concat([pd.read_csv(self.orig_cfg[f'train_path'], header=None),
                          pd.read_csv(self.orig_cfg[f'val_path'], header=None)])
        y = df_x.apply(lambda x: self.label_func(x), axis=1)
        logger.info(y.value_counts())
        
        k_fold = KFoldManager(self.cv_name, self.n_splits)

        for i, (train_idx, val_idx) in enumerate(k_fold.split(X=df_x.values, y=y.values, groups=self.groups)):
            logger.info(f'Fold {i + 1} started.')
            with tempfile.TemporaryDirectory() as temp_dir:
                self._save_path(df_x, train_idx, val_idx, temp_dir)

                result_series, fold_pred_list = super().train_with_validation(metrics)
                self.pred_list.append(fold_pred_list)
                self.set_metrics_df(result_series)

        self.metrics_df.columns = val_metrics
        self.cfg['train_path'] = self.orig_cfg['train_path']
        self.cfg['val_path'] = self.orig_cfg['val_path']

        return self.metrics_df.mean(axis=0).values, self.pred_list

    def experiment_with_validation(self, metrics: Metrics, infer=False) -> Tuple[np.array,
                                                                                       List[Dict[str, np.array]]]:
        df_x = pd.concat([pd.read_csv(self.orig_cfg[f'train_path'], header=None),
                          pd.read_csv(self.orig_cfg[f'val_path'], header=None)])
        y = df_x.apply(lambda x: self.label_func(x), axis=1)
        logger.info(y.value_counts())

        k_fold = KFoldManager(self.cv_name, self.n_splits)

        for i, (train_idx, val_idx) in enumerate(k_fold.split(X=df_x.values, y=y.values, groups=self.groups)):
            logger.info(f'Fold {i + 1} started.')
            with tempfile.TemporaryDirectory() as temp_dir:
                self._save_path(df_x, train_idx, val_idx, temp_dir)

                result_series, fold_pred_list = super().experiment_with_validation(metrics, infer=infer)
                self.pred_list.append(fold_pred_list)
                self.set_metrics_df(result_series)

        self.metrics_df.columns = val_metrics
        self.cfg['train_path'] = self.orig_cfg['train_path']
        self.cfg['val_path'] = self.orig_cfg['val_path']

        return self.metrics_df.mean(axis=0).values, self.pred_list

    def experiment_without_validation(self, infer=False, seed_average=0) -> np.array:
        raise NotImplementedError


def typical_train(expt_conf, load_func, label_func, process_func, dataset_cls, groups, metrics_names=None):
    if expt_conf['cv_name']:
        experimentor = CrossValidator(expt_conf, load_func, label_func, process_func, dataset_cls, expt_conf['cv_name'],
                                      expt_conf['n_splits'], groups)
    else:
        experimentor = BaseExperimentor(expt_conf, load_func, label_func, process_func, dataset_cls)

    phases = ['train', 'val']

    if not metrics_names:
        metrics = get_metrics(phases, expt_conf['task_type'], expt_conf['train_manager'])
    else:
        metrics = {p: get_metric_list(metrics_names[p]) for p in phases}

    result_series, val_pred = experimentor.train_with_validation(metrics)
    val_metric_names = [m.name for m in metrics['val']]
    mlflow.log_metrics({metric_name: value for metric_name, value in zip(val_metric_names, result_series)})

    return result_series, val_pred


def typical_experiment(expt_conf, load_func, label_func, process_func, dataset_cls, groups, metrics_names=None):
    infer = np.array(['infer' in key.replace('_path', '') for key in expt_conf.keys()]).any()
    if expt_conf['cv_name']:
        experimentor = CrossValidator(expt_conf, load_func, label_func, process_func, dataset_cls, expt_conf['cv_name'],
                                      expt_conf['n_splits'], groups)
    else:
        experimentor = BaseExperimentor(expt_conf, load_func, label_func, process_func, dataset_cls)

    phases = ['train', 'val', '']

    if not metrics_names:
        metrics = get_metrics(phases, cfg['task_type'], cfg['train_dataloader'])
    else:
        metrics = {p: get_metric_list[metrics_names[p]] for p in phases}

    result_series, pred_list = experimentor.experiment_with_validation(metrics)
    val_metric_names = [m.name for m in metrics['val']]
    mlflow.log_metrics({metric_name: value for metric_name, value in zip(val_metric_names, result_series)})

    return result_series, pred_list