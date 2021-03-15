import logging
import tempfile
from abc import ABCMeta
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Tuple, Dict, List

import mlflow
import numpy as np
import pandas as pd

from ml.models.train_managers.base_train_manager import TrainConfig
from ml.models.train_managers.ml_train_manager import MLTrainManager
from ml.models.train_managers.multitask_train_manager import MultitaskTrainManager
from ml.models.train_managers.nn_train_manager import NNTrainManager
from ml.preprocess.transforms import TransConfig
from ml.src.cv_manager import KFoldManager
from ml.src.cv_manager import SupportedCV
from ml.src.dataloader import DataConfig
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.src.metrics import get_metric_list
from ml.utils.enums import TrainManagerType, DataLoaderType
from ml.utils.utils import Metrics

logger = logging.getLogger(__name__)

DATALOADERS = {'normal': set_dataloader, 'ml': set_ml_dataloader}
TRAINMANAGERS = {'nn': NNTrainManager, 'multitask': MultitaskTrainManager, 'ml': MLTrainManager}


@dataclass
class BaseExptConfig:
    expt_id: str = 'timestamp'      # Data file for training
    manifest_path: str = 'input/train.csv'  # Manifest file for training
    n_seed_average: int = 0         # Seed averaging
    cv_name: SupportedCV = SupportedCV.none     # CV options
    n_splits: int = 0               # Number of splits on cv
    infer: bool = False             # Whether training with train+devel dataset after hyperparameter tuning
    test: bool = False  # Whether training with train+devel dataset after hyperparameter tuning
    data_loader: DataLoaderType = DataLoaderType.normal
    train_manager: TrainManagerType = TrainManagerType.nn

    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()
    transformers: List[TransConfig] = field(default_factory=lambda: [TransConfig()])


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
    def __init__(self, cfg, load_func, label_func, process_func=None, dataset_cls=None, collate_fn=None):
        self.cfg = cfg
        self.load_func = load_func
        self.label_func = label_func
        self.dataset_cls = dataset_cls
        self.data_loader_cls = DATALOADERS[cfg.data_loader.value]
        self.train_manager_cls = TRAINMANAGERS[cfg.train_manager.value]
        self.train_manager = None
        self.process_func = process_func
        self.collate_fn = collate_fn
        self.infer = cfg.infer
        
    def _experiment(self, metrics, phases) -> Tuple[Metrics, Dict[str, np.array]]:
        if not isinstance(self.process_func, dict):
            self.process_func = {phase: self.process_func for phase in phases}

        dataloaders = {}
        for phase in phases:
            dataset = self.dataset_cls(self.cfg.train[f'{phase}_path'], self.cfg.data, phase, self.load_func,
                                       self.process_func[phase], self.label_func)
            dataloaders[phase] = self.data_loader_cls(dataset, phase, self.cfg.data, collate_fn=self.collate_fn)

        self.train_manager = self.train_manager_cls(self.cfg.train['class_names'], self.cfg.train, dataloaders,
                                                    deepcopy(metrics))

        pred_list = {}
        if 'val' in phases:
            metrics, pred_list['val'] = self.train_manager.train()
        elif 'train' in phases:  # This is the case in ['train', 'infer'], ['train', 'test']
            metrics, _ = self.train_manager.train(with_validate=False)

        if 'infer' in phases:
            pred_list['infer'] = self.train_manager.infer()
        elif 'test' in phases:
            pred_list['test'], _, metrics = self.train_manager.test(return_metrics=True)

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
        if self.infer:
            return {'val': np.array([m.average_meter.best_score for m in metrics['val']])}, pred_list
        else:
            result_series = {'val': np.array([m.average_meter.best_score for m in metrics['val']]),
                             'test': np.array([m.average_meter.best_score for m in metrics['test']])}
            return result_series, pred_list

    def experiment_without_validation(self, metrics: Metrics, infer: bool = False, seed_average: int = 0
                                      ) -> Tuple[np.array, np.array]:
        if self.infer:
            phases = ['train', 'infer']
        else:
            phases = ['train', 'test']
            
        if not seed_average:
            metrics, pred_list = self._experiment(metrics=metrics, phases=phases)
            return np.array([m.average_meter.best_score for m in metrics['train']]), pred_list

        else:
            pred_list = []
            metrics_list = []
            for seed in range(seed_average):
                self.cfg.train.model.seed = seed
                metrics, pred = self._experiment(metrics=metrics, phases=phases)
                pred_list.append(pred)
                metrics_list.append(np.array([m.average_meter.best_score for m in metrics['train']]))

            pred_list = [pred[phases[-1]] for pred in pred_list]
            metrics = np.array(metrics_list).mean(axis=0)
            return metrics, np.array(pred_list).mean(axis=0)


class CrossValidator(BaseExperimentor):
    def __init__(self, cfg: Dict, load_func, label_func, process_func, dataset_cls, cv_name: str, n_splits: int,
                 groups: str = None, collate_fn=None):
        super().__init__(cfg, load_func, label_func, process_func, dataset_cls, collate_fn)
        self.orig_cfg = deepcopy(self.cfg)
        self.cv_name = cv_name
        self.n_splits = n_splits
        self.groups = groups
        self.pred_list = []

    def _save_path(self, df_x, train_idx, val_idx, temp_dir):
        df_x.iloc[train_idx, :].to_csv(f'{temp_dir}/train_manifest.csv', header=None, index=False)
        df_x.iloc[val_idx, :].to_csv(f'{temp_dir}/val_manifest.csv', header=None, index=False)
        self.cfg.train[f'train_path'] = f'{temp_dir}/train_manifest.csv'
        self.cfg.train[f'val_path'] = f'{temp_dir}/val_manifest.csv'

    def _set_metrics_df(self, metrics_df, result_list: List[float]):
        metrics_df = pd.concat([metrics_df.T, pd.Series(result_list)], axis=1).T
        assert metrics_df.shape[1] == len(result_list), metrics_df
        return metrics_df

    def train_with_validation(self, metrics: Metrics) -> Tuple[np.array, List[Dict[str, np.array]]]:
        pred_list = []
        metric_df = pd.DataFrame()

        df_x = pd.concat([pd.read_csv(self.orig_cfg.train.train_path, header=None),
                          pd.read_csv(self.orig_cfg.train.val_path, header=None)])
        y = df_x.apply(lambda x: self.label_func(x), axis=1)
        logger.info(y.value_counts())
        
        k_fold = KFoldManager(self.cv_name.value, self.n_splits, groups=self.groups)

        for i, (train_idx, val_idx) in enumerate(k_fold.split(X=df_x.values, y=y.values)):
            logger.info(f'Fold {i + 1} started.')
            with tempfile.TemporaryDirectory() as temp_dir:
                self._save_path(df_x, train_idx, val_idx, temp_dir)

                result_series, fold_pred_list = super().train_with_validation(metrics)
                pred_list.append(fold_pred_list)
                metrics_df = self._set_metrics_df(metric_df, result_series)

        metrics_df.columns = [m.name for m in metrics['val']]
        logger.debug(f'Cross validation metrics:\n{metrics_df}')
        self.cfg.train.train_path = self.orig_cfg.train.train_path
        self.cfg.train.val_path = self.orig_cfg.train.val_path

        return metrics_df.mean(axis=0).values, pred_list

    def experiment_with_validation(self, metrics: Metrics, infer=False) -> Tuple[np.array, List[Dict[str, np.array]]]:
        pred_list = []
        metrics_df = pd.DataFrame()

        df_x = pd.concat([pd.read_csv(self.orig_cfg.train.train_path, header=None),
                          pd.read_csv(self.orig_cfg.train.val_path, header=None)])
        y = df_x.apply(lambda x: self.label_func(x), axis=1)

        logger.info(y.value_counts())

        k_fold = KFoldManager(self.cv_name.value, self.n_splits, groups=self.groups)

        for i, (train_idx, val_idx) in enumerate(k_fold.split(X=df_x.values, y=y.values)):
            logger.info(f'Fold {i + 1} started.')
            with tempfile.TemporaryDirectory() as temp_dir:
                self._save_path(df_x, train_idx, val_idx, temp_dir)

                result_series, fold_pred_list = super().experiment_with_validation(metrics, infer=infer)
                pred_list.append(fold_pred_list)
                metrics_df = self._set_metrics_df(metrics_df, result_series)

        metrics_df.columns = [m.name for m in metrics['val' if self.infer else 'test']]
        logger.debug(f'Cross validation metrics:\n{metrics_df}')
        self.cfg.train.train_path = self.orig_cfg.train.train_path
        self.cfg.train.val_path = self.orig_cfg.train.val_path

        return metrics_df.mean(axis=0).values, pred_list

    def experiment_without_validation(self) -> np.array:
        raise NotImplementedError


def typical_train(expt_conf, load_func, label_func, process_func, dataset_cls, groups=None, metrics_names=None,
                  collate_fn=None):
    if (expt_conf['cv_name'] and expt_conf['cv_name'].value) or isinstance(groups, pd.Series):
        experimentor = CrossValidator(expt_conf, load_func, label_func, process_func, dataset_cls, expt_conf['cv_name'],
                                      expt_conf['n_splits'], groups, collate_fn=collate_fn)
    else:
        experimentor = BaseExperimentor(expt_conf, load_func, label_func, process_func, dataset_cls, collate_fn)

    phases = ['train', 'val']

    if not metrics_names:
        metrics = get_metrics(phases, expt_conf.train.task_type.value, expt_conf['train_manager'])
    else:
        metrics = {p: get_metric_list(metrics_names[p]) for p in phases}

    result_series, val_pred = experimentor.train_with_validation(metrics)
    val_metric_names = [m.name for m in metrics['val']]
    mlflow.log_metrics({metric_name: value for metric_name, value in zip(val_metric_names, result_series)})

    return result_series, val_pred, experimentor


def typical_experiment(expt_conf, load_func, label_func, process_func, dataset_cls, groups, metrics_names=None):
    infer = 'infer_path' in expt_conf.keys()
    if (expt_conf['cv_name'] and expt_conf['cv_name'].value) or isinstance(groups, pd.Series):
        experimentor = CrossValidator(expt_conf, load_func, label_func, process_func, dataset_cls, expt_conf['cv_name'],
                                      expt_conf['n_splits'], groups)
    else:
        experimentor = BaseExperimentor(expt_conf, load_func, label_func, process_func, dataset_cls)

    phases = ['train', 'val', 'infer'] if infer else ['train', 'val', 'test']

    if not metrics_names:
        metrics = get_metrics(phases, expt_conf.train.task_type.value, expt_conf['train_manager'])
    else:
        metrics = {p: get_metric_list(metrics_names[p]) for p in phases}

    result_series, pred_list = experimentor.experiment_with_validation(metrics)
    # metric_names_list = [m.name for m in metrics['val' if infer else 'test']]
    for phase in ['val'] if infer else ['val', 'test']:
        metric_names = [m.name for m in metrics[phase]]
        mlflow.log_metrics({f'{phase}_{metric_name}': value for metric_name, value in zip(metric_names, result_series[phase])})

    return result_series, pred_list, experimentor