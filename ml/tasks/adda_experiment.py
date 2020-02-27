import logging
import tempfile
from pathlib import Path
from typing import Tuple, List, Union

import mlflow
import numpy as np
import pandas as pd
from ml.models.train_managers.adda_train_manager import AddaTrainManager
from ml.src.metrics import get_metrics
from ml.src.preprocessor import Preprocessor
from ml.tasks.base_experiment import base_expt_args, BaseExperimentor
from ml.utils.utils import Metrics

logger = logging.getLogger(__name__)


def adda_expt_args(parser):
    parser = base_expt_args(parser)
    adda_parser = parser.add_argument_group("ADDA Experiment arguments")
    # adda_parser.add_argument('--iterations', type=int, default=500)
    adda_parser.add_argument('--adda-epochs', type=int, default=5)
    adda_parser.add_argument('--k-disc', type=int, default=5)
    adda_parser.add_argument('--k-clf', type=int, default=10)
    return parser


class AddaExperimentor(BaseExperimentor):
    def __init__(self, cfg, load_func, label_func, dataset_cls):
        super(AddaExperimentor, self).__init__(cfg, load_func, label_func, dataset_cls)
        self.train_manager_cls = AddaTrainManager
        if cfg['task_type'] == 'classify':
            self.train_metrics = get_metrics(['loss', 'uar'])
        else:
            self.train_metrics = get_metrics(['loss'])

    def _set_source_target_dataloaders(self, domain_manifests, dataloaders, temp_dir):
        for domain in domain_manifests.keys():
            df = pd.concat([pd.read_csv(self.cfg[f'{phase}_path'], header=None) for phase in domain_manifests[domain]])
            df.to_csv(Path(temp_dir) / f'{domain}_manifest.csv', index=False, header=None)
            self.cfg[f'{domain}_path'] = str(Path(temp_dir) / f'{domain}_manifest.csv')
            process_func = Preprocessor(self.cfg, domain, self.cfg['sample_rate']).preprocess
            dataset = self.dataset_cls(self.cfg[f'{domain}_path'], self.cfg, self.load_func, process_func,
                                       self.label_func, domain)
            dataloaders[domain] = self.data_loader_cls(dataset, domain, self.cfg)

        return dataloaders

    def _experiment(self, val_metrics, phases) -> Union[Tuple[Metrics, np.array], Tuple[Metrics, np.array, np.array]]:
        dataloaders = {}
        for phase in phases:
            process_func = Preprocessor(self.cfg, phase, self.cfg['sample_rate']).preprocess
            dataset = self.dataset_cls(self.cfg[f'{phase}_path'], self.cfg, self.load_func, process_func,
                                       self.label_func, phase)
            dataloaders[phase] = self.data_loader_cls(dataset, phase, self.cfg)

        if 'infer' in phases:
            domain_manifests = {'source': ['train', 'val'], 'target': ['infer']}
        else:   # Only train and val
            domain_manifests = {'source': ['train'], 'target': ['val']}

        with tempfile.TemporaryDirectory() as temp_dir:
            dataloaders = self._set_source_target_dataloaders(domain_manifests, dataloaders, temp_dir)

            if val_metrics:
                val_metrics = get_metrics(val_metrics, target_metric='loss')
                metrics = {'train': self.train_metrics, 'val': val_metrics}
            else:
                metrics = {'train': self.train_metrics}

            train_manager = self.train_manager_cls(self.cfg['class_names'], self.cfg, dataloaders, metrics)

            if 'infer' in phases:
                metrics, val_pred = train_manager.train(with_validate=False)
                pred = train_manager.infer()
                return metrics, val_pred, pred
            else:   # Only train and val
                metrics, val_pred = train_manager.train()
                return metrics, val_pred

    def train_with_validation(self, val_metrics: List[str]) -> Tuple[np.array, np.array]:
        phases = ['train', 'val']
        metrics, val_pred = self._experiment(val_metrics, phases)
        return np.array([m.average_meter.best_score for m in metrics['val']]), val_pred

    def experiment_without_validation(self, seed_average=0) -> np.array:
        phases = ['train', 'val', 'infer']

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


def typical_adda_train(expt_conf, load_func, label_func, dataset_cls, groups, val_metrics):
    # if expt_conf['cv_name']:
    #     experimentor = CrossValidator(expt_conf, load_func, label_func, dataset_cls, expt_conf['cv_name'], expt_conf['n_splits'],
    #                                   groups)
    # else:
    experimentor = AddaExperimentor(expt_conf, load_func, label_func, dataset_cls)

    result_series, val_pred = experimentor.train_with_validation(val_metrics)
    mlflow.log_metrics({metric_name: value for metric_name, value in zip(val_metrics, result_series)})

    return result_series, val_pred


# def typical_adda_experiment(expt_conf, load_func, label_func, dataset_cls, groups, val_metrics):
#     if expt_conf['cv_name']:
#         experimentor = CrossValidator(expt_conf, load_func, label_func, dataset_cls, expt_conf['cv_name'], expt_conf['n_splits'],
#                                       groups)
#     else:
#         experimentor = BaseExperimentor(expt_conf, load_func, label_func, dataset_cls)
#
#     result_series, pred = experimentor.experiment_with_validation(val_metrics)
#     mlflow.log_metrics({metric_name: value for metric_name, value in zip(val_metrics, result_series)})
#
#     return result_series, pred