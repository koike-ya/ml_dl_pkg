from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from ml.models.model_manager import model_manager_args, BaseModelManager
from ml.src.metrics import Metric


PHASES = ['train', 'val', 'test']


def train_manager_args(parser):
    train_parser = parser.add_argument_group('train arguments')
    train_parser.add_argument('--only-test', action='store_true', help='Load learned model and not training')
    train_parser.add_argument('--k-fold', type=int, default=5,
                              help='The number of folds. 1 means training with whole train data')
    train_parser.add_argument('--test', action='store_true', help='Do testing, You should be specify k-fold with 1.')
    train_parser.add_argument('--infer', action='store_true', help='Do inference with test_path data,')
    train_parser.add_argument('--manifest-path', help='data file for training', default='input/train.csv')
    parser = model_manager_args(parser)

    return parser


class TrainManager:
    def __init__(self, train_conf, load_func, label_func, dataset_cls, set_dataloader_func, metrics, process_func=None):
        self.train_conf = train_conf
        self.load_func = load_func
        self.label_func = label_func
        self.process_func = process_func
        self.dataset_cls = dataset_cls
        self.set_dataloader_func = set_dataloader_func
        self.metrics = metrics
        self.manifest_df = pd.read_csv(train_conf['manifest_path'], header=None).sample(frac=1)

    def _init_model_manager(self, dataloaders):
        return BaseModelManager(self.train_conf['class_names'], self.train_conf, dataloaders, deepcopy(self.metrics))

    def _normal_cv(self, fold_count, k):
        all_labels = self.manifest_df.apply(self.label_func, axis=1)

        data_dfs = {}
        for class_ in self.train_conf['class_names']:
            data_dfs[class_] = self.manifest_df[all_labels == class_].reset_index(drop=True)

        test_path_df = pd.DataFrame()
        val_path_df = pd.DataFrame()
        train_path_df = pd.DataFrame()
        for class_, label_df in data_dfs.items():
            one_phase_length = len(label_df) // k
            start_index = fold_count * one_phase_length
            leave_out = label_df.iloc[start_index:start_index + one_phase_length, :]
            test_path_df = pd.concat([test_path_df, leave_out]).reset_index(drop=True)

            train_val_df = label_df[~label_df.index.isin(leave_out.index)].reset_index(drop=True)
            val_start_index = (fold_count % (k - 1)) * one_phase_length
            leave_out = train_val_df.iloc[val_start_index:val_start_index + one_phase_length, :]
            val_path_df = pd.concat([val_path_df, leave_out])

            train_path_df = pd.concat([train_path_df, train_val_df[~train_val_df.index.isin(leave_out.index)]])

        return train_path_df, val_path_df, test_path_df

    def _train_test(self, model=None):
        # dataset, dataloaderの作成
        dataloaders = {}
        for phase in PHASES:
            dataset = self.dataset_cls(self.train_conf[f'{phase}_path'], self.train_conf, process_func=self.process_func,
                                       load_func=self.load_func, label_func=self.label_func, phase=phase)
            dataloaders[phase] = self.set_dataloader_func(dataset, phase, self.train_conf)
            if self.train_conf['loss_weight'] == 'balanced':
                self.train_conf['loss_weight'] = dataloaders['train'].get_label_balance()

        model_manager = self._init_model_manager(dataloaders)

        model_manager.train(model)
        _, _, metrics = model_manager.test(return_metrics=True)

        return metrics, model_manager

    def _update_data_paths(self, fold_count: int, k: int):
        # fold_count...k-foldのうちでいくつ目か

        train_path_df, val_path_df, test_path_df = self._normal_cv(fold_count, k)

        for phase in PHASES:
            file_name = f"{Path(self.train_conf['manifest_path'])}_{phase}_path_fold.csv"

            locals()[f'{phase}_path_df'].to_csv(file_name, index=False, header=None)
            self.train_conf[f'{phase}_path'] = str(file_name)
            print(f'{phase} data:\n', locals()[f'{phase}_path_df'].apply(self.label_func, axis=1).value_counts())

    def _train_test_cv(self):
        orig_train_path = self.train_conf['train_path']

        if Path(self.train_conf['model_path']).is_file():
            Path(self.train_conf['model_path']).unlink()

        if self.train_conf['k_fold'] == 0:
            # データ全体で学習を行う
            raise NotImplementedError

        val_cv_metrics = {metric.name: np.zeros(self.train_conf['k_fold']) for metric in self.metrics['val']}
        test_cv_metrics = {metric.name: np.zeros(self.train_conf['k_fold']) for metric in self.metrics['test']}

        # TODO k_foldが1のときにleave_one_outするかval_path, test_pathも読み込むようにする
        for i in range(self.train_conf['k_fold']):
            self._update_data_paths(i, self.train_conf['k_fold'])

            result_metrics, model_manager = self._train_test()

            print(f'Fold {i + 1} ended.')
            for phase in ['val', 'test']:
                metrics = result_metrics[phase]
                for metric in metrics:
                    locals()[f'{phase}_cv_metrics'][metric.name][i] = metric.average_meter.best_score
                # print(f"Metric {metric.name} best score: {metric.average_meter['val'].best_score}")

            for metrics in result_metrics.values():
                for metric in metrics:
                    metric.average_meter.reset()

        [print(f'{i + 1} fold {metric_name} score\t mean: {meter.mean() :.4f}\t std: {meter.std() :.4f}') for
         metric_name, meter in test_cv_metrics.items()]

        # 新しく作成したマニフェストファイルは削除
        [Path(self.train_conf[f'{phase}_path']).unlink() for phase in PHASES]

        return model_manager, val_cv_metrics, test_cv_metrics

    def test(self, model_manager=None) -> List[Metric]:
        if not model_manager:
            # dataset, dataloaderの作成
            dataloaders = {}
            dataset = self.dataset_cls(self.train_conf[f'test_path'], self.train_conf,
                                       load_func=self.load_func, label_func=self.label_func)
            dataloaders['test'] = self.set_dataloader_func(dataset, 'test', self.train_conf)

            model_manager = self._init_model_manager(dataloaders)

        return model_manager.test(return_metrics=True)

    def train_test(self):
        return self._train_test_cv()

    def infer(self) -> np.array:
        phase = 'infer'
        # dataset, dataloaderの作成
        dataloaders = {}
        dataset = self.dataset_cls(self.train_conf[f'test_path'], self.train_conf,
                                   load_func=self.load_func, label_func=self.label_func)
        dataloaders[phase] = self.set_dataloader_func(dataset, phase, self.train_conf)

        # modelManagerをインスタンス化、inferの実行
        model_manager = self._init_model_manager(dataloaders)
        return model_manager.infer()
