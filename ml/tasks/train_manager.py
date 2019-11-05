import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
import argparse
from copy import deepcopy
from ml.src.metrics import Metric
from ml.models.model_manager import model_manager_args, BaseModelManager
from ml.models.keras_model_manager import KerasModelManager
from ml.src.metrics import AverageMeter
import torch

LABELS = {'none': 0, 'seiz': 1, 'arch': 2}
PHASES = ['train', 'val', 'test']


def train_manager_args(parser):
    train_parser = parser.add_argument_group('train arguments')
    train_parser.add_argument('--only-test', action='store_true', help='Load learned model and not training')
    train_parser.add_argument('--k-fold', type=int, default=0,
                              help='The number of folds. 1 means training with whole train data')
    train_parser.add_argument('--test', action='store_true', help='Do testing, You should be specify k-fold with 1.')
    train_parser.add_argument('--infer', action='store_true', help='Do inference with test_path data,')
    train_parser.add_argument('--model-manager', default='pytorch')
    parser = model_manager_args(parser)

    return parser


def label_func(path):
    return LABELS[path.split('/')[-1].replace('.pkl', '').split('_')[-1]]


def load_func(path):
    return torch.from_numpy(eeg.load_pkl(path).values.reshape(-1, ))


class TrainManager:
    def __init__(self, train_conf, load_func, label_func, dataset_cls, set_dataloader_func, metrics):
        self.train_conf = train_conf
        self.load_func = load_func
        self.label_func = label_func
        self.dataset_cls = dataset_cls
        self.set_dataloader_func = set_dataloader_func
        self.metrics = metrics
        self.is_manifest = 'manifest' in self.train_conf['train_path']
        self.each_label_df = self._set_each_label_df()

    def _set_each_label_df(self):
        each_label_df = {}
        data_df = pd.DataFrame()

        if self.is_manifest:
            for phase in PHASES:
                data_df = pd.concat([data_df, pd.read_csv(self.train_conf[f'{phase}_path'], header=None)])
        else:
            for phase in PHASES:
                data_df = pd.concat([data_df, self.load_func(self.train_conf[f'{phase}_path'])])

        all_labels = data_df.squeeze().apply(lambda x: self.label_func(x))

        for class_ in self.train_conf['class_names']:
            each_label_df[class_] = data_df[all_labels == class_]

        return each_label_df

    def _init_model_manager(self, dataloaders):
        # modelManagerをインスタンス化、trainの実行
        if self.train_conf['model_manager'] == 'pytorch':
            model_manager = BaseModelManager(self.train_conf['class_names'], self.train_conf, dataloaders,
                                             deepcopy(self.metrics))
        elif self.train_conf['model_manager'] == 'keras':
            model_manager = KerasModelManager(self.train_conf['class_names'], self.train_conf, dataloaders,
                                              deepcopy(self.metrics))
        else:
            raise NotImplementedError

        return model_manager

    def _train_test(self):
        # dataset, dataloaderの作成
        dataloaders = {}
        for phase in PHASES:
            dataset = self.dataset_cls(self.train_conf[f'{phase}_path'], self.train_conf, phase,
                                       load_func=self.load_func, label_func=self.label_func)
            dataloaders[phase] = self.set_dataloader_func(dataset, phase, self.train_conf)

        model_manager = self._init_model_manager(dataloaders)

        model_manager.train()
        _, _, test_metrics = model_manager.test(return_metrics=True)

        return test_metrics, model_manager.model

    def _update_data_paths(self, fold_count: int, k: int):
        # fold_count...k-foldのうちでいくつ目か

        test_path_df = pd.DataFrame()
        val_path_df = pd.DataFrame()
        train_path_df = pd.DataFrame()
        for class_, label_df in self.each_label_df.items():
            one_phase_length = len(label_df) // self.train_conf['k_fold']
            start_index = fold_count * one_phase_length
            leave_out = label_df.iloc[start_index:start_index + one_phase_length, :]
            test_path_df = pd.concat([test_path_df, leave_out])

            train_val_df = label_df[~label_df.index.isin(test_path_df.index)].reset_index(drop=True)
            val_start_index = (fold_count % (k - 1)) * one_phase_length
            leave_out = label_df.iloc[val_start_index:val_start_index + one_phase_length, :]
            val_path_df = pd.concat([val_path_df, leave_out])

            train_path_df = pd.concat([train_path_df, train_val_df[~train_val_df.index.isin(leave_out.index)]])

        for phase in PHASES:
            file_name = self.train_conf[f'{phase}_path'][:-4].replace('_fold', '') + '_fold.csv'
            locals()[f'{phase}_path_df'].to_csv(file_name, index=False, header=None)
            self.train_conf[f'{phase}_path'] = file_name

    def _train_test_k_fold(self):
        orig_train_path = self.train_conf['train_path']

        k_fold_metrics = {metric.name: np.zeros(self.train_conf['k_fold']) for metric in self.metrics}

        if self.train_conf['k_fold'] == 0:
            # データ全体で学習を行う
            raise NotImplementedError

        for i in range(self.train_conf['k_fold']):
            if Path(self.train_conf['model_path']).is_file():
                Path(self.train_conf['model_path']).unlink()

            self._update_data_paths(i, self.train_conf['k_fold'])

            result_metrics, model = self._train_test()

            print(f'Fold {i + 1} ended.')
            for metric in result_metrics:
                k_fold_metrics[metric.name][i] = metric.average_meter['test'].best_score
                # print(f"Metric {metric.name} best score: {metric.average_meter['val'].best_score}")

        [print(f'{i + 1} fold {metric_name} score\t mean: {meter.mean()}\t std: {meter.std()}') for metric_name, meter
         in k_fold_metrics.items()]

        # 新しく作成したマニフェストファイルは削除
        [Path(self.train_conf[f'{phase}_path']).unlink() for phase in PHASES]

        return model

    def test(self, model_manager=None) -> List[Metric]:
        if not model_manager:
            # dataset, dataloaderの作成
            dataloaders = {}
            dataset = self.dataset_cls(self.train_conf[f'test_path'], self.train_conf, phase='test',
                                       load_func=self.load_func, label_func=self.label_func)
            dataloaders['test'] = self.set_dataloader_func(dataset, 'test', self.train_conf)

            model_manager = self._init_model_manager(dataloaders)

        return model_manager.test()

    def train_test(self):
        if not self.train_conf['only_test']:
            self._train_test_k_fold()
        else:
            self.test()

    def infer(self) -> np.array:
        phase = 'infer'
        # dataset, dataloaderの作成
        dataloaders = {}
        dataset = self.dataset_cls(self.train_conf[f'test_path'], self.train_conf, phase=phase,
                                   load_func=self.load_func, label_func=self.label_func)
        dataloaders[phase] = self.set_dataloader_func(dataset, phase, self.train_conf)

        # modelManagerをインスタンス化、inferの実行
        model_manager = BaseModelManager(self.train_conf['class_names'], self.train_conf, dataloaders, self.metrics)
        return model_manager.infer()
