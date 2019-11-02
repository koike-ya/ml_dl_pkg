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
            for phase in ['train', 'val']:
                data_df = pd.concat([data_df, pd.read_csv(self.train_conf[f'{phase}_path'], header=None)])
        else:
            for phase in ['train', 'val']:
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

    def _train(self):
        # dataset, dataloaderの作成
        dataloaders = {}
        for phase in ['train', 'val']:
            dataset = self.dataset_cls(self.train_conf[f'{phase}_path'], self.train_conf, phase,
                                       load_func=self.load_func, label_func=self.label_func)
            dataloaders[phase] = self.set_dataloader_func(dataset, phase, self.train_conf)

        model_manager = self._init_model_manager(dataloaders)

        return model_manager.train(), model_manager.model

    def _update_data_paths(self, fold_count: int):
        # fold_count...k-foldのうちでいくつ目か

        train_path_df = pd.DataFrame()
        val_path_df = pd.DataFrame()

        for class_ in self.train_conf['class_names']:
            one_val_length = len(self.each_label_df[class_]) // self.train_conf['k_fold']
            val_start_index = fold_count * one_val_length
            val_leave = self.each_label_df[class_].iloc[val_start_index:val_start_index + one_val_length, :]
            val_path_df = pd.concat([val_path_df, val_leave])

            train_leave = self.each_label_df[class_][~self.each_label_df[class_].index.isin(val_path_df.index)]
            train_path_df = pd.concat([train_path_df, train_leave])

        val_file_name = self.train_conf['val_path'][:-4].replace('_fold', '') + '_fold.csv'
        val_path_df.to_csv(val_file_name, index=False, header=None)
        self.train_conf['val_path'] = val_file_name

        train_file_name = self.train_conf['train_path'][:-4].replace('_fold', '') + '_fold.csv'
        train_path_df.to_csv(train_file_name, index=False, header=None)
        self.train_conf['train_path'] = train_file_name

    def _train_k_fold(self):
        orig_train_path = self.train_conf['train_path']

        k_fold_metrics = {metric.name: np.zeros(self.train_conf['k_fold']) for metric in self.metrics}
        for i in range(self.train_conf['k_fold']):
            if Path(self.train_conf['model_path']).is_file():
                Path(self.train_conf['model_path']).unlink()

            self._update_data_paths(i)

            result_metrics, model = self._train()

            print(f'Fold {i + 1} ended.')
            for metric in result_metrics:
                k_fold_metrics[metric.name][i] = metric.average_meter['val'].best_score
                # print(f"Metric {metric.name} best score: {metric.average_meter['val'].best_score}")

        [print(f'{i + 1} fold {metric_name} score\t mean: {meter.mean()}\t std: {meter.std()}')
         for metric_name, meter in k_fold_metrics.items()]

        if orig_train_path != self.train_conf['train_path']:  # 新しく作成したマニフェストファイルは削除
            [Path(self.train_conf[f'{phase}_path']).unlink() for phase in ['train', 'val']]

        return model

    def train(self):

        # モデルの学習を行う場合
        if not self.train_conf['only_test']:
            if self.train_conf['k_fold']:
                return self._train_k_fold()
            else:
                result_metrics, model = self._train()

            return model

    def test(self) -> List[Metric]:
        # dataset, dataloaderの作成
        dataloaders = {}
        dataset = self.dataset_cls(self.train_conf[f'test_path'], self.train_conf, phase='test',
                                   load_func=self.load_func, label_func=self.label_func)
        dataloaders['test'] = self.set_dataloader_func(dataset, 'test', self.train_conf)

        model_manager = self._init_model_manager(dataloaders)

        return model_manager.test()

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
