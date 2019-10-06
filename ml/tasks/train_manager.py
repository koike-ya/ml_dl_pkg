import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
import argparse
from copy import deepcopy
from ml.src.metrics import Metric
from ml.models.model_manager import model_manager_args, BaseModelManager
from ml.src.metrics import AverageMeter
import torch

LABELS = {'none': 0, 'seiz': 1, 'arch': 2}


def train_manager_args(parser):
    train_parser = parser.add_argument_group('train arguments')
    train_parser.add_argument('--only-model-test', action='store_true', help='Load learned model and not training')
    train_parser.add_argument('--k-fold', type=int, default=0,
                              help='The number of folds. 1 means training with whole train data')
    train_parser.add_argument('--test', action='store_true', help='Do testing, You should be specify k-fold with 1.')
    parser = model_manager_args(parser)

    return parser


def label_func(path):
    return LABELS[path.split('/')[-1].replace('.pkl', '').split('_')[-1]]


def load_func(path):
    return torch.from_numpy(eeg.load_pkl(path).values.reshape(-1, ))


class TrainManager:
    def __init__(self, train_conf, load_func, label_func, dataset_cls, set_dataloader_func, metrics):
        self.train_conf = train_conf
        self.train_conf['load_func'] = load_func
        self.train_conf['label_func'] = label_func
        self.dataset_cls = dataset_cls
        self.set_dataloader_func = set_dataloader_func
        self.metrics = metrics
        self.is_manifest = 'manifest' in self.train_conf['train_path']
        self.data_df = self._set_data_df()

    def _set_data_df(self):
        if self.is_manifest:
            data_df = pd.DataFrame()
            for phase in ['train', 'val']:
                data_df = pd.concat([data_df, pd.read_csv(self.train_conf[f'{phase}_path'], header=None)])
        else:
            raise NotImplementedError

        return data_df

    def _train(self) -> List[Metric]:
        # dataset, dataloaderの作成
        dataloaders = {}
        for phase in ['train', 'val']:
            dataset = self.dataset_cls(self.train_conf[f'{phase}_path'], self.train_conf)
            dataloaders[phase] = self.set_dataloader_func(dataset, phase, self.train_conf)

        # modelManagerをインスタンス化、trainの実行
        model_manager = BaseModelManager(self.train_conf['class_names'], self.train_conf, dataloaders, deepcopy(self.metrics))
        return model_manager.train()

    def _update_data_paths(self, fold_count: int):
        # fold_count...k-foldのうちでいくつ目か

        one_val_length = len(self.data_df) // self.train_conf['k_fold']
        val_start_index = fold_count * one_val_length
        val_path_df = self.data_df.iloc[val_start_index:val_start_index + one_val_length, :]
        val_path_df.to_csv(self.train_conf['val_path'][:-4].replace('_fold', '') + '_fold.csv', index=False, header=None)
        self.train_conf['val_path'] = self.train_conf['val_path'][:-4].replace('_fold', '') + '_fold.csv'
        train_df = self.data_df[~self.data_df.index.isin(val_path_df.index)]
        train_df.to_csv(self.train_conf['train_path'][:-4].replace('_fold', '') + '_fold.csv', index=False, header=None)
        self.train_conf['train_path'] = self.train_conf['train_path'][:-4].replace('_fold', '') + '_fold.csv'

    def train(self):
        orig_train_path = self.train_conf['train_path']

        # モデルの学習を行う場合
        if not self.train_conf['only_model_test']:
            if self.train_conf['k_fold']:
                k_fold_metrics = {metric.name: np.zeros(self.train_conf['k_fold']) for metric in self.metrics}
                for i in range(self.train_conf['k_fold']):
                    if Path(self.train_conf['model_path']).is_file():
                        Path(self.train_conf['model_path']).unlink()

                    self._update_data_paths(i)

                    result_metrics = self._train()
                    print(f'Fold {i + 1} ended.')
                    for metric in result_metrics:
                        k_fold_metrics[metric.name][i] = metric.average_meter['val'].best_score
                        # print(f"Metric {metric.name} best score: {metric.average_meter['val'].best_score}")

                [print(f'{i + 1} fold {metric_name} score\t mean: {meter.mean()}\t std: {meter.std()}')
                 for metric_name, meter in k_fold_metrics.items()]

                if orig_train_path != self.train_conf['train_path']:    # 新しく作成したマニフェストファイルは削除
                    [Path(self.train_conf[f'{phase}_path']).unlink() for phase in ['train', 'val']]

            else:
                result_metrics = self._train()

    def test(self) -> List[Metric]:
        # dataset, dataloaderの作成
        dataloaders = {}
        dataset = self.dataset_cls(self.train_conf[f'test_path'], self.train_conf)
        dataloaders['test'] = self.set_dataloader_func(dataset, 'test', self.train_conf)

        # modelManagerをインスタンス化、trainの実行
        model_manager = BaseModelManager(self.train_conf['class_names'], self.train_conf, dataloaders, self.metrics)
        return model_manager.test()
