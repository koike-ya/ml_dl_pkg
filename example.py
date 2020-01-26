import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ml.src.dataset import ManifestWaveDataSet
from librosa.core import load
from ml.models.model_manager import BaseModelManager
from ml.models.pretrained_models import supported_pretrained_models
from ml.src.dataloader import set_dataloader, set_ml_dataloader
from ml.src.metrics import metrics2df, Metric
from ml.src.preprocessor import Preprocessor, preprocess_args
from ml.tasks.train_manager import TrainManager, train_manager_args
from ml.src.gradcam import gradcam_main

DATALOADERS = {'normal': set_dataloader, 'ml': set_ml_dataloader}


def train_args(parser):
    train_manager_args(parser)
    expt_parser = parser.add_argument_group("Experiment arguments")
    expt_parser.add_argument('--expt-id', help='data file for training', default='')
    expt_parser.add_argument('--dataloader-type', help='Dataloader type.', choices=['normal', 'ml'], default='normal')

    return parser


def label_func(row):
    labels = ['F', 'N', 'O', 'S', 'Z']
    return labels.index(row[0].split('/')[-1][0])


def set_load_func(sr, one_audio_sec):
    def load_func(path):
        const_length = int(sr * one_audio_sec)
        with open(path[0], 'r') as f:
            wave = np.array(list(map(float, f.read().split('\n')[:-1])))
        if wave.shape[0] > const_length:
            wave = wave[:const_length]
        elif wave.shape[0] < const_length:
            n_pad = (const_length - wave.shape[0]) // 2 + 1
            wave = np.pad(wave[:const_length], n_pad)[:const_length]
        return wave.reshape((1, -1))

    return load_func


def create_manifest():
    DATA_DIR = Path(__file__).resolve().parent / 'input'
    path_list = []

    for dir in [d for d in DATA_DIR.iterdir() if d.is_dir()]:
        path_list.extend([str(p.resolve()) for p in dir.iterdir()])
    path_list.sort()

    pd.Series(path_list).to_csv(DATA_DIR / 'bonn_manifest.csv', index=False, header=None)


def experiment(train_conf) -> float:
    phases = ['train', 'val', 'test']

    train_conf['class_names'] = [0, 1, 2, 3, 4]
    train_conf['manifest_path'] = DATA_DIR = Path(__file__).resolve().parent / 'input' / 'bonn_manifest.csv'

    one_audio_sec = 10
    sr = 173.61

    set_dataloader_func = set_dataloader
    dataset_cls = ManifestWaveDataSet
    process_func = Preprocessor(train_conf, phase='test', sr=sr).preprocess

    metrics = dict(
        train=[
            Metric('loss', direction='minimize', save_model=True),
            Metric('uar', direction='maximize')],
        val=[
            Metric('loss', direction='minimize', save_model=True),
            Metric('uar', direction='maximize')],
        test=[
            Metric('loss', direction='minimize', save_model=True),
            Metric('uar', direction='maximize'),
            Metric('accuracy', direction='maximize')]
    )

    load_func = set_load_func(sr, one_audio_sec)
    train_manager = TrainManager(train_conf, load_func, label_func, dataset_cls, set_dataloader_func, metrics,
                                 process_func=process_func)
    model_manager, _, test_cv_metrics = train_manager.train_test()

    return test_cv_metrics['uar'].mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    train_conf = vars(train_args(preprocess_args(parser)).parse_args())
    assert train_conf['train_path'] != '' or train_conf['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'

    create_manifest()

    res = []

    # for seed in range(3):
    #     train_conf['seed'] = seed
    res.append(experiment(train_conf))

    print(np.array(res).mean())
    print(np.array(res).std())
