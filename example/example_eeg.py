import itertools
import logging
import pprint
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from hydra import utils
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from ml.models.nn_models.cnn import CNNConfig
from ml.models.nn_models.cnn_rnn import CNNRNNConfig
from ml.models.nn_models.rnn import RNNConfig
from ml.src.dataset import ManifestWaveDataSet
from ml.tasks.base_experiment import typical_train, typical_experiment
from ml.utils.config import ExptConfig, before_hydra
from ml.utils.utils import dump_dict

BINARY_LABELS = {'Z': 0, 'O': 0, 'N': 0, 'F': 0, 'S': 1}


@dataclass
class ExampleEEGConfig(ExptConfig):
    n_parallel: int = 1
    mlflow: bool = False


def label_func(row):
    return BINARY_LABELS[row[0].split('/')[-1][0]]


def set_load_func(sr, one_audio_sec):
    def load_func(row):
        const_length = int(sr * one_audio_sec)
        with open(row[0], 'r') as f:
            wave = np.array(list(map(float, f.read().split('\n')[:-1])))
        if wave.shape[0] > const_length:
            wave = wave[:const_length]
        elif wave.shape[0] < const_length:
            n_pad = (const_length - wave.shape[0]) // 2 + 1
            wave = np.pad(wave[:const_length], n_pad)[:const_length]
        return torch.from_numpy(wave.reshape((1, -1)))

    return load_func


def create_manifest(cfg, expt_dir):
    data_dir = Path(utils.to_absolute_path('input')) / 'eeg'
    path_list = [str(p.resolve()) for p in sorted(list(data_dir.iterdir())) if p.is_file()]

    path_df = pd.DataFrame([path_list]).T.sample(n=500).T
    labels = path_df.apply(label_func)
    negatives = labels[labels == 0]
    positives = labels[labels == 1]

    train_df = pd.concat([path_df[negatives[:int(len(negatives) * 0.6)].index],
                          path_df[positives[:int(len(positives) * 0.6)].index]], axis=1).T
    test_df = pd.concat([path_df[negatives[int(len(negatives) * 0.8):].index],
                         path_df[positives[int(len(positives) * 0.8):].index]], axis=1).T
    val_df = path_df.T[~path_df.columns.isin(train_df.index) & ~path_df.columns.isin(test_df.index)]

    for phase in ['train', 'val', 'test']:
        locals()[f'{phase}_df'].to_csv(expt_dir / f'{phase}_manifest.csv', index=False, header=None)
        cfg.train[f'{phase}_path'] = expt_dir / f'{phase}_manifest.csv'

    return cfg


def set_hyperparameter(expt_conf, param, param_value):
    if len(param.split('.')) == 1:
        expt_conf[param] = param_value
    else:
        tmp = expt_conf
        for attr in param.split('.')[:-1]:
            tmp = getattr(tmp, str(attr))
        setattr(tmp, param.split('.')[-1], param_value)

    return expt_conf


def main(cfg, expt_dir, hyperparameters):
    if cfg.expt_id == 'timestamp':
        cfg.expt_id = dt.today().strftime('%Y-%m-%d_%H:%M')

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')

    cfg.train.class_names = [0, 1]
    cfg.transformer.sample_rate = 173.61

    one_audio_sec = 10
    dataset_cls = ManifestWaveDataSet
    load_func = set_load_func(cfg.transformer.sample_rate, one_audio_sec)
    metrics_names = {'train': ['loss', 'uar'],
                     'val': ['loss', 'uar'],
                     'test': ['loss', 'uar']}

    dataset_cls = ManifestWaveDataSet
    cfg = create_manifest(cfg, expt_dir)
    process_func = None

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(metrics_names['val']))),
                               columns=list(hyperparameters.keys()) + metrics_names['val'])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)
    groups = None

    def experiment(pattern, cfg):
        for i, param in enumerate(hyperparameters.keys()):
            cfg = set_hyperparameter(cfg, param, pattern[i])

        cfg.train.model.model_path = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in pattern])}.pth")
        cfg.train.log_id = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"

        with mlflow.start_run():
            result_series, val_pred, _ = typical_train(cfg, load_func, label_func, process_func, dataset_cls,
                                                       groups)

            mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})

        return result_series, val_pred

    # For debugging
    if cfg.n_parallel == 1:
        result_pred_list = [experiment(pattern, deepcopy(cfg)) for pattern in patterns]
    else:
        cfg.n_jobs = 0
        result_pred_list = Parallel(n_jobs=cfg.n_parallel, verbose=0)(
            [delayed(experiment)(pattern, deepcopy(cfg)) for pattern in patterns])

    val_results.iloc[:, :len(hyperparameters)] = [[str(param) for param in p] for p in patterns]
    result_list = np.array([result for result, pred in result_pred_list])
    val_results.iloc[:, len(hyperparameters):] = result_list
    pp.pprint(val_results)
    pp.pprint(val_results.iloc[:, len(hyperparameters):].describe())

    val_results.to_csv(expt_dir / 'val_results.csv', index=False)
    print(f"Devel results saved into {expt_dir / 'val_results.csv'}")
    for (_, _), pattern in zip(result_pred_list, patterns):
        pattern_name = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
        dump_dict(expt_dir / f'{pattern_name}.txt', cfg)

    # Train with train + devel dataset
    if cfg.test:
        best_trial_idx = val_results['uar'].argmax()

        best_pattern = patterns[best_trial_idx]
        for i, param in enumerate(hyperparameters.keys()):
            cfg = set_hyperparameter(cfg, param, best_pattern[i])

        dump_dict(expt_dir / 'best_parameters.txt', {p: v for p, v in zip(hyperparameters.keys(), best_pattern)})

        metrics, pred_dict_list, _ = typical_experiment(cfg, load_func, label_func, process_func, dataset_cls,
                                                        groups)

        sub_name = f"uar-{metrics[-1]:.4f}_sub_{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.csv"
        pd.DataFrame(pred_dict_list['test']).to_csv(expt_dir / f'{sub_name}_prob.csv', index=False, header=None)
        pd.DataFrame(pred_dict_list['test'].argmax(axis=1)).to_csv(expt_dir / sub_name, index=False, header=None)
        print(f"Submission file is saved in {expt_dir / sub_name}")

    mlflow.end_run()


@hydra.main(config_name="config")
def hydra_main(cfg: ExampleEEGConfig):
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.INFO)
    logging.getLogger("ml").addHandler(console)

    if OmegaConf.get_type(cfg.train.model) == CNNConfig:
        hyperparameters = {
            'transformer.transform': ['none'],
            'train.model.channel_list': [[4, 8, 16, 32]],
            'train.model.kernel_sizes': [[[4]] * 4],
            'train.model.stride_sizes': [[[2]] * 4],
            'train.model.padding_sizes': [[[1]] * 4],
            'train.model.optim.lr': [1e-4],
        }
    elif OmegaConf.get_type(cfg.train.model) == CNNRNNConfig:
        hyperparameters = {
            'train.model.optim.lr': [1e-3, 1e-4, 1e-5],
            'transformer.transform': ['none'],
            'train.model.channel_list': [[4, 8, 16, 32]],
            'train.model.kernel_sizes': [[[4]] * 4],
            'train.model.stride_sizes': [[[2]] * 4],
            'train.model.padding_sizes': [[[1]] * 4],
            'train.model.rnn_type': [cfg.train.model.rnn_type],
            'train.model.bidirectional': [True],
            'train.model.rnn_n_layers': [1, 2],
            'train.model.rnn_hidden_size': [10, 50],
        }
    elif OmegaConf.get_type(cfg.train.model) == RNNConfig:
        hyperparameters = {
            'train.model.bidirectional': [True, False],
            'train.model.rnn_type': ['lstm', 'gru'],
            'train.model.rnn_n_layers': [1, 2],
            'train.model.rnn_hidden_size': [10, 50],
            'transformer.transform': ['none'],
            'train.model.optim.lr': [1e-4],
        }
    else:
        hyperparameters = {
            'train.model.optim.lr': [1e-4],
            'batch_size': [16],
            'transformer.transform': ['logmel'],
            'loss_func': ['ce'],
            'epoch_rate': [1.0],
            'sample_balance': ['same'],
        }

    cfg.expt_id = f'{OmegaConf.get_type(cfg.train.model_type)}'
    expt_dir = Path(utils.to_absolute_path('output')) / 'example_face' / f'{cfg.expt_id}'
    expt_dir.mkdir(exist_ok=True, parents=True)
    main(cfg, expt_dir, hyperparameters)

    if not cfg.mlflow:
        shutil.rmtree('mlruns')


if __name__ == '__main__':
    config_store = before_hydra(ExampleEEGConfig)
    hydra_main()
