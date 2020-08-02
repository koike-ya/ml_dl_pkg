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
from hydra import utils
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from ml.models.nn_models.cnn import CNNConfig
from ml.models.nn_models.cnn_rnn import CNNRNNConfig
from ml.models.nn_models.rnn import RNNConfig
from ml.src.dataset import ManifestDataSet
from ml.tasks.base_experiment import typical_train, typical_experiment
from ml.utils.config import ExptConfig, before_hydra
from ml.utils.utils import dump_dict

LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


@dataclass
class ExampleFaceConfig(ExptConfig):
    n_parallel: int = 1
    mlflow: bool = False


def label_func(row):
    return row[0]


def load_func(row):
    im = np.array(list(map(int, row[1].split(' ')))).reshape((48, 48)) / 255
    return im[None, :, :]


def create_manifest(expt_conf, expt_dir):
    data_dir = Path(utils.to_absolute_path('input'))
    manifest_df = pd.read_csv(data_dir / 'fer2013.csv')

    train_val_df = manifest_df[manifest_df['Usage'] == 'Training']
    train_df = train_val_df.iloc[:int(len(train_val_df) * 0.7), :]
    train_df.to_csv(expt_dir / 'train_manifest.csv', index=False, header=None)
    expt_conf.train.train_path = expt_dir / 'train_manifest.csv'

    val_df = train_val_df.iloc[int(len(train_val_df) * 0.7):, :]
    val_df.to_csv(expt_dir / 'val_manifest.csv', index=False, header=None)
    expt_conf.train.val_path = expt_dir / 'val_manifest.csv'

    test_df = manifest_df[manifest_df['Usage'] != 'Training']
    test_df.to_csv(expt_dir / 'test_manifest.csv', index=False, header=None)
    expt_conf.train.test_path = expt_dir / 'test_manifest.csv'

    return expt_conf


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

    cfg.train.class_names = LABELS
    dataset_cls = ManifestDataSet
    metrics_names = {'train': ['loss', 'uar'],
                     'val': ['loss', 'uar'],
                     'test': ['loss', 'uar']}

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
            result_series, val_pred, _ = typical_train(cfg, load_func, label_func, process_func, dataset_cls, groups)

            mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})

        return result_series, val_pred

    # For debugging
    if cfg.n_parallel == 1:
        result_pred_list = [experiment(pattern, deepcopy(cfg)) for pattern in patterns]
    else:
        cfg.n_jobs = 0
        result_pred_list = Parallel(n_jobs=cfg.n_parallel, verbose=0)(
            [delayed(experiment)(pattern, deepcopy(cfg)) for pattern in patterns])

    val_results.iloc[:, :len(hyperparameters)] = patterns
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
def hydra_main(cfg: ExampleFaceConfig):
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.INFO)
    logging.getLogger("ml").addHandler(console)

    if OmegaConf.get_type(cfg.train.model) == CNNConfig:
        hyperparameters = {
            'train.model.optim.lr': [1e-4],
        }
    elif OmegaConf.get_type(cfg.train.model) == CNNRNNConfig:
        hyperparameters = {
            'train.model.optim.lr': [1e-3, 1e-4, 1e-5],
            'window_size': [0.5],
            'window_stride': [0.1],
            'transform': ['logmel'],
            'rnn_type': [cfg.rnn_type],
            'bidirectional': [True],
            'rnn_n_layers': [1],
            'rnn_hidden_size': [10],
        }
    elif OmegaConf.get_type(cfg.train.model) == RNNConfig:
        hyperparameters = {
            'bidirectional': [True, False],
            'rnn_type': ['lstm', 'gru'],
            'rnn_n_layers': [1, 2],
            'rnn_hidden_size': [10, 50],
            'transform': [None],
            'train.model.optim.lr': [1e-3, 1e-4, 1e-5],
        }
    else:
        hyperparameters = {
            'train.model.optim.lr': [1e-4, 1e-5],
            'data.batch_size': [64],
            'data.epoch_rate': [1.0],
            'data.sample_balance': ['same'],
        }

    cfg.expt_id = f'{OmegaConf.get_type(cfg.train.model_type)}_{cfg.train.model.pretrained}'
    expt_dir = Path(utils.to_absolute_path('output')) / 'example_face' / f'{cfg.expt_id}'
    expt_dir.mkdir(exist_ok=True, parents=True)
    main(cfg, expt_dir, hyperparameters)

    if not cfg.mlflow:
        shutil.rmtree('mlruns')


if __name__ == '__main__':
    config_store = before_hydra(ExampleFaceConfig)
    hydra_main()
