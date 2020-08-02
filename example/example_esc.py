import itertools
import logging
import pprint
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path

import hydra
import librosa
import mlflow
import numpy as np
import pandas as pd
import torch
from hydra import utils
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from ml.src.dataset import ManifestWaveDataSet
from ml.tasks.base_experiment import typical_train, typical_experiment
from ml.utils.config import ExptConfig, before_hydra
from ml.utils.utils import dump_dict


@dataclass
class ExampleEscConfig(ExptConfig):
    n_parallel: int = 1
    mlflow: bool = False


def label_func(row):
    return row[2]


def set_load_func(orig_sr, re_sr):
    def load_func(row):
        wave, _ = librosa.load(row[0], sr=orig_sr)
        wave = librosa.resample(wave, orig_sr, re_sr, res_type='kaiser_fast')

        return wave

    return load_func


def create_manifest(cfg, expt_dir):
    data_dir = Path(utils.to_absolute_path('input')) / 'ESC-50-master'

    path_df = pd.read_csv(data_dir / 'meta' / 'esc50.csv')
    path_df['filename'] = str(data_dir / 'audio') + '/' + path_df['filename']
    path_df = path_df[path_df['esc10']]
    labels = sorted(path_df['target'].unique())
    path_df['target'] = path_df['target'].apply(lambda x: labels.index(x))

    train_df = path_df.iloc[:8, :]
    val_df = path_df.iloc[8:, :]
    groups = path_df['fold']

    for phase in ['train', 'val']:
        locals()[f'{phase}_df'].to_csv(expt_dir / f'{phase}_manifest.csv', index=False, header=None)
        cfg.train[f'{phase}_path'] = expt_dir / f'{phase}_manifest.csv'

    return cfg, groups


class LoadDataSet(ManifestWaveDataSet):
    def __init__(self, manifest_path, data_conf, phase='train', load_func=None, transform=None, label_func=None):
        super(LoadDataSet, self).__init__(manifest_path, data_conf, phase, load_func, transform, label_func)

    def __getitem__(self, idx):
        try:
            x = torch.load(self.path_df.iloc[idx, 0].replace('.wav', '.pt'))
        except FileNotFoundError as e:
            print(e)
            return super().__getitem__(idx)
        # print(x.size())
        label = self.labels[idx]

        return x, label


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

    cfg.train.class_names = list(range(10))
    cfg.transformer.sample_rate = 22050

    load_func = set_load_func(44100, cfg.transformer.sample_rate)
    metrics_names = {'train': ['loss', 'uar'],
                     'val': ['loss', 'uar'],
                     'test': ['loss', 'uar']}

    dataset_cls = ManifestWaveDataSet
    cfg, groups = create_manifest(cfg, expt_dir)

    process_func = ['logmel', 'time_mask']

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(metrics_names['val']))),
                               columns=list(hyperparameters.keys()) + metrics_names['val'])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)

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
def hydra_main(cfg: ExampleEscConfig):
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.INFO)
    logging.getLogger("ml").addHandler(console)

    hyperparameters = {
        'train.model.optim.lr': [1e-5],
        'transformer.transform': ['logmel'],
        'train.model.loss_config.loss_func': ['ce'],
        'data.sample_balance': ['same'],
        'transformer.n_mels': [200],
    }

    cfg.expt_id = f'{OmegaConf.get_type(cfg.train.model_type)}'
    expt_dir = Path(utils.to_absolute_path('output')) / 'example_face' / f'{cfg.expt_id}'
    expt_dir.mkdir(exist_ok=True, parents=True)
    main(cfg, expt_dir, hyperparameters)

    if not cfg.mlflow:
        shutil.rmtree('mlruns')


if __name__ == '__main__':
    config_store = before_hydra(ExampleEscConfig)
    hydra_main()
