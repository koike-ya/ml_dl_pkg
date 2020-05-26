import argparse
import itertools
import logging
import pprint
import shutil
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path

import librosa
import mlflow
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from ml.src.dataset import ManifestWaveDataSet
from ml.tasks.base_experiment import typical_train, base_expt_args, typical_experiment
from ml.utils.utils import dump_dict


def expt_args(parser):
    parser = base_expt_args(parser)
    expt_parser = parser.add_argument_group("Experiment arguments")
    expt_parser.add_argument('--n-parallel', default=1, type=int)
    expt_parser.add_argument('--mlflow', action='store_true')

    return parser


def label_func(row):
    return row[2]


def set_load_func(orig_sr, re_sr):
    def load_func(row):
        wave, _ = librosa.load(row[0], sr=orig_sr)
        wave = librosa.resample(wave, orig_sr, re_sr, res_type='kaiser_fast')

        return wave

        # if wave.shape[0] > const_length:
        #     wave = wave[:const_length]
        # elif wave.shape[0] < const_length:
        #     n_pad = (const_length - wave.shape[0]) // 2 + 1
        #     wave = np.pad(wave[:const_length], n_pad)[:const_length]
        # return wave.reshape((1, -1))

    return load_func


def create_manifest(expt_conf, expt_dir):
    data_dir = Path(__file__).resolve().parent / 'input' / 'ESC-50-master'

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
        expt_conf[f'{phase}_path'] = expt_dir / f'{phase}_manifest.csv'

    return expt_conf, groups


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


def main(expt_conf, expt_dir, hyperparameters):
    if expt_conf['expt_id'] == 'timestamp':
        expt_conf['expt_id'] = dt.today().strftime('%Y-%m-%d_%H:%M')

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')

    expt_conf['class_names'] = list(range(10))
    expt_conf['sample_rate'] = 11025

    load_func = set_load_func(44100, expt_conf['sample_rate'])
    metrics_names = {'train': ['loss', 'uar'],
                     'val': ['loss', 'uar'],
                     'test': ['loss', 'uar']}

    dataset_cls = ManifestWaveDataSet
    expt_conf, groups = create_manifest(expt_conf, expt_dir)

    def parallel_preprocess(dataset, idx):
        processed, _ = dataset[idx]
        path = dataset.path_df.iloc[idx, 0]
        torch.save(processed, path.replace('.wav', '.pt'))

    for phase in tqdm(['train', 'val']):
        # process_func = Preprocessor(expt_conf, phase).preprocess
        # dataset = ManifestWaveDataSet(expt_conf[f'{phase}_path'], expt_conf, phase, load_func, process_func, label_func)
        # Parallel(n_jobs=8, verbose=0)(
        #     [delayed(parallel_preprocess)(dataset, idx) for idx in range(len(dataset))])
        print(f'{phase} done')

    process_func = None

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(metrics_names['val']))),
                               columns=list(hyperparameters.keys()) + metrics_names['val'])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)

    def experiment(pattern, expt_conf):
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = pattern[i]

        expt_conf['model_path'] = str(expt_dir / f"{'_'.join([str(p).replace('/', '-') for p in pattern])}.pth")
        expt_conf['log_id'] = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
        # TODO cv時はmean と stdをtrainとvalの分割後に求める必要がある
        with mlflow.start_run():
            result_series, val_pred, _ = typical_train(expt_conf, load_func, label_func, process_func, dataset_cls,
                                                       groups)

            mlflow.log_params({hyperparameter: value for hyperparameter, value in zip(hyperparameters.keys(), pattern)})
            # mlflow.log_artifacts(expt_dir)

        return result_series, val_pred

    # For debugging
    if expt_conf['n_parallel'] == 1:
        result_pred_list = [experiment(pattern, deepcopy(expt_conf)) for pattern in patterns]
    else:
        expt_conf['n_jobs'] = 0
        result_pred_list = Parallel(n_jobs=expt_conf['n_parallel'], verbose=0)(
            [delayed(experiment)(pattern, deepcopy(expt_conf)) for pattern in patterns])

    val_results.iloc[:, :len(hyperparameters)] = patterns
    result_list = np.array([result for result, pred in result_pred_list])
    val_results.iloc[:, len(hyperparameters):] = result_list
    pp.pprint(val_results)
    pp.pprint(val_results.iloc[:, len(hyperparameters):].describe())

    val_results.to_csv(expt_dir / 'val_results.csv', index=False)
    print(f"Devel results saved into {expt_dir / 'val_results.csv'}")
    for (_, _), pattern in zip(result_pred_list, patterns):
        pattern_name = f"{'_'.join([str(p).replace('/', '-') for p in pattern])}"
        dump_dict(expt_dir / f'{pattern_name}.txt', expt_conf)

    # Train with train + devel dataset
    if expt_conf['test']:
        best_trial_idx = val_results['uar'].argmax()

        best_pattern = patterns[best_trial_idx]
        for i, param in enumerate(hyperparameters.keys()):
            expt_conf[param] = best_pattern[i]
        dump_dict(expt_dir / 'best_parameters.txt', {p: v for p, v in zip(hyperparameters.keys(), best_pattern)})

        metrics, pred_dict_list, _ = typical_experiment(expt_conf, load_func, label_func, process_func, dataset_cls,
                                                        groups)

        sub_name = f"uar-{metrics[-1]:.4f}_sub_{'_'.join([str(p).replace('/', '-') for p in best_pattern])}.csv"
        pd.DataFrame(pred_dict_list['test']).to_csv(expt_dir / f'{sub_name}_prob.csv', index=False, header=None)
        pd.DataFrame(pred_dict_list['test'].argmax(axis=1)).to_csv(expt_dir / sub_name, index=False, header=None)
        print(f"Submission file is saved in {expt_dir / sub_name}")

    mlflow.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    expt_conf = vars(expt_args(parser).parse_args())

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.INFO)
    logging.getLogger("ml").addHandler(console)

    if expt_conf['model_type'] == 'cnn':
        hyperparameters = {
            'model_type': ['cnn'],
            'window_size': [0.6],
            'window_stride': [0.5],
            'transform': ['logmel'],
            'cnn_channel_list': [[4, 8, 16, 32]],
            'cnn_kernel_sizes': [[[4, 4]] * 4],
            'cnn_stride_sizes': [[[2, 2]] * 4],
            'cnn_padding_sizes': [[[1, 1]] * 4],
            'lr': [1e-4],
        }
    elif expt_conf['model_type'] == 'cnn_rnn':
        hyperparameters = {
            'lr': [1e-4],
            'window_size': [0.2],
            'window_stride': [0.05],
            'transform': ['logmel'],
            'rnn_type': [expt_conf['rnn_type']],
            'bidirectional': [True],
            'rnn_n_layers': [1],
            'rnn_hidden_size': [10],
        }
    elif expt_conf['model_type'] == 'rnn':
        hyperparameters = {
            'bidirectional': [True, False],
            'rnn_type': ['lstm', 'gru'],
            'rnn_n_layers': [1, 2],
            'rnn_hidden_size': [10, 50],
            'transform': [None],
            'lr': [1e-4],
        }
    elif expt_conf['model_type'] == '1dcnn':
        hyperparameters = {
            'cnn_channel_list': [[32, 64, 'M']],
            'cnn_kernel_sizes': [[[1, 64], [1, 16], [1, 64]]],
            'cnn_stride_sizes': [[[1, 2], [1, 2], [1, 64]]],
            'transform': [None],
            'lr': [1e-4],
        }
    else:
        hyperparameters = {
            'lr': [1e-5],
            'transform': ['logmel'],
            'loss_func': ['ce'],
            'epoch_rate': [1.0],
            'sample_balance': ['same'],
            'window_size': [0.3],
            'window_stride': [0.04],
            'n_mels': [200],
        }

    hyperparameters['model_type'] = [expt_conf['model_type']]

    expt_conf['expt_id'] = f"{expt_conf['model_type']}_{expt_conf['transform']}"
    # expt_conf['window_size'] = hyperparameters['window_size'][0]
    # expt_conf['window_stride'] = hyperparameters['window_stride'][0]
    expt_dir = Path(__file__).resolve().parent / 'output' / 'example_esc' / f"{expt_conf['expt_id']}"
    expt_dir.mkdir(exist_ok=True, parents=True)
    main(expt_conf, expt_dir, hyperparameters)

    if not expt_conf['mlflow']:
        shutil.rmtree('mlruns')
