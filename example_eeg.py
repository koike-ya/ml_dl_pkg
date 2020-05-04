import argparse
import itertools
import logging
import pprint
import shutil
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ml.src.dataset import ManifestWaveDataSet
from ml.tasks.base_experiment import typical_train, base_expt_args, typical_experiment
from ml.utils.utils import dump_dict

BINARY_LABELS = {'Z': 0, 'O': 0, 'N': 0, 'F': 0, 'S': 1}


def expt_args(parser):
    parser = base_expt_args(parser)
    expt_parser = parser.add_argument_group("Experiment arguments")
    expt_parser.add_argument('--n-parallel', default=1, type=int)
    expt_parser.add_argument('--mlflow', action='store_true')

    return parser


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
        return wave.reshape((1, -1))

    return load_func


def create_manifest(expt_conf, expt_dir):
    data_dir = Path(__file__).resolve().parent / 'input' / 'eeg'
    path_list = [str(p.resolve()) for p in sorted(list(data_dir.iterdir())) if p.is_file()]

    path_df = pd.DataFrame([path_list])
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
        expt_conf[f'{phase}_path'] = expt_dir / f'{phase}_manifest.csv'

    return expt_conf


def main(expt_conf, expt_dir, hyperparameters):
    if expt_conf['expt_id'] == 'timestamp':
        expt_conf['expt_id'] = dt.today().strftime('%Y-%m-%d_%H:%M')

    logging.basicConfig(level=logging.DEBUG, format="[%(name)s] [%(levelname)s] %(message)s",
                        filename=expt_dir / 'expt.log')

    expt_conf['class_names'] = [0, 1]
    expt_conf['sample_rate'] = 173.61

    one_audio_sec = 10
    dataset_cls = ManifestWaveDataSet
    load_func = set_load_func(expt_conf['sample_rate'], one_audio_sec)
    metrics_names = {'train': ['loss', 'uar'],
                     'val': ['loss', 'uar'],
                     'test': ['loss', 'uar']}

    dataset_cls = ManifestWaveDataSet
    expt_conf = create_manifest(expt_conf, expt_dir)
    process_func = None

    patterns = list(itertools.product(*hyperparameters.values()))
    val_results = pd.DataFrame(np.zeros((len(patterns), len(hyperparameters) + len(metrics_names['val']))),
                               columns=list(hyperparameters.keys()) + metrics_names['val'])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(hyperparameters)
    groups = None

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
            'window_size': [2.0],
            'window_stride': [0.2],
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
            'window_size': [0.5],
            'window_stride': [0.1],
            'transform': ['logmel'],
            'rnn_type': [expt_conf['rnn_type']],
            'bidirectional': [True],
            'rnn_n_layers': [1, 2],
            'rnn_hidden_size': [10, 50],
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
    else:
        hyperparameters = {
            'lr': [1e-4],
            'batch_size': [16],
            'transform': ['logmel'],
            'loss_func': ['ce'],
            'epoch_rate': [1.0],
            'sample_balance': ['same'],
        }

    hyperparameters['model_type'] = [expt_conf['model_type']]

    expt_conf['expt_id'] = f"{expt_conf['model_type']}_{expt_conf['transform']}"
    expt_dir = Path(__file__).resolve().parent / 'output' / 'example_eeg' / f"{expt_conf['expt_id']}"
    expt_dir.mkdir(exist_ok=True, parents=True)
    main(expt_conf, expt_dir, hyperparameters)

    if not expt_conf['mlflow']:
        shutil.rmtree('mlruns')
