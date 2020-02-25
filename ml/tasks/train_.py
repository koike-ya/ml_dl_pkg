from __future__ import print_function, division

import pandas as pd
import argparse
from ml.src import Metric
from ml.src.dataloader import set_dataloader
from ml.src.dataset import ManifestDataSet
from ml.models.train_managers.train_manager import train_manager_args, BaseTrainManager


def train_args(parser):
    train_parser = parser.add_argument_group('train arguments')
    train_parser.add_argument('--only-model-test', action='store_true', help='Load learned model and not training')
    train_parser.add_argument('--test', action='store_true', help='Do testing')
    parser = train_manager_args(parser)

    return parser


def voting(args, pred_list, path_list):
    def ensemble_preds(pred_list, path_list, sub_df, thresh):
        # もともとのmatファイルごとに振り分け直す
        patient_name = path_list[0][0].split('/')[-3]
        orig_mat_list = sub_df[sub_df['clip'].apply(lambda x: '_'.join(x.split('_')[:2])) == patient_name]
        ensembled_pred_list = []
        for orig_mat_name in orig_mat_list['clip']:
            seg_number = int(orig_mat_name[-8:-4])
            one_segment_preds = [pred for path, pred in zip(path_list[0], pred_list) if
                                 int(path.split('/')[-2].split('_')[-1]) == seg_number]
            ensembled_pred = int(sum(one_segment_preds) >= len(one_segment_preds) * thresh)
            ensembled_pred_list.append(ensembled_pred)
        orig_mat_list['preictal'] = ensembled_pred_list
        return orig_mat_list

    # preds to csv
    sub_df = pd.read_csv(args.sub_path, engine='python')
    thresh = args.thresh  # 1の割合がthreshを超えたら1と判断
    pred_df = ensemble_preds(pred_list, path_list, sub_df, thresh)
    sub_df.loc[pred_df.index, 'preictal'] = pred_df['preictal']
    sub_df.to_csv(args.sub_path, index=False)


def label_func(path):
    return path.split('/')[-2].split('_')[2]


def train(model_conf):

    if model_conf['task_type'] == 'classify':
        metrics = [
            Metric('loss', direction='minimize'),
            Metric('accuracy', direction='maximize', save_model=True),
            Metric('far', direction='minimize')
        ]
    else:
        class_names = ['0']
        metrics = [Metric('loss', direction='minimize', save_model=True)]

    # dataset, dataloaderの作成
    dataloaders = {}
    load_func = lambda x: pd.read_csv(x, index_col=0)
    for phase in phases:
        dataset = ManifestDataSet(model_conf[f'{phase}_path'], model_conf, load_func=load_func, label_func=label_func)
        dataloaders[phase] = set_dataloader(dataset, phase, model_conf)

    # modelManagerをインスタンス化、trainの実行
    train_manager = BaseTrainManager(class_names, model_conf, dataloaders, metrics)

    # モデルの学習を行う場合
    if not model_conf['only_test']:
        train_manager.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train arguments')
    model_conf = vars(train_args(parser).parse_args())
    assert model_conf['train_path'] != '' or model_conf['val_path'] != '', \
        'You need to select training, validation data file to training, validation in --train-path, --val-path argments'
    train(model_conf)
