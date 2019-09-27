from __future__ import print_function, division

from ml.models import *
from ml.src.dataset import CSVDataSet, ManifestDataSet


def init_seed(args):
    # Set seeds for determinism
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def init_device(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)
    return device


def set_dataset(args, data_conf, phase):
    dataset_cls = CSVDataSet if args.input_type == 'csv' else ManifestDataSet
    data_path = getattr(args, phase + '_path')
    if phase in ['test', 'inference']:
        dataset = dataset_cls(data_path, data_conf)
    else:
        dataset = dataset_cls(data_path, data_conf)

    return dataset


def set_dataloader(args, dataset, class_names, phase):
    if phase in ['test', 'inference']:
        return_path = True if phase == 'inference' else False
        dataloader = WrapperDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                          pin_memory=True, shuffle=False)
    else:
        # weights = make_weights_for_balanced_classes(dataset.get_labels(), len(class_names))
        # sampler = WeightedRandomSampler(weights, int(len(dataset) * args.epoch_rate))
        dataloader = WrapperDataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                          pin_memory=True, sampler=None, drop_last=True, shuffle=False)
    return dataloader


def get_feature_size(args):
    dataset = CSVDataSet if args.input_type == 'csv' else ManifestDataSet


def set_model(args, class_names, input_shape, device):
    if args.model_name == '2dcnn_1':
        model = cnn_1_16_399(n_labels=len(class_names), input_shape=input_shape)
    elif args.model_name == '2dcnn_2':
        model = cnn_16_751_751(n_labels=len(class_names))
    elif args.model_name == 'rnn':
        cnn, out_ftrs = cnn_ftrs_16_751_751(eeg_conf)
        model = RNN(cnn, out_ftrs, args.batch_size, args.rnn_type, class_names, eeg_conf=eeg_conf,
                    rnn_hidden_size=args.rnn_hidden_size, nb_layers=args.rnn_n_layers)
    elif args.model_name == '3dcnn':
        model = cnn_1_16_751_751(n_labels=len(class_names))
    elif args.model_name == 'xgboost':
        model = XGBoost(list(range(len(class_names))))
    elif args.model_name == 'sgdc':
        model = SGDC(list(range(len(class_names))))
    elif args.model_name in ['kneighbor', 'knn']:
        args.model_name = 'kneighbor'
        model = KNN(list(range(len(class_names))))
    else:
        raise NotImplementedError

    if 'nn' in args.model_name:
        model = model.to(device)
        if hasattr(args, 'silent') and (not args.silent):
            print(model)

    return model
