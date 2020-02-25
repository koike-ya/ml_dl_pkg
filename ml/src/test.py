from __future__ import print_function, division

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from ml.models.nn_models.rnn import *
from ml.utils import init_seed, init_device, set_dataloader, set_model
from ml.utils import test_args


def inference(args, model, eeg_conf, numpy, device):
    if args.model_name in ['kneighbor', 'knn']:
        args.model_name = 'kneighbor'

    # class_names is None when don't need labels
    dataloader = set_dataloader(args, eeg_conf, label_func=None, class_names=None, phase='inference', device=device)

    pred_list = []
    path_list = []

    for i, (inputs, paths) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device)

        if numpy:
            preds = model.predict(inputs)
        else:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        pred_list.extend(preds)
        # Transpose paths, but I don't know why dataloader outputs aukward
        path_list.extend([list(pd.DataFrame(paths).iloc[:, i].values) for i in range(len(paths[0]))])

    return pred_list, path_list


def test(args, model, eeg_conf, label_func, class_names, numpy, device):

    dataloader = set_dataloader(args, eeg_conf, class_names, label_func=label_func, phase='test', device=device)
    pred_list = torch.empty((len(dataloader) * args.batch_size, 1), dtype=torch.long, device=device)
    label_list = torch.empty((len(dataloader) * args.batch_size, 1), dtype=torch.long, device=device)
    
    for i, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.to(device)

        if numpy:
            preds = model.predict(inputs)
        else:
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

        pred_list[i * args.batch_size:i * args.batch_size + preds.size(0), 0] = preds
        label_list[i * args.batch_size:i * args.batch_size + labels.size(0), 0] = labels

    print(confusion_matrix(label_list.cpu().numpy(), pred_list.cpu().numpy(),
                               labels=list(range(len(class_names)))))
    print('accuracy:', accuracy_score(label_list.cpu().numpy(), pred_list.cpu().numpy()))


def main(args, class_names):
    init_seed(args)
    device = init_device(args)
    if args.model_name in ['kneighbor', 'knn']:
        args.model_name = 'kneighbor'
    numpy = 'nn' not in args.model_name

    eeg_conf = set_eeg_conf(args)

    model = set_model(args, class_names, eeg_conf, device)
    if numpy:
        model.load_model_(args.model_path)
    else:
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
    args.weight = list(map(float, args.loss_weight.split('-')))

    def label_func(path):
        return path[-8:-4]

    if args.test:
        test(args, model, eeg_conf, label_func, class_names, numpy, device)
    if args.inference:
        return inference(args, model, eeg_conf, numpy, device)


if __name__ == '__main__':
    args = test_args().parse_args()
    class_names = ['null', 'bckg', 'seiz']
    init_seed(args)
    device = init_device(args)

    main(args, class_names)
