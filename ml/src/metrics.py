import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score, balanced_accuracy_score, confusion_matrix


ALLOWED_METRICS = ['loss', 'far', 'accuracy', 'f1', 'precision', 'uar', 'specificity']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, direction=None):
        self.reset()
        self.best_score = 1000000000 if direction == 'minimize' else -1
        self.direction = direction
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count

    def update_best(self):
        if self.direction == 'maximize':
            if self.average >= self.best_score:
                self.best_score = self.average
                return True
        elif self.direction == 'minimize':
            if self.average <= self.best_score:
                self.best_score = self.average
                return True
        else:
            raise NotImplementedError('direction needs to be either maximize or minimize when update the best score.')
        return False


def metrics2df(metrics, phase='test'):
    df = pd.DataFrame()

    if isinstance(metrics, dict):
        for metric_name, meter in metrics.items():
            df = pd.concat([df, pd.DataFrame([metric_name, meter.mean(), meter.std()]).T])
        df.columns = ['metric_name', 'mean', 'std']

    elif isinstance(metrics, list):
        for metric in metrics:
            df = pd.concat([df, pd.DataFrame(
                [metric.name, metric.average_meter[phase].best_score]).T])
        df.columns = ['metric_name', 'value']

    return df


class Metric:
    def __init__(self, name, save_model: bool = False, label_to_detect: int = 1, numpy_: bool = True):
        self.name = name
        self.direction = 'minimize' if name == 'loss' else 'maximize'
        self.average_meter = AverageMeter(self.direction)
        self.save_model = save_model
        self.label_to_detect = label_to_detect
        # numpy を変更可能に
        self.numpy_ = numpy_

    def add_average_meter(self, phase_name):
        self.average_meter[phase_name] = AverageMeter(self.direction)

    def update(self, loss_value, preds, labels):
        if len(preds.shape) > 1:
            preds = np.argmax(preds, axis=1)

        if self.name == 'loss':
            self.average_meter.update(loss_value / len(labels), len(labels))
        elif 'recall' in self.name:
            recall_label = int(self.name[-1])
            self.average_meter.update(recall(labels.copy(), preds.copy(), recall_label, self.numpy_))
        elif self.name == 'far':
            self.average_meter.update(false_detection_rate(preds, labels, self.label_to_detect, self.numpy_))
        elif self.name == 'accuracy':
            self.average_meter.update(accuracy(preds, labels, self.numpy_))
        elif self.name == 'f1':
            self.average_meter.update(f1_score(labels, preds))
        elif self.name == 'precision':
            self.average_meter.update(precision_score(labels, preds, self.numpy_))
        elif self.name == 'uar':
            self.average_meter.update(balanced_accuracy_score(labels, preds))
        elif self.name == 'specificity':
            self.average_meter.update(specificity(labels, preds))
        else:
            raise NotImplementedError


def get_metric_list(metric_names, target_metric=None):
    for name in metric_names:
        assert name in ALLOWED_METRICS, f'You need to select metrics from {ALLOWED_METRICS}'

    metric_list = []
    for one_metric in metric_names:
        metric_list.append(Metric(one_metric, save_model=one_metric == target_metric))

    return metric_list


def false_detection_rate(pred, true, label_to_detect: int = 1, numpy_: bool = True):
    if numpy_:
        return np.dot(true != label_to_detect, pred == label_to_detect) / len(pred)

    fp = torch.dot(true.le(0).float(), pred.ge(1).float()).sum()
    return fp.div(len(pred)).item()


def accuracy(pred, true, numpy_=False):
    return accuracy_score(true, pred)
    if numpy_:
        return accuracy_score(true, pred)

    true, pred = true.float(), pred.float()
    # le(0.0) makes bits reverse
    tp = torch.dot(true, pred).sum()
    tn = torch.dot(true.le(0).float(), pred.le(0).float()).sum()
    return (tp + tn).div(len(pred)).item()


def recall(labels, preds, recall_label: int = 1, numpy_: bool = False):
    # labelと同じラベルを1にして、それ以外を0にする
    assert recall_label != 0    # recall labelが0のときは計算がおかしくなる

    labels[labels != recall_label] = 0
    labels[labels == recall_label] = 1
    preds[preds != recall_label] = 0
    preds[preds == recall_label] = 1
    if numpy_:
        return recall_score(labels, preds)

    return recall_score(labels, preds)


def specificity(pred, true):
    cm1 = confusion_matrix(true, pred)
    return cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])