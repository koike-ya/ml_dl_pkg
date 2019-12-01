import numpy as np
import torch
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score


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


class Metric:
    def __init__(self, name, direction, save_model: bool = False, label_to_detect: int = 1, numpy_: bool = True):
        self.name = name
        self.direction = direction
        self.average_meter = {'train': AverageMeter(direction),
                              'val': AverageMeter(direction),
                              'test': AverageMeter(direction)}
        self.save_model = save_model
        self.label_to_detect = label_to_detect
        # numpy を変更可能に
        self.numpy_ = numpy_

    def add_average_meter(self, phase_name):
        self.average_meter[phase_name] = AverageMeter(self.direction)

    def update(self, phase, loss_value, preds, labels):
        if self.name == 'loss':
            self.average_meter[phase].update(loss_value / len(labels), len(labels))
        elif 'recall' in self.name:
            recall_label = int(self.name[-1])
            self.average_meter[phase].update(recall(labels.copy(), preds.copy(), recall_label, self.numpy_))
        elif self.name == 'far':
            self.average_meter[phase].update(false_detection_rate(preds, labels, self.label_to_detect, self.numpy_))
        elif self.name == 'accuracy':
            self.average_meter[phase].update(accuracy(preds, labels, self.numpy_))
        elif self.name == 'f1':
            self.average_meter[phase].update(f1_score(labels, preds, self.numpy_))
        elif self.name == 'precision':
            self.average_meter[phase].update(precision_score(labels, preds, self.numpy_))
        else:
            raise NotImplementedError


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