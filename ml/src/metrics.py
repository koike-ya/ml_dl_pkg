import numpy as np
import torch
from sklearn.metrics import recall_score, accuracy_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, direction=None):
        self.reset()
        self.best_score = 1000000 if direction == 'minimize' else -1
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
    def __init__(self, name, direction, save_model=False, numpy_=True):
        self.name = name
        self.average_meter = {'train': AverageMeter(direction),
                              'val': AverageMeter(direction),
                              'test': AverageMeter(direction)}
        self.save_model = save_model
        # numpy を変更可能に
        self.numpy_ = numpy_

    def update(self, phase, loss_value, preds, labels):
        if self.name == 'loss':
            self.average_meter[phase].update(loss_value / len(labels), len(labels))
        elif self.name == 'recall':
            self.average_meter[phase].update(recall_score(labels, preds))
        elif self.name == 'far':
            self.average_meter[phase].update(false_detection_rate(preds, labels, self.numpy_))
        elif self.name == 'accuracy':
            self.average_meter[phase].update(accuracy(preds, labels, self.numpy_))
        else:
            raise NotImplementedError


def false_detection_rate(pred, true, numpy_=True):
    if numpy_:
        return np.dot(true == 0, pred == 1) / len(pred)

    fp = torch.dot(true.le(0).float(), pred.float()).sum()
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
