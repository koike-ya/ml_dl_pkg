import torch
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, initial_value=0, inequality=None):
        self.reset()
        self.best_score = initial_value
        self.inequality = inequality

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
        if self.inequality == 'more':
            if self.average > self.best_score:
                self.best_score = self.average
                return True
        else:
            if self.average < self.best_score:
                self.best_score = self.average
                return True
        return False


class Metric:
    def __init__(self, name, initial_value, inequality, save_model=False):
        self.name = name
        self.average_meter = {'train': AverageMeter(initial_value, inequality),
                              'val': AverageMeter(initial_value, inequality)}
        self.save_model = save_model

    def update(self, phase, loss_value, n_example, preds, labels, classes, numpy):
        if self.name == 'loss':
            self.average_meter[phase].update(loss_value / n_example, n_example)
        elif self.name == 'recall':
            self.average_meter[phase].update(recall_rate(preds, labels, numpy))
        elif self.name == 'far':
            self.average_meter[phase].update(false_detection_rate(preds, labels, classes, numpy))
        elif self.name == 'accuracy':
            self.average_meter[phase].update(accuracy(preds, labels, classes, numpy))
        elif self.name == 'confusion_matrix':
            self.average_meter[phase].update(confusion_matrix_(preds, labels, classes, numpy))
        else:
            raise NotImplementedError


def recall_rate(pred, true, numpy=False):
    # When labels are multi class recall is invalid.
    if numpy:
        return recall_score(true, pred)

    true, pred = true.float(), pred.float()
    # le(0.0) makes bits reverse
    tp = torch.dot(true, pred).sum()
    fn = torch.dot(true, pred.le(0).float()).sum()
    if torch.add(tp, fn) == 0:
        return torch.zeros(1)
    return tp.div(torch.add(tp, fn)).item()


def false_detection_rate(pred, true, classes, numpy=False):
    if numpy:
        return np.dot(true == 0, pred == 1) / len(pred)

    fp = torch.dot(true.le(0).float(), pred.float()).sum()
    return fp.div(len(pred)).item()


def accuracy(pred, true, classes, numpy=False):
    if numpy:
        return accuracy_score(true, pred)

    true, pred = true.float(), pred.float()
    # le(0.0) makes bits reverse
    tp = torch.dot(true, pred).sum()
    tn = torch.dot(true.le(0).float(), pred.le(0).float()).sum()
    return (tp + tn).div(len(pred)).item()


def confusion_matrix_(pred, true, classes, numpy):
    if numpy:
        return confusion_matrix(true, pred)

    matrix = torch.zeros(len(classes), len(classes))
    for class_ in classes:
        vec = torch.eq(true, class_).nonzero()
        for scalar in vec:
            matrix[class_, pred[scalar]] += 1

    tp = torch.diag(matrix).sum()
    return tp / len(pred)


# def specificity(pred, true, numpy=False)
#     true, pred = true.float(), pred.float()
#     # fdr = 1 - specifity
#     fp = torch.dot(true.le(0).float(), pred).sum()
#     tn = torch.dot(true.le(0).float(), pred.le(0).float()).sum()
#     if torch.add(fp, tn) == 0:
#         return torch.zeros(1)
#     return fp.div(torch.add(fp, tn))
