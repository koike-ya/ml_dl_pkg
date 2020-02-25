import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from ml.models.nn_models.multitask_panns_model import construct_multitask_panns
from ml.models.model_managers.nn_model_manager import NNModelManager
from ml.models.nn_models.nn_utils import get_param_size
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)
# from apex import amp


class MultitaskCriterion(torch.nn.BCELoss, torch.nn.MSELoss):
    def __init__(self, classify_flag):
        super().__init__()
        self.classify_flag = classify_flag

    def forward(self, input, target):
        if self.classify_flag:
            return F.binary_cross_entropy(input, target, weight=self.weight)
        else:
            return F.mse_loss(input, target)


class MultitaskNNModelManager(NNModelManager):
    def __init__(self, class_labels, cfg):
        super().__init__(class_labels, cfg)
        self.n_tasks = cfg['n_tasks']

    def _set_criterion(self):
        return MultitaskCriterion(len(self.class_labels) > 1)

    def _init_model(self, transfer=False):
        if self.cfg['model_type'] == 'multitask_panns':
            model = construct_multitask_panns(self.cfg)
        else:
            raise NotImplementedError('model_type should be either rnn or cnn, nn would be implemented in the future.')

        logger.info(f'Model Parameters: {get_param_size(model)}')

        return model

    def _fit_classify(self, inputs, labels, phase) -> Tuple[float, np.ndarray]:
        with torch.set_grad_enabled(phase == 'train'):
            outputs = self.model(inputs)

            loss = 0
            for i in range(self.n_tasks):
                y_onehot = torch.zeros(labels[i].size(0), len(self.class_labels))
                y_onehot = y_onehot.scatter_(1, labels[i].view(-1, 1).type(torch.LongTensor), 1)

                loss += self.criterion(outputs[i], y_onehot.to(self.device))

            if phase == 'train':
                self.optimizer.zero_grad()
                if self.amp:
                    with amp.scale_loss(loss, self.optimizer) as loss:
                        loss.backward()
                else:
                    loss.backward(retain_graph=True)
                self.optimizer.step()

            preds = np.zeros((outputs[0].size(0), self.n_tasks))
            for i in range(self.n_tasks):
                preds[:, i] = torch.max(outputs[i], 1)[1].cpu().numpy()

        return loss.item(), preds

    def _fit_regress(self, inputs, labels, phase) -> Tuple[float, np.ndarray]:
        with torch.set_grad_enabled(phase == 'train'):
            preds = self.model(inputs)

            loss = 0
            for i in range(self.n_tasks):
                loss += self.criterion(preds[i], labels[i].to(self.device).float())

            if phase == 'train':
                self.optimizer.zero_grad()
                if self.amp:
                    with amp.scale_loss(loss, self.optimizer) as loss:
                        loss.backward()
                else:
                    loss.backward(retain_graph=True)
                self.optimizer.step()

            pred_list = np.zeros((preds[0].size(0), self.n_tasks))
            for i in range(self.n_tasks):
                pred_list[:, i] = preds[i].detach().cpu().numpy().reshape(-1,)

        return loss.item(), pred_list

    def predict(self, inputs) -> np.array:  # NNModelManagerは自身がfittedを管理している
        if not self.fitted:
            raise NotFittedError(f'This NNModelManager instance is not fitted yet.')

        with torch.set_grad_enabled(False):
            self.model.eval()
            preds = self.model(inputs)

            pred_list = np.zeros((preds[0].size(0), self.n_tasks))
            for i in range(self.n_tasks):
                if self.cfg['task_type'] == 'classify':
                    pred_list[:, i] = torch.max(preds[i], 1)[1].cpu().numpy()
                else:
                    pred_list[:, i] = preds[i].detach().cpu().numpy().reshape(-1,)

        return pred_list