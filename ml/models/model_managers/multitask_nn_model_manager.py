import logging
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from ml.models.model_managers.nn_model_manager import NNModelManager
from sklearn.exceptions import NotFittedError

# from apex import amp

logger = logging.getLogger(__name__)


class MultitaskCriterion(torch.nn.BCELoss, torch.nn.MSELoss):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, labels):
        losses = []

        for i in range(labels.size(0)):
            input = inputs[i]
            label = labels[i]   # shape: [task_i, batch, one_hot] -> [batch, one_hot]
            if label.size(1) == 1:
                losses.append(F.mse_loss(input, label))
            else:
                losses.append(F.binary_cross_entropy_with_logits(input, label))

        return torch.stack(losses, dim=0)


class MultitaskNNModelManager(NNModelManager):
    def __init__(self, class_labels, cfg):
        super().__init__(class_labels, cfg)
        self.n_labels_in_each_task = cfg.n_labels_in_each_task
        self.n_tasks = len(self.n_labels_in_each_task)
        self.criterion = MultitaskCriterion()

    def set_criterion(self):
        return MultitaskCriterion()

    def _fit_classify(self, inputs, labels, phase) -> Tuple[List[float], np.ndarray]:
        if self.mixup_alpha:
            raise NotImplementedError

        with torch.set_grad_enabled(phase == 'train'):
            self.model.train() if phase == 'train' else self.model.eval()

            outputs = self.model(inputs)
            outputs = torch.stack(outputs, dim=0)

            y_onehot_list = []
            for i in range(self.n_tasks):
                y_onehot = torch.zeros(labels[i].size(0), self.n_labels_in_each_task[i])
                y_onehot = y_onehot.scatter_(1, labels[i].view(-1, 1).type(torch.LongTensor), 1).to(self.device)
                y_onehot_list.append(y_onehot)

            losses = self.criterion(outputs, torch.stack(y_onehot_list, dim=0))

            if phase == 'train':
                self.optimizer.zero_grad()
                if self.amp:
                    raise NotImplementedError
                    # with amp.scale_loss(loss, self.optimizer) as loss:
                    #     loss.backward()
                else:
                    [losses[i_task].backward(retain_graph=True) for i_task in range(self.n_tasks)]
                self.optimizer.step()

            if self.cfg.return_prob:
                raise NotImplementedError
            else:
                preds = []
                for i in range(self.n_tasks):
                    preds.append(torch.max(outputs[i], 1)[1])
                preds = torch.stack(preds, dim=0)

        losses = [losses[i].item() for i in range(self.n_tasks)]
        return losses, preds.cpu().numpy()

    def _fit_regress(self, inputs, labels, phase) -> Tuple[float, np.ndarray]:
        raise NotImplementedError

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
                    raise NotImplementedError

        return pred_list
