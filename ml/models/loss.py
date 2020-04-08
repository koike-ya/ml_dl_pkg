
import torch
from torch import Tensor
from typing import List
from torch.distributions.categorical import Categorical
from torch.functional import F


def loss_args(parser):
    loss_parser = parser.add_argument_group("ML/DL loss arguments")
    loss_parser.add_argument('--loss-func', help='Loss function', choices=['mse', 'ce', 'kl_div'])
    loss_parser.add_argument('--kl-penalty', help='Weight of KL regularization term', type=float, default=0.0)
    loss_parser.add_argument('--entropy-penalty', help='Weight of entropy regularization term', type=float, default=0.0)
    return parser


def set_criterion(cfg):
    if isinstance(cfg['loss_weight'], str):
        cfg['loss_weight'] = [1.0] * len(cfg['class_names'])
    if cfg['task_type'] == 'regress' or cfg['loss_func'] == 'mse':
        criterion = torch.nn.MSELoss()
    elif cfg['loss_func'] == 'ce':
        criterion = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(cfg['loss_weight']))
    elif cfg['loss_func'] == 'kl_div':
        criterion = KLLoss()

    penalties = []
    if cfg['kl_penalty']:
        penalties.append({'weight': cfg['kl_penalty'], 'func': KLLoss(batch_wise=True)})
    if cfg['entropy_penalty']:
        penalties.append({'weight': cfg['entropy_penalty'], 'func': EntropyLoss()})

    return LossManager(criterion, penalties)


class LossManager(torch.nn.Module):
    def __init__(self, criterion, penalties: List):
        super(LossManager, self).__init__()
        self.criterion = criterion
        self.penalties = penalties

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = self.criterion(input, target)
        for penalty in self.penalties:
            loss += penalty['weight'] * penalty['func'](input, target)

        return loss


class KLLoss(torch.nn.KLDivLoss):
    def __init__(self, batch_wise=False):
        super(KLLoss, self).__init__(reduction='batchmean')
        self.batch_wise = batch_wise

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.batch_wise:
            n_labels = target.size()[1]
            target = target.sum(dim=0)
            input = input.argmax(dim=1)
            input = torch.Tensor([input.eq(label).sum() for label in range(n_labels)]).to(input.device)

        input = torch.nn.LogSigmoid()(input)

        return super().forward(input, target)


class EntropyLoss(torch.nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return Categorical(target).entropy().mean()


