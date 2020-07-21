import torch
from torch import Tensor
from torch.distributions.categorical import Categorical


def loss_args(parser):
    loss_parser = parser.add_argument_group("ML/DL loss arguments")
    loss_parser.add_argument('--loss-func', help='Loss function', choices=['mse', 'ce', 'kl_div'], default='ce')
    loss_parser.add_argument('--kl-penalty', help='Weight of KL regularization term', type=float, default=0.0)
    loss_parser.add_argument('--entropy-penalty', help='Weight of entropy regularization term', type=float, default=0.0)
    return parser


from dataclasses import dataclass, field
from typing import List
from ml.utils.enums import LossType


@dataclass
class LossConfig:    # RNN model arguments
    loss_func: LossType = LossType.ce
    loss_weight: List[float] = field(default_factory=lambda: [])  # The weights of all class about loss
    kl_penalty: float = 0.0      # Weight of KL regularization term
    entropy_penalty: float = 0.0      # Weight of entropy regularization term


def set_criterion(cfg, task_type, class_names):
    if sum(cfg.loss_weight) == 0:
        cfg.loss_weight = [1.0] * len(class_names)

    if task_type == 'regress' or cfg.loss_func == 'mse':
        criterion = torch.nn.MSELoss()
    elif cfg.loss_func.value == 'ce':
        criterion = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(cfg.loss_weight))
    elif cfg.loss_func.value == 'kl_div':
        criterion = KLLoss()
    else:
        raise NotImplementedError

    penalties = []
    if cfg.kl_penalty:
        penalties.append({'weight': cfg.kl_penalty, 'func': KLLoss(batch_wise=True)})
    if cfg.entropy_penalty:
        penalties.append({'weight': cfg.entropy_penalty, 'func': EntropyLoss()})

    return LossManager(criterion, penalties)


class LossManager(torch.nn.Module):
    def __init__(self, criterion, penalties: List):
        super(LossManager, self).__init__()
        self.criterion = criterion
        self.penalties = penalties

    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        loss = self.criterion(predict, target)
        for penalty in self.penalties:
            loss += penalty['weight'] * penalty['func'](predict, target)

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


