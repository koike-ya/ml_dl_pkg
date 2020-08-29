from torch import nn


def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)

    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)

    return model


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


class Predictor(nn.Module):
    def __init__(self, in_features, n_classes):
        super(Predictor, self).__init__()
        self.in_features = in_features
        self.predictor = nn.Linear(in_features, n_classes)
        if n_classes >= 2:
            self.predictor = nn.Sequential(
                self.predictor,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        return self.predictor(x)
