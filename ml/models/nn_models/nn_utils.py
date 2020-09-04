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
    def __init__(self, in_features, n_classes, n_fc=3, tagging=False):
        super(Predictor, self).__init__()
        self.in_features = in_features

        if n_fc == 1:
            self.predictor = nn.Sequential(nn.Linear(in_features, n_classes))
        else:
            self.predictor = nn.Sequential(
                nn.Linear(in_features, 1000), nn.ReLU(), nn.Dropout(p=0.2),
                nn.Linear(1000, 200), nn.ReLU(), nn.Dropout(p=0.2),
                nn.Linear(200, n_classes),
            )

        if n_classes >= 2:
            self.predictor = nn.Sequential(
                *self.predictor,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        if x.dim() >= 3:
            x = x.reshape(x.size(0), -1)
        return self.predictor(x)
