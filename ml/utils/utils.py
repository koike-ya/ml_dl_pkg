import json
import os
import random
from typing import Sequence, Dict

import numpy as np
import torch
from ml.src.metrics import Metric

Metrics = Dict[str, Sequence[Metric]]


def dump_dict(path, dict_):
    dict_ = {key: str(value) for key, value in dict_.items()}

    with open(path, 'w') as f:
        json.dump(dict_, f, indent=4)


def init_seed(seed) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
