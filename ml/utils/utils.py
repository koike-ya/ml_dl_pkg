import json
from typing import Sequence, Dict

from ml.src.metrics import Metric

Metrics = Dict[str, Sequence[Metric]]


def dump_dict(path, dict_):
    dict_ = {key: str(value) for key, value in dict_.items()}

    with open(path, 'w') as f:
        json.dump(dict_, f, indent=4)
