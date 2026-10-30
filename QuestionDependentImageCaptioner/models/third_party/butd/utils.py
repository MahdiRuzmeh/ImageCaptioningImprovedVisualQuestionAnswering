"""Minimal replacement for lxmert ``utils.py``: only ``Config`` (yaml → attribute access).

Finglish: download helper haye asli (wget / pickle / S3) inja nistan; weight ha local
ba ``torch.load(weights_only=True)`` load mishan (scripts/extract_visual.py).
"""

import copy

from yaml import safe_load

WEIGHTS_NAME = "pytorch_model.bin"


class Config:
    """Nested dict → attributes; har key ham ba esm-e khodesh ham UPPER (mesl-e asli)."""

    def __init__(self, dictionary: dict, name: str = "root", level: int = 0):
        self._name = name
        self._level = level
        d = {}
        for k, v in dictionary.items():
            if v is None:
                raise ValueError(f"config key {k!r} is None")
            v = copy.deepcopy(v)
            if isinstance(v, dict):
                v = Config(v, name=k, level=level + 1)
            d[k] = v
            setattr(self, k, v)
        self._pointer = d

    def __setattr__(self, key, val):
        self.__dict__[key] = val
        self.__dict__[key.upper()] = val

    @staticmethod
    def load_yaml(path):
        with open(path, encoding="utf-8") as f:
            return safe_load(f)


def _unavailable(*args, **kwargs):
    raise NotImplementedError("download helpers are not vendored; load weights locally")


cached_path = hf_bucket_url = is_remote_url = load_checkpoint = img_tensorize = _unavailable
