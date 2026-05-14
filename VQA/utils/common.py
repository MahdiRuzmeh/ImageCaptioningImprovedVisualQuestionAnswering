"""Shared helpers for VQA entry scripts.

Paper alignment
---------------
``set_seed`` implements reproducibility commitments stated in thesis methodology (fixed splits +
optimization RNG). ``load_config`` centralizes YAML hyperparameters referenced by experiment tables.

Examples
--------
::

    from utils.common import load_config, set_seed

    cfg = load_config("configs/default.yaml")
    set_seed(cfg["seed"])

``load_config`` returns nested dicts/lists parsed by PyYAML—access optional keys defensively::

    resume = cfg.get("resume_from")
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Parse YAML training/evaluation configuration.

    Args:
        path: Relative or absolute path (typically ``configs/default.yaml``).

    Returns:
        Dictionary merged by PyYAML (may contain ints/floats/bools/nested maps).

    Examples:
        >>> # cfg = load_config("configs/default.yaml")
        >>> # bs = cfg["batch_size"]
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path_fields(cfg: Dict[str, Any], keys: Iterable[str]) -> None:
    """Expand ``~`` and resolve relative paths against the process cwd (in-place).

    Kaggle notebooks should set absolute paths under ``/kaggle/input/...``; for local use,
    ``cd VQA`` keeps ``../dataset``-style entries consistent with earlier versions.
    """
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, str) and v:
            cfg[k] = str(Path(v).expanduser().resolve())


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/Torch and disable cudnn benchmarking for deterministic kernels.

    Args:
        seed: Integer propagated from YAML / CLI.

    Examples:
        >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
