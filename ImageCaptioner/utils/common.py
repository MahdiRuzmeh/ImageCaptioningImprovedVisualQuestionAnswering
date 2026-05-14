"""Config + RNG parity helpers for captioner scripts.

Cross-project consistency
-------------------------
Mirrors ``VQA.utils.common`` so caption/VQA runs cite identical reproducibility footnotes in the
thesis (*Image captioning improved visual question answering*).

Examples
--------
::

    cfg = load_config("configs/default.yaml")
    set_seed(cfg["seed"])

See Also
--------
``VQA.utils.common`` — authoritative twin definitions.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML for caption training/evaluation.

    Args:
        path: Usually ``ImageCaptioner/configs/default.yaml``.

    Examples:
        >>> # cfg = load_config("configs/default.yaml")
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path_fields(cfg: Dict[str, Any], keys: Iterable[str]) -> None:
    """Expand ``~`` and resolve relative paths against the process cwd (in-place).

    Use absolute paths in YAML on Kaggle (e.g. ``/kaggle/input/...``); local runs typically
    ``cd ImageCaptioner`` so ``../dataset`` keeps prior layout semantics.
    """
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, str) and v:
            cfg[k] = str(Path(v).expanduser().resolve())


def set_seed(seed: int) -> None:
    """Delegate to same deterministic recipe as VQA utilities."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
