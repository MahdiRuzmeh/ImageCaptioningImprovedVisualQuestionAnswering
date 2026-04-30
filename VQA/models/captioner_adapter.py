import importlib.util
from pathlib import Path
from typing import Any, Dict

import torch


def load_captioner(cfg: Dict[str, Any], vocab_size: int, pad_id: int, device: torch.device) -> torch.nn.Module:
    p = Path(cfg["captioner_project_root"]).resolve() / "models" / "captioner_v1.py"
    spec = importlib.util.spec_from_file_location("cap_mod", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    cls = getattr(mod, cfg.get("captioner_class", "ImageCaptionerV1"))
    m = cls(vocab_size=vocab_size, pad_id=pad_id, word_dim=cfg["word_dim"], hidden_dim=cfg["hidden_dim"], max_regions=cfg["max_regions"], question_dim=cfg["question_dim"])
    ck = Path(cfg["captioner_ckpt"])
    if ck.exists():
        st = torch.load(ck, map_location="cpu")
        m.load_state_dict(st.get("model", st), strict=False)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad = False
    return m
