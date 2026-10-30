"""Visual token cache baraye visual-feature comparison (V0 / V2 / V3; V1 BUTD az Kaggle).

Finglish:
    Har image → ``{cache_dir}/{image_id}.pt`` = {"feats": (T, D) fp16, "boxes": (T, 4) fp16
    normalized xyxy}. ``models/visual_encoder.py`` (CachedVisualEncoder) hamino mikhoone.
    Backend (``visual_backend`` dar config):
      frcnn_box — V0: hamoon torchvision Faster R-CNN, vali aspect ratio hefz mishe (resize
                  faghat dakhel-e detector), 32 RPN proposal-e aval + 1 box-e kol-e tasvir,
                  1024-D box-head feature + box.
      butd      — V1: bottom-up attention Faster R-CNN (Visual Genome, 1600 object + 400 attribute,
                  ResNet-101 C4; models/third_party/butd), 36 detection × 2048-D + box, + 1 global
                  token (miangin-e 36 ta). Weight: outputs/butd_weights (unc-nlp/frcnn-vg-finetuned).
      siglip    — V2: SigLIP so400m/14-384 patch token ha (27x27) → avg-pool 12x12 = 144 token
                  (box = cell) + 1 global token (MAP pooler), 1152-D. Text-aligned.
      dinov2    — V3: DINOv2-L/14 + registers, 448x448 → 32x32 patch → 12x12 = 144 token + CLS, 1024-D.
                  Ghavi-tarin feature-e local/spatial (bedoon text).
    Model ha az Hugging Face; HF_HOME ro roye E: bezar (C: ja nadare):
      HF_HOME=outputs/hf_cache python scripts/extract_visual.py --config configs/vis_v2_siglip.yaml

    python scripts/extract_visual.py --config configs/vis_v0_frcnn_box.yaml
    # Finglish: image haye digar (e.g. CaptionVQA split) → hamoon cache (COCO image_id unique-e)
    python scripts/extract_visual.py --config configs/vis_v1_butd.yaml --batch-size 1         --ids-json ../CaptionVQA/outputs/small/split/test.json --images-dir ../dataset/val2014         --template COCO_val2014_{image_id:012d}.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import load_config, region_cache_dir_for_split, resolve_path_fields, split_rows  # noqa: E402

Batch = Tuple[List[torch.Tensor], List[torch.Tensor]]  # per image: feats (T, D), boxes (T, 4)


class _Images(Dataset):
    def __init__(self, ids, images_dir: Path, template: str) -> None:
        self.ids, self.dir, self.template = ids, images_dir, template

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        iid = self.ids[i]
        return Image.open(self.dir / self.template.format(image_id=iid)).convert("RGB"), iid


def _collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


# ---------------------------------------------------------------------------
# Backends: build(cfg, device) → fn(list[PIL]) → (feats list, boxes list)
# ---------------------------------------------------------------------------
def build_frcnn_box(cfg, device) -> Callable[[List[Image.Image]], Batch]:
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
    from torchvision.transforms.functional import to_tensor

    det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()
    k = int(cfg["max_regions"]) - 1  # Finglish: token-e akhar = kol-e tasvir

    @torch.no_grad()
    def run(imgs: List[Image.Image]) -> Batch:
        # Finglish: detector khodesh normalize + resize (min 800, aspect hefz) mikone
        t, _ = det.transform([to_tensor(im).to(device) for im in imgs], None)
        feats = det.backbone(t.tensors)
        props, _ = det.rpn(t, feats, None)
        boxes = []
        for p, (h, w) in zip(props, t.image_sizes):
            full = torch.tensor([[0.0, 0.0, w, h]], device=device)
            boxes.append(torch.cat([p[:k], full]))
        roi = det.roi_heads.box_head(det.roi_heads.box_roi_pool(feats, boxes, t.image_sizes))
        out_f, out_b = [], []
        for f, b, (h, w) in zip(torch.split(roi, [len(b) for b in boxes]), boxes, t.image_sizes):
            scale = torch.tensor([w, h, w, h], device=device, dtype=b.dtype)
            out_f.append(f)
            out_b.append((b / scale).clamp(0, 1))
        return out_f, out_b

    return run


def build_butd(cfg, device):
    import numpy as np

    from models.third_party.butd.modeling_frcnn import GeneralizedRCNN
    from models.third_party.butd.processing_image import Preprocess
    from models.third_party.butd.utils import Config

    wdir = Path(cfg.get("visual_model", "outputs/butd_weights"))
    if not wdir.is_absolute():
        wdir = PROJECT_ROOT / wdir
    fcfg = Config(Config.load_yaml(wdir / "config.yaml"))
    fcfg.MODEL.DEVICE = str(device)
    model = GeneralizedRCNN(fcfg)
    model.load_state_dict(torch.load(wdir / "pytorch_model.bin", map_location="cpu", weights_only=True))
    model.to(device).eval()
    prep = Preprocess(fcfg)
    k = int(cfg["max_regions"]) - 1  # Finglish: token-e akhar = global

    @torch.no_grad()
    def run(imgs: List[Image.Image]) -> Batch:
        # Finglish: Preprocess HWC BGR 0..255 mikhad (caffe format), resize min 800 / max 1333
        x = [torch.from_numpy(np.asarray(im)[:, :, ::-1].copy()).float() for im in imgs]
        images, sizes, scales = prep(x)
        out = model(images, sizes, scales_yx=scales, padding="max_detections", max_detections=k,
                    return_tensors="pt")
        feats, boxes = [], []
        full = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device)
        for f, b, n in zip(out["roi_features"], out["normalized_boxes"], out["preds_per_image"].tolist()):
            f, b = f[:n].to(device), b[:n].to(device).float().clamp(0, 1)
            feats.append(torch.cat([f, f.mean(0, keepdim=True)]))
            boxes.append(torch.cat([b, full]))
        return feats, boxes

    return run


def _cell_boxes(g: int, device) -> torch.Tensor:
    """g x g grid → (g*g, 4) normalized xyxy, row-major (hamoon tartib-e flatten)."""
    r = torch.arange(g, device=device, dtype=torch.float32) / g
    y, x = torch.meshgrid(r, r, indexing="ij")
    return torch.stack([x, y, x + 1 / g, y + 1 / g], dim=-1).reshape(-1, 4)


def _build_grid(cfg, device, load_model, size: int, mean, std, global_token) -> Callable:
    """Finglish: ViT patch token ha → adaptive avg-pool be grid x grid + yek global token."""
    import torch.nn.functional as F
    from torchvision import transforms

    model = load_model().to(device).eval()
    g = int(cfg.get("visual_grid", 12))
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    full = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device)
    boxes = torch.cat([_cell_boxes(g, device), full])

    @torch.no_grad()
    def run(imgs: List[Image.Image]) -> Batch:
        x = torch.stack([tf(im) for im in imgs]).to(device, torch.float16)
        out = model(pixel_values=x)
        patches, glob = global_token(out)
        n, t, d = patches.shape
        side = int(round(t ** 0.5))
        grid = patches.transpose(1, 2).reshape(n, d, side, side).float()
        grid = F.adaptive_avg_pool2d(grid, g).flatten(2).transpose(1, 2)  # (n, g*g, d)
        feats = torch.cat([grid, glob.float()[:, None]], dim=1)
        return list(feats), [boxes] * n

    return run


def build_siglip(cfg, device):
    from transformers import SiglipVisionModel

    name = cfg.get("visual_model", "google/siglip-so400m-patch14-384")
    return _build_grid(cfg, device, lambda: SiglipVisionModel.from_pretrained(name, torch_dtype=torch.float16),
                       384, [0.5] * 3, [0.5] * 3,
                       lambda o: (o.last_hidden_state, o.pooler_output))


def build_dinov2(cfg, device):
    from transformers import AutoModel

    name = cfg.get("visual_model", "facebook/dinov2-with-registers-large")
    model_ref = {}

    def load():
        model_ref["m"] = AutoModel.from_pretrained(name, torch_dtype=torch.float16)
        return model_ref["m"]

    def split(o):
        # Finglish: [CLS, 4 register, patch ha...]; register ha dor rikhte mishan
        n_reg = int(getattr(model_ref["m"].config, "num_register_tokens", 0))
        h = o.last_hidden_state
        return h[:, 1 + n_reg:], h[:, 0]

    return _build_grid(cfg, device, load, 448, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], split)


BACKENDS: Dict[str, Callable] = {
    "frcnn_box": build_frcnn_box,
    "butd": build_butd,
    "siglip": build_siglip,
    "dinov2": build_dinov2,
}


def main() -> None:
    p = argparse.ArgumentParser(description="Extract cached visual tokens")
    p.add_argument("--config", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ids-json", default=None, help="JSON list of records with image_id (instead of the QD split)")
    p.add_argument("--images-dir", default=None)
    p.add_argument("--template", default=None)
    args = p.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(str(config_path))
    resolve_path_fields(cfg, ("train_captions_json", "train_images_dir", "val_images_dir",
                              "train_region_cache_dir", "val_region_cache_dir", "test_region_cache_dir"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = BACKENDS[cfg["visual_backend"]](cfg, device)
    dim = int(cfg["visual_feat_dim"])

    jobs = []  # (name, cache_dir, ids, images_dir, template)
    if args.ids_json:
        recs = json.loads(Path(args.ids_json).read_text(encoding="utf-8"))
        jobs.append((Path(args.ids_json).stem, Path(region_cache_dir_for_split(cfg, "train")),
                     sorted({int(r["image_id"]) for r in recs}), Path(args.images_dir), args.template))
    else:
        for split in ("train", "val", "test"):
            img_key = "train" if split == "train" else "val"
            jobs.append((split, Path(region_cache_dir_for_split(cfg, split)),
                         sorted({int(r["image_id"]) for r in split_rows(cfg, split, cfg["train_captions_json"])}),
                         Path(cfg[f"{img_key}_images_dir"]), cfg[f"{img_key}_image_filename_template"]))
    for split, cache_dir, ids, images_dir, template in jobs:
        cache_dir.mkdir(parents=True, exist_ok=True)
        todo = [i for i in ids if not (cache_dir / f"{i}.pt").exists()]
        print(f"split={split} images={len(ids)} to_extract={len(todo)} dir={cache_dir}")
        dl = DataLoader(_Images(todo, images_dir, template),
                        batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=_collate)
        for imgs, chunk in tqdm(dl, desc=split):
            feats, boxes = run(imgs)
            for iid, f, b in zip(chunk, feats, boxes):
                assert f.shape[1] == dim, f"feat dim {f.shape[1]} != visual_feat_dim {dim}"
                torch.save({"feats": f.half().cpu(), "boxes": b.half().cpu()}, cache_dir / f"{iid}.pt")


if __name__ == "__main__":
    main()
