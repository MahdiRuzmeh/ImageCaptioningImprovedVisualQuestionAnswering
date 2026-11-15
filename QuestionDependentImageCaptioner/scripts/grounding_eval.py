"""Visual/question grounding test — how much does a QD captioner depend on the image and on the question?

Finglish:
    Hamoon test split, hamoon decode (beam 3), panj halat-e voroodi:
      normal         — tasvir-e khodesh + soal-e khodesh (= test_metrics)
      no_image       — "faghat soal": tasvir bi-etelaat (MMT: miangin-e region haye test; BLIP: tasvir-e
                       khakestari = 0 ba'd az normalize)
      no_question    — "faghat tasvir": soal khali (MMT: [BOS, EOS]; BLIP: "question:  caption:")
      swap_image     — "soal-e yeksan, tasvir-e digar": har row tasvir-e yek image-e *dige* az test migire
      swap_question  — "tasvir-e yeksan, soal-e digar": har row soal-e dige-ie hamoon image ro migire
                       (faghat image haye ba >= 2 soal-e motefavet)
    Score: CIDEr / BLEU-4 / AnswerHit / YesNoPolarity nesbat be javab-e *asli* (own);
    swap_question → ham "follow" (nesbat be javab + caption-e soal-e jadid). Changed = sahm-e caption
    haye ke ba normal fargh daran. Prior = baseline-e bedoon model (por-tekrar tarin javab-e train).
    Diversity (az normal): yek image ba chand soal → caption ha motefavet? yek soal roye chand image →
    caption ha motefavet?

    python scripts/grounding_eval.py --family mmt  --config configs/eval_local.yaml --ckpt a_full_butd/best.pt --name a_full_butd
    python scripts/grounding_eval.py --family blip --config configs/b_blip_full.yaml --ckpt <best.pt> --name b_blip_full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import score_block  # noqa: E402
from metrics import NON_LEXICAL_ANSWERS, answer_in_caption, caption_negated  # noqa: E402
from train import load_config, resolve_path_fields, set_seed, split_rows, tok  # noqa: E402

CONDITIONS = ("normal", "no_image", "no_question", "swap_image", "swap_question")
KEEP = ("BLEU-4", "CIDEr", "METEOR", "ROUGE-L", "AnswerHit", "AnswerHit_n", "YesNoPolarity", "YesNo_n", "n")


# ---------------------------------------------------------------------------
# Row transforms (model-agnostic)
# ---------------------------------------------------------------------------
def image_derangement(rows: List[Dict[str, Any]], seed: int) -> Dict[int, int]:
    """Finglish: har image_id → yek image_id-e *dige* (permute + cyclic shift → hich kodom sare jash nist)."""
    ids = sorted({int(r["image_id"]) for r in rows})
    g = torch.Generator().manual_seed(seed)
    perm = [ids[i] for i in torch.randperm(len(ids), generator=g).tolist()]
    return {a: b for a, b in zip(perm, perm[1:] + perm[:1])}


def question_swap(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    """Finglish: row i → row j az hamoon image ba soal-e motefavet (cyclic dar group). Bedoon joft → nist."""
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[int(r["image_id"])].append(i)
    out = {}
    for idx in groups.values():
        for k, i in enumerate(idx):
            for step in range(1, len(idx)):
                j = idx[(k + step) % len(idx)]
                if rows[j]["question"].strip().lower() != rows[i]["question"].strip().lower():
                    out[i] = j
                    break
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def scores(preds: List[str], rows: List[Dict[str, Any]]) -> Dict[str, float]:
    sc = score_block([tok(p) for p in preds], rows)
    return {k: round(float(sc[k]), 4) for k in KEEP if k in sc}


def changed(a: List[str], b: List[str]) -> float:
    return round(sum(x.strip().lower() != y.strip().lower() for x, y in zip(a, b)) / max(1, len(a)), 4)


def diversity(preds: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Finglish: (1) yek image, chand soal → chand caption-e motefavet? (2) yek soal, chand image → ?"""
    by_img: Dict[int, List[int]] = defaultdict(list)
    by_q: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_img[int(r["image_id"])].append(i)
        by_q[r["question"].strip().lower()].append(i)

    def distinct(idx: List[int]) -> float:
        return len({preds[i].strip().lower() for i in idx}) / len(idx)

    img_groups = [v for v in by_img.values() if len({rows[i]["question"].strip().lower() for i in v}) >= 2]
    q_groups = [v for v in by_q.values() if len({int(rows[i]["image_id"]) for i in v}) >= 2]
    # Finglish: soal-e yeksan, javab-e motefavet → model bayad caption-e motefavet bede
    q_diff_ans = [v for v in q_groups if len({rows[i]["answer"] for i in v}) >= 2]

    def lex_hit(idx: List[int]) -> Optional[float]:
        lx = [i for i in idx if " ".join(tok(rows[i]["answer"])) not in NON_LEXICAL_ANSWERS and tok(rows[i]["answer"])]
        if not lx:
            return None
        return sum(answer_in_caption(tok(rows[i]["answer"]), tok(preds[i])) for i in lx) / len(lx)

    flat = [i for v in q_diff_ans for i in v]
    return {
        "same_image_diff_questions": {
            "images": len(img_groups), "rows": sum(map(len, img_groups)),
            "distinct_caption_ratio": round(sum(distinct(v) for v in img_groups) / max(1, len(img_groups)), 4),
        },
        "same_question_diff_images": {
            "questions": len(q_groups), "rows": sum(map(len, q_groups)),
            "distinct_caption_ratio": round(sum(distinct(v) for v in q_groups) / max(1, len(q_groups)), 4),
        },
        "same_question_diff_answers": {
            "questions": len(q_diff_ans), "rows": len(flat),
            "distinct_caption_ratio": round(sum(distinct(v) for v in q_diff_ans) / max(1, len(q_diff_ans)), 4),
            "AnswerHit": None if lex_hit(flat) is None else round(lex_hit(flat), 4),
            "YesNoPolarity": round(sum(caption_negated(tok(preds[i])) == (rows[i]["answer"] == "no")
                                       for i in flat if rows[i]["answer"] in ("yes", "no"))
                                   / max(1, sum(rows[i]["answer"] in ("yes", "no") for i in flat)), 4),
        },
    }


def answer_prior(train_rows: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> List[str]:
    """Finglish: baseline-e bedoon tasvir va model: por-tekrar tarin javab-e train baraye 3 kalame-ye aval-e soal."""
    def key(q: str) -> str:
        return " ".join(tok(q)[:3])
    cnt: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    glob: Dict[str, int] = defaultdict(int)
    for r in train_rows:
        cnt[key(r["question"])][r["answer"]] += 1
        glob[r["answer"]] += 1
    top = max(glob, key=glob.get)
    out = []
    for r in rows:
        c = cnt.get(key(r["question"]))
        a = max(c, key=c.get) if c else top
        # Finglish: yes/no → caption-e manfi/mosbat; baghie → khod-e javab
        out.append("it is not" if a == "no" else ("it is" if a == "yes" else a))
    return out


# ---------------------------------------------------------------------------
# Model families: return decode(rows, blank_image) -> List[str]
# ---------------------------------------------------------------------------
DecodeFn = Callable[[List[Dict[str, Any]], bool], List[str]]


def build_mmt(cfg: Dict[str, Any], ckpt: Path, beam: int, device: torch.device) -> DecodeFn:
    from torch.utils.data import DataLoader

    from benchmark import DATA_KEYS, vocab_from_itos
    from decoding import generate_split_captions
    from models import build_captioner
    from train import VqaQdCaptionDataset, collate_batch, load_model_state, region_cache_dir_for_split

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = {**dict(state["config"]), **{k: cfg[k] for k in DATA_KEYS if k in cfg}}
    vocab, q_vocab = vocab_from_itos(state["vocab"]), vocab_from_itos(state["q_vocab"])
    model = build_captioner(mcfg, vocab, q_vocab)
    load_model_state(model, state["model"])
    model.to(device).eval()
    cache = region_cache_dir_for_split(mcfg, "test")
    enc = model.region_encoder
    orig = enc.forward
    mean_vec: Dict[str, torch.Tensor] = {}

    def run(rows: List[Dict[str, Any]], blank: bool) -> List[str]:
        ds = VqaQdCaptionDataset(
            mcfg["val_images_dir"], mcfg["train_captions_json"], vocab, q_vocab, int(mcfg["max_caption_len"]),
            int(mcfg.get("max_question_len", 14)), mcfg["val_image_filename_template"], rows=rows,
            images_from_cache=True, region_cache_dir=cache)
        dl = DataLoader(ds, batch_size=int(cfg.get("batch_size", 64)), shuffle=False, num_workers=0,
                        collate_fn=collate_batch)
        if blank and "v" not in mean_vec:
            # Finglish: miangin-e hame region token haye test (bedoon etelaat-e yek tasvir-e khas)
            ids = sorted({int(r["image_id"]) for r in rows})
            acc = None
            for k in range(0, len(ids), 256):
                t = orig(torch.zeros(1), image_ids=torch.tensor(ids[k:k + 256]), cache_dir=cache).float()
                acc = t.sum(0) if acc is None else acc + t.sum(0)
            mean_vec["v"] = (acc / len(ids)).cpu()

        def fwd(images, image_ids=None, **kw):
            out = orig(images, image_ids=image_ids, **kw)
            return mean_vec["v"].to(out).expand_as(out).clone() if blank else out

        enc.forward = fwd
        try:
            hyps, _ = generate_split_captions(model, dl, vocab.itos, device, int(mcfg["max_caption_len"]),
                                              beam_size=beam, region_cache_dir=cache)
        finally:
            enc.forward = orig
        return [" ".join(h) for h in hyps]

    run.params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # type: ignore[attr-defined]
    return run


from train_blip import QdBlipDataset  # noqa: E402


class BlankBlipDataset(QdBlipDataset):
    """Finglish: pixel = 0 ba'd az normalize = tasvir-e khakestari-e miangin (module level → pickle-able)."""

    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["pixel_values"] = torch.zeros_like(item["pixel_values"])
        return item


def build_blip(cfg: Dict[str, Any], ckpt: Path, beam: int, device: torch.device) -> DecodeFn:
    from train_blip import QdBlipDataset, build_model, decode_split

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model, tokenizer = build_model(cfg, device, state["model"])
    model.eval()

    def run(rows: List[Dict[str, Any]], blank: bool) -> List[str]:
        ds = (BlankBlipDataset if blank else QdBlipDataset)(rows, cfg, tokenizer, model.bos, model.sep)
        return decode_split(model, ds, tokenizer, cfg, device, beam, 0, 1, False)

    run.params = sum(v.numel() for v in state["model"].values())  # type: ignore[attr-defined]
    return run


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--family", required=True, choices=("mmt", "blip"))
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--limit", type=int, default=None, help="First N test rows (smoke test)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    path = Path(args.config) if Path(args.config).is_absolute() else PROJECT_ROOT / args.config
    cfg = load_config(str(path))
    resolve_path_fields(cfg, ("train_captions_json", "val_captions_json", "train_images_dir", "val_images_dir",
                              "train_region_cache_dir", "val_region_cache_dir", "test_region_cache_dir"))
    if cfg.get("blip_model") and not str(cfg["blip_model"]).startswith("Salesforce/"):
        resolve_path_fields(cfg, ("blip_model",))
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt).resolve()

    rows = split_rows(cfg, "test", cfg["train_captions_json"])
    if args.limit:
        rows = rows[: args.limit]
    decode = (build_mmt if args.family == "mmt" else build_blip)(cfg, ckpt, args.beam, device)
    conds = [c.strip() for c in args.conditions.split(",")]

    img_map = image_derangement(rows, int(cfg.get("seed", 42)))
    q_map = question_swap(rows)
    result: Dict[str, Any] = {"name": args.name, "family": args.family, "ckpt": str(ckpt), "beam": args.beam,
                              "params": getattr(decode, "params", None), "test_rows": len(rows), "conditions": {}}
    preds: Dict[str, List[str]] = {}
    for c in conds:
        t0 = time.time()
        if c == "normal":
            inp, own = rows, rows
        elif c == "no_image":
            inp, own = rows, rows
        elif c == "no_question":
            inp, own = [{**r, "question": ""} for r in rows], rows
        elif c == "swap_image":
            inp, own = [{**r, "image_id": img_map[int(r["image_id"])]} for r in rows], rows
        elif c == "swap_question":
            idx = sorted(q_map)
            inp = [{**rows[i], "question": rows[q_map[i]]["question"]} for i in idx]
            own = [rows[i] for i in idx]
        else:
            raise SystemExit(f"unknown condition {c}")
        out = decode(inp, c == "no_image")
        preds[c] = out
        entry: Dict[str, Any] = {"own": scores(out, own), "secs": round(time.time() - t0, 1)}
        if c == "swap_question":
            entry["follow"] = scores(out, [rows[q_map[i]] for i in idx])
            if "normal" in preds:
                entry["changed_vs_normal"] = changed(out, [preds["normal"][i] for i in idx])
            entry["examples"] = [{"image_id": rows[i]["image_id"], "orig_question": rows[i]["question"],
                                  "orig_answer": rows[i]["answer"], "new_question": rows[q_map[i]]["question"],
                                  "new_answer": rows[q_map[i]]["answer"], "pred": out[k],
                                  "normal_pred": preds["normal"][i] if "normal" in preds else None}
                                 for k, i in enumerate(idx[:400:40])]
        else:
            if c != "normal" and "normal" in preds:
                entry["changed_vs_normal"] = changed(out, preds["normal"])
            entry["examples"] = [{"image_id": r["image_id"], "shown_image_id": x["image_id"],
                                  "question": r["question"], "answer": r["answer"], "gt": r["caption"],
                                  "pred": out[k], "normal_pred": preds["normal"][k] if "normal" in preds else None}
                                 for k, (r, x) in list(enumerate(zip(rows, inp)))[:400:40]]
        result["conditions"][c] = entry
        print(f"{args.name} {c:14s} " + "  ".join(f"{k}={v}" for k, v in entry["own"].items())
              + (f"  follow_AnswerHit={entry['follow']['AnswerHit']}" if "follow" in entry else "")
              + (f"  changed={entry.get('changed_vs_normal')}" if "changed_vs_normal" in entry else "")
              + f"  ({entry['secs']}s)", flush=True)

    train_rows = split_rows(cfg, "train", cfg["train_captions_json"])
    result["prior"] = scores(answer_prior(train_rows, rows), rows)
    if "normal" in preds:
        result["diversity"] = diversity(preds["normal"], rows)
    n = result["conditions"].get("normal", {}).get("own", {})
    if n:
        def dep(c: str, k: str) -> Optional[float]:
            v = result["conditions"].get(c, {}).get("own", {}).get(k)
            return None if v is None else round(n[k] - v, 4)
        result["dependence"] = {
            "image_dep_AnswerHit (normal - swap_image)": dep("swap_image", "AnswerHit"),
            "image_dep_AnswerHit (normal - no_image)": dep("no_image", "AnswerHit"),
            "question_dep_AnswerHit (normal - no_question)": dep("no_question", "AnswerHit"),
            "image_dep_Polarity (normal - swap_image)": dep("swap_image", "YesNoPolarity"),
            "question_dep_Polarity (normal - no_question)": dep("no_question", "YesNoPolarity"),
        }
    result["predictions"] = {c: v for c, v in preds.items()}
    out = Path(args.out) if args.out else ckpt.parent / "grounding.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("prior", "diversity", "dependence") if k in result}, indent=1))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
