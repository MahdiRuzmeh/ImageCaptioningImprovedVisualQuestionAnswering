"""Captioner probe — question vs image dependence + answer metrics (yek report-e kamel).

Finglish:
    Checkpoint ro roye yek split ba 5 halat decode mikone:
      normal  — tasvir + soal-e khodesh
      shuffle — tasvir-e *dige* (derangement), hamoon soal   → "same question, different image"
      mean    — region ha = miangin-e split (tasvir-e "average"), hamoon soal → question only
      mask    — bedoon region (faghat soal)                  → question only
      noq     — tasvir-e khodesh, soal khali ([BOS, EOS])     → image only (caption-e generic)
    + prior (bedoon model): javab-e por-tekrar-e train baraye (qtype, 2 kalame aval soal).

    Metric ha (har halat, kol + har qtype = yes/no | number | color | other):
      BLEU-1/4, CIDEr-D, ROUGE-L (nesbat be QD reference caption)
      AnswerHit       — javab (gheir-e yes/no) toye caption hast
      PolarityAcc     — yes/no: caption manfi (no/not/n't/...) ⇔ javab "no"
      CountAcc        — number: avalin adad-e caption == javab
      AnswerConsistency (AC) — caption ba javab-e GT jor dar miad: yes/no → Polarity,
                         number → Count, baghie → AnswerHit (roye HAME item ha)
    Pair analysis (halat normal):
      same_image  — yek tasvir, do soal-e motefavet (qtype yeksan / motefavet)
      same_question — yek soal, do tasvir-e motefavet
    noq ham ba COCO human caption haye hamoon tasvir score mishe (generic caption).

    python scripts/captioner_probe.py --config configs/kaggle_mini_local.yaml \\
        --ckpt outputs/kaggle_mini/best.pt --split test --out outputs/probe/kaggle_mini_test.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import DATA_KEYS, PATH_KEYS, vocab_from_itos  # noqa: E402
from decoding import generate_split_captions  # noqa: E402
from image_ablation import derangement, patch_regions, split_mean_region  # noqa: E402
from metrics import _ZERO_ALIASES, _norm_num, answer_in_caption, compute_caption_metrics  # noqa: E402
from models import build_captioner  # noqa: E402
from train import (  # noqa: E402
    PROJECT_ROOT,
    build_split_dataset,
    collate_batch,
    load_config,
    load_model_state,
    region_cache_dir_for_split,
    resolve_path_fields,
    set_seed,
    split_rows,
    tok,
)

MODES = ("normal", "shuffle", "mean", "mask", "noq")
QTYPES = ("yes/no", "number", "color", "other")
NEG_WORDS = {"no", "not", "nothing", "none", "nobody", "never", "neither", "nor", "nowhere", "cannot"}
COCO_CAPTIONS = PROJECT_ROOT.parents[3] / "dataset" / "MSCOCO Captions" / "2014" / "captions_train2014.json"


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def qtype(s: Dict[str, Any]) -> str:
    a = " ".join(tok(str(s["answer"])))
    if a in ("yes", "no"):
        return "yes/no"
    if a.isdigit() or s.get("rule") == "how_many":
        return "number"
    if s.get("rule") == "what_color" or re.match(r"^what colou?rs?\b", s["question"].lower()):
        return "color"
    return "other"


def is_negative(words: List[str]) -> bool:
    return any(w in NEG_WORDS or w.endswith("n't") for w in words)


def caption_count(words: List[str]) -> Optional[str]:
    """Finglish: avalin adad-e caption (digit / number word / "no"=0); None age nadare."""
    for w in words:
        n = _norm_num(w)
        if n.isdigit():
            return str(int(n))
        if w in _ZERO_ALIASES:
            return "0"
    return None


def item_checks(hyp: List[str], s: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    """Polarity / Count / AnswerHit baraye yek item (None = in metric baraye in item nist)."""
    t, ans = qtype(s), tok(str(s["answer"]))
    out: Dict[str, Optional[bool]] = {"polarity": None, "count": None, "hit": None}
    if t == "yes/no":
        out["polarity"] = is_negative(hyp) == (ans == ["no"])
    else:
        out["hit"] = answer_in_caption(ans, hyp)
        if t == "number" and len(ans) == 1:
            gt = _norm_num(ans[0])
            out["count"] = gt.isdigit() and caption_count(hyp) == str(int(gt))
    ac = out["polarity"] if t == "yes/no" else (out["count"] if out["count"] is not None else out["hit"])
    out["ac"] = bool(ac)
    return out


def score(hyps: List[List[str]], samples: List[Dict[str, Any]], refs=None, caption_metrics=True) -> Dict[str, float]:
    out: Dict[str, float] = {"n": float(len(samples))}
    if caption_metrics:
        refs = refs or [[tok(s["caption"])] for s in samples]
        m = compute_caption_metrics(hyps, refs)
        out.update({k: m[k] for k in ("BLEU-1", "BLEU-4", "CIDEr", "ROUGE-L")})
    sums, cnt = Counter(), Counter()
    for h, s in zip(hyps, samples):
        for k, v in item_checks(h, s).items():
            if v is not None:
                sums[k] += int(v)
                cnt[k] += 1
    names = {"hit": "AnswerHit", "polarity": "PolarityAcc", "count": "CountAcc", "ac": "AnswerConsistency"}
    for k, name in names.items():
        out[name] = sums[k] / cnt[k] if cnt[k] else float("nan")
        out[f"{name}_n"] = float(cnt[k])
    out["CountGiven"] = (sum(caption_count(h) is not None for h, s in zip(hyps, samples) if qtype(s) == "number")
                         / max(1, sum(qtype(s) == "number" for s in samples)))
    out["Negative"] = sum(is_negative(h) for h in hyps) / max(1, len(hyps))
    out["QuestionWordShare"] = sum(
        sum(w in set(tok(s["question"])) for w in h) / len(h) for h, s in zip(hyps, samples) if h
    ) / max(1, sum(bool(h) for h in hyps))
    out["Distinct"] = len({" ".join(h) for h in hyps}) / max(1, len(hyps))
    return out


def score_by_type(hyps, samples, **kw) -> Dict[str, Dict[str, float]]:
    out = {"all": score(hyps, samples, **kw)}
    for t in QTYPES:
        idx = [i for i, s in enumerate(samples) if qtype(s) == t]
        if idx:
            refs = [kw["refs"][i] for i in idx] if kw.get("refs") else None
            out[t] = score([hyps[i] for i in idx], [samples[i] for i in idx],
                           **{**kw, "refs": refs})
    return out


# ---------------------------------------------------------------------------
# Pair analyses (normal decode)
# ---------------------------------------------------------------------------
def bag_f1(a: List[str], b: List[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    if not common:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def same_image_pairs(hyps, samples) -> Dict[str, Dict[str, float]]:
    """Finglish: yek tasvir, do soal-e motefavet — caption soal ro donbal mikone?"""
    by_img: Dict[int, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_img[int(s["image_id"])].append(i)
    checks = [item_checks(h, s) for h, s in zip(hyps, samples)]
    refs = [tok(s["caption"]) for s in samples]
    ans = [tok(str(s["answer"])) for s in samples]
    res: Dict[str, Counter] = defaultdict(Counter)
    for idx in by_img.values():
        for a, i in enumerate(idx):
            for j in idx[a + 1:]:
                if tok(samples[i]["question"]) == tok(samples[j]["question"]):
                    continue
                key = "diff_type" if qtype(samples[i]) != qtype(samples[j]) else "same_type"
                for k in (key, "all"):
                    c = res[k]
                    c["pairs"] += 1
                    c["caption_changed"] += hyps[i] != hyps[j]
                    for x, y in ((i, j), (j, i)):
                        own, other = bag_f1(hyps[x], refs[x]), bag_f1(hyps[x], refs[y])
                        c["align"] += 1.0 if own > other else (0.5 if own == other else 0.0)
                        c["align_n"] += 1
                    c["both_ac"] += checks[i]["ac"] and checks[j]["ac"]
                    lex = all(qtype(samples[x]) != "yes/no" for x in (i, j)) and ans[i] != ans[j]
                    if lex:
                        c["lex_n"] += 2
                        c["own_hit"] += checks[i]["hit"] + checks[j]["hit"]
                        c["cross_hit"] += answer_in_caption(ans[j], hyps[i]) + answer_in_caption(ans[i], hyps[j])
    out = {}
    for k, c in res.items():
        out[k] = {
            "pairs": float(c["pairs"]),
            "images": float(sum(1 for idx in by_img.values() if len(idx) >= 2)) if k == "all" else float("nan"),
            "caption_changed": c["caption_changed"] / max(1, c["pairs"]),
            "question_alignment": c["align"] / max(1, c["align_n"]),
            "both_consistent": c["both_ac"] / max(1, c["pairs"]),
            "own_answer_hit": c["own_hit"] / max(1, c["lex_n"]),
            "cross_answer_hit": c["cross_hit"] / max(1, c["lex_n"]),
            "lexical_pairs": c["lex_n"] / 2,
        }
    return out


def extracted_answer(h: List[str], s: Dict[str, Any]) -> str:
    t = qtype(s)
    if t == "yes/no":
        return "no" if is_negative(h) else "yes"
    if t == "number":
        return caption_count(h) or "-"
    return " ".join(h)


def same_question_pairs(hyps, samples) -> Dict[str, float]:
    """Finglish: yek soal, do tasvir-e motefavet — javab-e caption tasvir ro donbal mikone?"""
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[" ".join(tok(s["question"]))].append(i)
    checks = [item_checks(h, s) for h, s in zip(hyps, samples)]
    c = Counter()
    n_groups = n_items = 0
    for idx in groups.values():
        if len({int(samples[i]["image_id"]) for i in idx}) < 2:
            continue
        n_groups += 1
        n_items += len(idx)
        for a, i in enumerate(idx):
            for j in idx[a + 1:]:
                if samples[i]["image_id"] == samples[j]["image_id"]:
                    continue
                same = tok(str(samples[i]["answer"])) == tok(str(samples[j]["answer"]))
                k = "same" if same else "diff"
                c[f"{k}_n"] += 1
                c[f"{k}_caption_equal"] += hyps[i] == hyps[j]
                c[f"{k}_answer_equal"] += extracted_answer(hyps[i], samples[i]) == extracted_answer(hyps[j], samples[j])
                c[f"{k}_both_consistent"] += checks[i]["ac"] and checks[j]["ac"]
    r = lambda k, n: c[k] / c[n] if c[n] else float("nan")  # noqa: E731
    return {
        "groups": float(n_groups), "items": float(n_items),
        "diff_answer_pairs": float(c["diff_n"]),
        "diff_answer_caption_changed": 1 - r("diff_caption_equal", "diff_n"),
        "diff_answer_extracted_answer_changed": 1 - r("diff_answer_equal", "diff_n"),
        "diff_answer_both_consistent": r("diff_both_consistent", "diff_n"),
        "same_answer_pairs": float(c["same_n"]),
        "same_answer_caption_equal": r("same_caption_equal", "same_n"),
        "same_answer_extracted_answer_equal": r("same_answer_equal", "same_n"),
        "same_answer_both_consistent": r("same_both_consistent", "same_n"),
    }


# ---------------------------------------------------------------------------
# Baselines / patches
# ---------------------------------------------------------------------------
def answer_prior(cfg: Dict[str, Any], samples: List[Dict[str, Any]]) -> List[List[str]]:
    """Finglish: question-only baseline (yes/no ham) — javab-e por-tekrar-e train baraye (qtype, 2 kalame aval)."""
    train = split_rows(cfg, "train", cfg["train_captions_json"])
    by_key, by_type = defaultdict(Counter), defaultdict(Counter)
    for r in train:
        a = " ".join(tok(str(r["answer"])))
        by_key[(qtype(r), tuple(tok(r["question"])[:2]))][a] += 1
        by_type[qtype(r)][a] += 1
    out = []
    for s in samples:
        c = by_key.get((qtype(s), tuple(tok(s["question"])[:2]))) or by_type[qtype(s)]
        out.append(c.most_common(1)[0][0].split())
    return out


def patch_question(model: torch.nn.Module, empty: bool) -> None:
    """Finglish: noq → soal = [BOS, EOS] (bedoon kalame)."""
    if not hasattr(model, "_orig_generate"):
        model._orig_generate = model.generate_caption
    orig = model._orig_generate

    def generate_caption(image, question_ids=None, *a, **kw):
        if empty:
            question_ids = torch.tensor([[1, 2]], device=image.device).expand(image.size(0), 2)
        return orig(image, question_ids, *a, **kw)

    model.generate_caption = generate_caption


def coco_refs(samples: List[Dict[str, Any]]) -> List[List[List[str]]]:
    caps: Dict[int, List[List[str]]] = defaultdict(list)
    for a in json.loads(COCO_CAPTIONS.read_text(encoding="utf-8"))["annotations"]:
        caps[int(a["image_id"])].append(tok(a["caption"]))
    return [caps[int(s["image_id"])] for s in samples]


def main() -> None:
    p = argparse.ArgumentParser(description="QD captioner probe (question/image dependence)")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    bench_cfg = load_config(str(config_path))
    resolve_path_fields(bench_cfg, PATH_KEYS)
    seed = int(bench_cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(Path(args.ckpt).resolve(), map_location="cpu", weights_only=False)
    cfg = {**dict(state.get("config") or {}), **{k: bench_cfg[k] for k in DATA_KEYS if k in bench_cfg}}
    cfg.setdefault("train_captions_json", bench_cfg["train_captions_json"])
    vocab, q_vocab = vocab_from_itos(state["vocab"]), vocab_from_itos(state["q_vocab"])
    model = build_captioner(cfg, vocab, q_vocab)
    load_model_state(model, state["model"])
    model.to(device).eval()

    ds = build_split_dataset(cfg, args.split, vocab, q_vocab)
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 16)), shuffle=False,
                        num_workers=0, collate_fn=collate_batch)
    cache_dir = region_cache_dir_for_split(cfg, args.split)
    samples = ds.samples
    id_map = derangement(sorted({int(s["image_id"]) for s in samples}), seed)
    mean_vec = split_mean_region(model, loader, device, cache_dir)

    hyps: Dict[str, List[List[str]]] = {}
    for mode in MODES:
        set_seed(seed)
        patch_regions(model, "normal" if mode == "noq" else mode, id_map, mean_vec)
        patch_question(model, mode == "noq")
        hyps[mode], _ = generate_split_captions(model, loader, vocab.itos, device,
                                                int(cfg["max_caption_len"]), beam_size=args.beam,
                                                region_cache_dir=cache_dir)
        print(f"decoded {mode}")
    hyps["prior"] = answer_prior(cfg, samples)

    results: Dict[str, Any] = {}
    for mode, h in hyps.items():
        results[mode] = score_by_type(h, samples, caption_metrics=mode != "prior")
        if mode not in ("normal", "prior"):
            results[mode]["all"]["changed_vs_normal"] = sum(
                a != b for a, b in zip(hyps["normal"], h)) / len(h)
        o = results[mode]["all"]
        print(f"{mode:8s} " + "  ".join(f"{k}={o[k]:.3f}" for k in (
            "CIDEr", "BLEU-4", "AnswerHit", "PolarityAcc", "CountAcc", "AnswerConsistency") if k in o))
    coco = coco_refs(samples)
    results["coco_reference"] = {m: score(hyps[m], samples, refs=coco) for m in ("normal", "noq")}
    results["same_image"] = same_image_pairs(hyps["normal"], samples)
    results["same_question"] = same_question_pairs(hyps["normal"], samples)
    results["same_question_shuffle_consistency"] = {
        "answer_changed_vs_normal": sum(
            extracted_answer(a, s) != extracted_answer(b, s)
            for a, b, s in zip(hyps["normal"], hyps["shuffle"], samples)) / len(samples)}
    results["noq_captions_per_image"] = (
        len({(int(s["image_id"]), " ".join(h)) for s, h in zip(samples, hyps["noq"])})
        / len({int(s["image_id"]) for s in samples}))
    for k in ("same_image", "same_question"):
        print(k, json.dumps(results[k], indent=None)[:600])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": str(Path(args.ckpt).resolve()), "split": args.split, "beam": args.beam,
        "results": results,
        "items": [
            {"image_id": int(s["image_id"]), "shuffle_image_id": id_map[int(s["image_id"])],
             "question": s["question"], "answer": s["answer"], "qtype": qtype(s), "rule": s.get("rule"),
             "gt": s["caption"], **{m: " ".join(h[i]) for m, h in hyps.items()}}
            for i, s in enumerate(samples)
        ],
    }, indent=1), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
