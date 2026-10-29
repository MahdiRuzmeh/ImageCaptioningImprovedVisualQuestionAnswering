"""Score enrich_captions.py outputs: how many rows were enriched, how much they add.

    python audit/score_enrichment.py outputs/enrich_pilot_qwen3b.json outputs/enrich_pilot_mistral.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validation.tokens import content_words  # noqa: E402

_NUM = re.compile(r"^\d+$")


def qtype(row: dict) -> str:
    a = row["answer"].strip().lower()
    if a in {"yes", "no"}:
        return "yes/no"
    if _NUM.match(a) or row["rule"] == "how_many":
        return "number"
    if row["rule"] == "what_color" or row["question"].lower().startswith("what color"):
        return "color"
    return "other"


def score(path: Path) -> dict:
    data = json.load(open(path, encoding="utf-8"))
    rows = data["annotations"]
    out = {"file": path.name, "model": data["info"]["enrich"]["model"], "n": len(rows)}
    by_type = Counter(qtype(r) for r in rows)
    enr = [r for r in rows if r["enrich_status"] == "enriched"]
    out["status"] = dict(Counter(r["enrich_status"] for r in rows))
    out["enriched_rate"] = len(enr) / len(rows)
    out["enriched_rate_by_type"] = {
        t: sum(qtype(r) == t for r in enr) / n for t, n in sorted(by_type.items())
    }
    out["added_words_mean"] = mean(len(r["enrich_added"]) for r in enr) if enr else 0.0
    out["len_words"] = {
        "base": mean(len(r["caption_base"].split()) for r in rows),
        "final": mean(len(r["caption"].split()) for r in rows),
    }
    # Share of final caption content that is not in the question/answer (the new information)
    def novelty(r: dict) -> float:
        cap = content_words(r["caption"])
        qa = content_words(r["question"]) | content_words(r["answer"])
        return len(cap - qa) / max(len(cap), 1)

    out["novel_content_share"] = {
        "base": mean(novelty({**r, "caption": r["caption_base"]}) for r in rows),
        "final": mean(novelty(r) for r in rows),
    }
    rejects = Counter()
    for r in rows:
        for att in r.get("enrich_rejects", []):
            for reason in att:
                rejects[reason.split(":")[0]] += 1
    out["reject_reasons"] = dict(rejects.most_common())
    return out


def main() -> int:
    for p in sys.argv[1:]:
        print(json.dumps(score(Path(p)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
