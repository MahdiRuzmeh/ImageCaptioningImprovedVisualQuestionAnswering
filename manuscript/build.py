"""Build the Word manuscript from manuscript/paper.md (pandoc).

The Markdown file is the source under version control; the .docx is a build product and is not committed.
Styles come from a reference .docx (default: Paper/reference_style.docx, a copy of the supervisor's draft) when present.

    pip install pypandoc_binary          # bundles pandoc
    python manuscript/build.py           # -> Paper/QD_Caption_VQA_Manuscript_v<version>.docx
    python manuscript/build.py --version 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pypandoc

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--version", help="version label; default: from the subtitle of paper.md")
    p.add_argument("--reference", default=str(REPO / "Paper" / "reference_style.docx"))
    p.add_argument("--out-dir", default=str(REPO / "Paper"))
    args = p.parse_args()
    src = HERE / "paper.md"
    version = args.version or re.search(r"version (\S+)", src.read_text(encoding="utf-8")).group(1)
    out = Path(args.out_dir) / f"QD_Caption_VQA_Manuscript_v{version}.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = ["--resource-path", str(HERE), "--wrap=none"]
    if Path(args.reference).exists():
        extra += ["--reference-doc", args.reference]
    pypandoc.convert_file(str(src), "docx", format="markdown", outputfile=str(out), extra_args=extra)
    print("->", out)


if __name__ == "__main__":
    main()
