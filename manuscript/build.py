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
    style_tables(out)
    print("->", out)


def style_tables(path: Path) -> None:
    """Pandoc's default "Table" style has no borders: add thin grid lines and the draft's shaded, bold header row."""
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = Document(str(path))
    for table in doc.tables:
        tbl_pr = table._tbl.tblPr
        for old in tbl_pr.findall(qn("w:tblBorders")):
            tbl_pr.remove(old)
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "8C8C8C")
            borders.append(el)
        tbl_pr.append(borders)
        for cell in table.rows[0].cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "DCE8F0")
            tc_pr.append(shd)
            for par in cell.paragraphs:
                for run in par.runs:
                    run.bold = True
    doc.save(str(path))


if __name__ == "__main__":
    main()
