"""CAPTIONER_COMPARISON.md → styled HTML → PDF (headless Chrome / Edge), no extra Python packages.

Finglish: converter-e kochik baraye hamoon element haye report (heading, paragraph, table, list,
code block, image, bold/italic/code, ---). Image ha base64 dakhel-e HTML → yek file. Chrome
``--print-to-pdf`` → A4.

    python scripts/build_report_pdf.py                       # → CAPTIONER_COMPARISON.pdf
    python scripts/build_report_pdf.py --md X.md --out X.pdf
"""

from __future__ import annotations

import argparse
import base64
import html
import re
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "msedge",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
:root { --ink:#0b0b0b; --ink2:#52514e; --rule:#dcdad4; --soft:#f3f2ee; --accent:#2a78d6; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 9.6pt; line-height: 1.45;
       color: var(--ink); background: #fff; margin: 0; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 4pt; }
.meta { color: var(--ink2); font-size: 9pt; margin-bottom: 12pt; border-bottom: 2px solid var(--accent); padding-bottom: 8pt; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid var(--rule);
     break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 5pt; break-after: avoid; }
p { margin: 4pt 0 6pt; }
ul, ol { margin: 3pt 0 7pt; padding-left: 16pt; }
li { margin: 2pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.6pt; break-inside: avoid; }
th, td { border-bottom: 1px solid var(--rule); padding: 3.5pt 6pt; vertical-align: top; text-align: left; }
th { background: var(--soft); font-weight: 600; border-bottom: 1.5px solid #b9b7b0; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr:nth-child(even) td { background: #fbfbf9; }
pre { background: var(--soft); border: 1px solid var(--rule); border-radius: 4px; padding: 7pt 9pt;
      font-family: Consolas, "Courier New", monospace; font-size: 7.6pt; line-height: 1.35; white-space: pre;
      overflow: hidden; break-inside: avoid; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.4pt; background: var(--soft);
       padding: 0 2.5pt; border-radius: 3px; }
pre code { background: none; padding: 0; font-size: inherit; }
figure { margin: 8pt 0 12pt; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; }
figcaption { color: var(--ink2); font-size: 8.4pt; margin-top: 3pt; }
hr { border: none; border-top: 1px solid var(--rule); margin: 14pt 0; }
strong { font-weight: 650; }
h2 + p, h3 + p, h3 + table, h2 + table { break-before: avoid; }
"""


def inline(s: str) -> str:
    """Finglish: inline markdown → HTML (code, bold, italic, link); aval escape."""
    parts = re.split(r"(`[^`]+`)", s)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(f"<code>{html.escape(p[1:-1])}</code>")
            continue
        t = html.escape(p, quote=False)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
        out.append(t)
    return "".join(out)


NUM = re.compile(r"^[\s*]*[-+−]?[\d.,]+\s*(%|M|pts.*|×.*)?[\s*]*$|^—$|^[\d.,]+%? \(.+\)$")


def table(lines: list[str]) -> str:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    align = rows[1]
    right = [a.strip().endswith(":") for a in align]
    head, body = rows[0], rows[2:]
    h = "".join(f'<th class="{"num" if right[i] else ""}">{inline(c)}</th>' for i, c in enumerate(head))
    b = []
    for r in body:
        cells = []
        for i, c in enumerate(r):
            cls = "num" if (i < len(right) and right[i]) else ""
            cells.append(f'<td class="{cls}">{inline(c)}</td>')
        b.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(b)}</tbody></table>"


def lists(lines: list[str]) -> str:
    """Finglish: list-e tu dar tu (2 space indent), - / 1. ."""
    out, stack = [], []  # stack of (indent, tag)
    for ln in lines:
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if not m:  # continuation line
            if out:
                out[-1] = out[-1][:-5] + " " + inline(ln.strip()) + "</li>"
            continue
        ind, tag = len(m.group(1)), ("ol" if m.group(2)[0].isdigit() else "ul")
        while stack and stack[-1][0] > ind:
            out.append(f"</{stack.pop()[1]}>")
        if not stack or stack[-1][0] < ind:
            if stack and out and out[-1].endswith("</li>"):
                out[-1] = out[-1][:-5]  # nest inside previous li
                stack.append((ind, tag))
                out.append(f"<{tag}>")
            else:
                stack.append((ind, tag))
                out.append(f"<{tag}>")
        out.append(f"<li>{inline(m.group(3))}</li>")
    while stack:
        ind, tag = stack.pop()
        out.append(f"</{tag}>")
        if stack:
            out.append("</li>")
    return "".join(out)


def convert(md: str, base: Path, subtitle: str = "Stage 1 report") -> str:
    lines = md.splitlines()
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", ln)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
            if n == 1:
                out.append(f'<div class="meta">Master thesis: image captioning improved visual question answering · '
                           f'{html.escape(subtitle)} · {date.today().strftime("%d %B %Y")}</div>')
            i += 1
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", ln)
        if m:
            p = (base / m.group(2)).resolve()
            data = base64.b64encode(p.read_bytes()).decode()
            out.append(f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(m.group(1))}">'
                       f"<figcaption>{inline(m.group(1))}</figcaption></figure>")
            i += 1
            continue
        if ln.strip() == "---":
            out.append("<hr>")
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            out.append(table(lines[i:j]))
            i = j
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", ln):
            j = i
            while j < len(lines) and lines[j].strip() and (re.match(r"^\s*([-*]|\d+\.)\s+", lines[j])
                                                         or lines[j].startswith("  ")):
                j += 1
            out.append(lists(lines[i:j]))
            i = j
            continue
        if not ln.strip():
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].strip() and not re.match(r"^(#|\||```|!\[|\s*([-*]|\d+\.)\s|---)", lines[j]):
            j += 1
        out.append("<p>" + inline(" ".join(x.strip() for x in lines[i:j])) + "</p>")
        i = max(j, i + 1)
    return "\n".join(out)


def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    raise SystemExit("Chrome / Edge not found")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--md", default=str(ROOT / "CAPTIONER_COMPARISON.md"))
    p.add_argument("--out", default=None)
    p.add_argument("--subtitle", default="Stage 1 report", help="text in the line under the title")
    args = p.parse_args()
    md = Path(args.md)
    out = Path(args.out) if args.out else md.with_suffix(".pdf")
    title = re.search(r"^#\s+(.*)$", md.read_text(encoding="utf-8"), re.M).group(1)
    body = convert(md.read_text(encoding="utf-8"), md.parent, args.subtitle)
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    html_path = md.with_suffix(".html")
    html_path.write_text(page, encoding="utf-8")
    with tempfile.TemporaryDirectory() as prof:
        cmd = [browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={prof}",
               f"--print-to-pdf={out.resolve()}", html_path.resolve().as_uri()]
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    print(f"HTML -> {html_path}\nPDF  -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
