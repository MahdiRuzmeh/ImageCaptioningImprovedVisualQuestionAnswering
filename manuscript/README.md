# Manuscript: source, figures and evidence

The paper is written in **Markdown** (`paper.md`, equations in LaTeX) and kept in git. The Word file is built from it and is
**not** committed (Word files are binary: git can store them, but cannot show what changed between two versions).

| Path | Content |
|---|---|
| `paper.md` | manuscript source (the only file to edit for text changes) |
| `build.py` | builds `Paper/QD_Caption_VQA_Manuscript_v<version>.docx` with pandoc; styles from `Paper/reference_style.docx` if present |
| `figures/make_figures.py` | all figures: SVG (vector, for the journal) + PNG (300 dpi, for Word); PDF copies are built locally and not committed |
| `evidence/paper_evidence.md` | numbers behind the tables: data flow, cross-fitting proof, answer vocabulary, bootstrap, official seeds (built by `CaptionVQA/scripts/paper_evidence.py`) |
| `VERIFICATION.md` | the supervisor's author checklist answered against the code |
| `CHANGELOG.md` | what changed in each version |
| `CONTROLS_PROPOSAL.md` | the planned generic-caption and null-caption controls |

## Rebuild

```
pip install pypandoc_binary python-docx
python CaptionVQA/scripts/paper_evidence.py      # numbers (needs the local run outputs and dataset)
python manuscript/figures/make_figures.py         # figures
python manuscript/build.py                        # Word file in Paper/
```

## Versions

Each version sent to the supervisor gets three things:

1. the subtitle in `paper.md` ("Manuscript version N (date)") and an entry in `CHANGELOG.md`;
2. a commit, then a git tag: `git tag -a paper-vN -m "Manuscript version N"` and `git push origin paper-vN`;
3. the built file `Paper/QD_Caption_VQA_Manuscript_vN.docx` (sent by email, not committed).

To see what changed between two versions: `git diff paper-v1 paper-v2 -- manuscript/paper.md`
(or compare the two tags on GitHub). Supervisor comments made in Word are copied back into `paper.md` by hand or with
`pandoc -t markdown` on the commented file.
