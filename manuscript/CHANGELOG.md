# Manuscript changelog

## Version 2 — 2026-11-26

- Keywords replaced: Visual question answering; Image captioning; Image captioning improved visual question answering;
  Question-dependent image captioning.
- Abstract and Introduction state the three contributions: the question-dependent caption dataset and task, the
  BLIP-Large question-dependent captioner, and the question-dependent-caption-improved VQA model.
- Section 3 rewritten as the dataset generation pipeline in four steps (classification, generation, validation,
  revalidation with Qwen2.5-7B), each with its rules, prompts and counts; Table 1 and Fig. 1 (flow chart) redrawn.
- Tables: thin grid borders and a shaded header row (pandoc's default table style has none).

## Version 1 — 2026-11-26

Starting point: the supervisor's draft "QD_Caption_VQA_Manuscript_Revised_V2.docx" (converted with pandoc; text, structure and
references kept unless listed here).

**New results**
- Single-seed test-dev results for all four final models (Table 5, abstract, Fig. 6b).
- Image-clustered bootstrap intervals (Section 5.2, Tables 7–8).
- Exact corpus data flow from the corpus provenance block (Table 1, Fig. 1).
- Cross-fitting verification: selection data, 0 leaked captions, test2015 disjointness (Section 3.3, Table 2).
- Blank Table 11 for the generic-caption and null-caption controls (Section 6.5).

**Corrections**
- Qwen models: generation, classification and re-classification `qwen2.5:3b-instruct-q4_K_M`; screening and repair
  `qwen2.5:7b-instruct-q4_K_M` (was "Qwen-7B" for everything).
- Table 1: GRAMMAR_ISSUE_ONLY and WRONG records are not in the training corpus; initial 4,567 vs final 4,712 explained;
  29,844 `ok` records with the heuristic suspicious flag reported.
- Eq. (7): mean over unmasked caption positions; prompt/padding masked. Eq. (8): beam score = length-normalised log-probability.
- Eq. (9)–(12): type embeddings on regions and caption tokens; decoder order SA → GA → FFN with post-LN; AttFlat pooling.
- Eq. (16): BCE summed over answers and averaged over the batch. Eq. (17): sigmoid probabilities.
- AnswerHit and polarity defined exactly (Eqs. 20–21); answer normalisation listed.
- Table 3 (data splits) added; configuration table completed (optimiser details, parameters).
- "Diagnostic set" renamed **development set**; the development → official shift explained by 15,001 extra training questions,
  3,148 vs 3,091 answers and a different question sample.
- Sharma & Jalal values marked as taken from the original Table 6.
- Table 7: "MCAN, GloVe 6B" row uses batch 128.
- Table 10: 10,000 vs 10,001 explained.
- References [11] Qwen2.5, [12] Visual Genome, [13] COCO Captions added.
- Author checklist replaced by a status list (resolved / open).
- Figures redrawn as vector graphics from the result files.
