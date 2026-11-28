# Internal checklist (not part of the manuscript)

Resolved in this version (evidence: `manuscript/VERIFICATION.md`, `manuscript/evidence/paper_evidence.md`):

- Ensemble = mean of the two seeds' sigmoid scores (Eq. 17); answer lists identical between the final models (3,148).
- Cross-fitting: fold disjointness, selection data inside the own fold, 0 questions captioned by a model trained on their image, no val2014/test2015 image in the corpus, answer vocabulary rebuilt from training questions only.
- Corpus counts reconciled exactly from the corpus provenance block; Qwen model IDs and prompt versions stated; GRAMMAR_ISSUE_ONLY and WRONG records are not used; suspicious-flag overlap reported.
- Single-seed test-dev results added (Table 5).
- Sharma & Jalal values confirmed from the original Table 6.
- 10,000 vs. 10,001 explained (Table 10).
- Image-clustered bootstrap intervals (Tables 7–8); loss masking, label smoothing, beam score, decoder order, BCE normalisation and ensemble domain checked against the code.

Still open:

- Archive the dated EvalAI screenshots and leaderboard observation with the result files.
- Decide whether the full prompt texts (classifier v20, generation v10, validator v10, judge j2, regeneration r2) go into a supplementary appendix; they are in the repository (`QuestionDependentCaptionGenerator/`).
- Small blinded human audit of a random corpus sample.
- Generic-caption and fixed-caption controls (Table 11), after supervisor approval.
- Final author list, affiliations, repository URL.
