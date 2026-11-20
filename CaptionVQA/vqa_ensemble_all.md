# VQA ensemble (test = 10,001 val2014 questions)

| Combination | Dev | **Test** | Yes/no | Number | Other |
|---|---:|---:|---:|---:|---:|
| single: vqa_strong_baseline/nocap_s0 | 66.48 | **65.47** | 82.20 | 47.87 | 57.67 |
| single: vqa_strong_baseline/nocap_s1 | 65.93 | **66.01** | 83.25 | 47.67 | 58.04 |
| single: vqa_strong_crossfit/blip_s0 | 68.96 | **67.71** | 83.53 | 49.11 | 60.89 |
| single: vqa_strong_crossfit/blip_s1 | 69.26 | **67.81** | 83.18 | 49.07 | 61.36 |
| single: vqa_b64_crossfit_drop/blip_s0 | 69.43 | **68.56** | 85.18 | 51.17 | 60.80 |
| single: vqa_b64_crossfit_drop/blip_s1 | 69.47 | **68.07** | 84.63 | 51.26 | 60.18 |
| single: vqa_b64_crossfit/blip_s0 | 69.59 | **67.99** | 83.71 | 49.65 | 61.17 |
| single: vqa_b64_crossfit/blip_s1 | 69.66 | **68.18** | 84.69 | 49.25 | 60.92 |
| baseline seeds averaged (2) | 66.80 | **66.82** | 83.70 | 48.87 | 59.02 |
| caption-model seeds averaged (6) | 70.82 | **69.36** | 85.02 | 51.36 | 62.48 |
| all 8 models, equal weight | 70.66 | **69.48** | 85.39 | 51.88 | 62.32 |
| weighted: 0.9 × caption models + 0.1 × baselines (w chosen on dev) | 70.83 | **69.49** | 85.28 | 51.48 | 62.52 |
