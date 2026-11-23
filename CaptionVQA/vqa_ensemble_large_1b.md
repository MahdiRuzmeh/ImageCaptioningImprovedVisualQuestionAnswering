# VQA ensemble (test = 10,001 val2014 questions)

| Combination | Dev | **Test** | Yes/no | Number | Other |
|---|---:|---:|---:|---:|---:|
| single: vqa_b64_baseline/nocap_s0 | 66.58 | **66.62** | 83.74 | 48.64 | 58.65 |
| single: vqa_b64_baseline/nocap_s1 | 67.19 | **66.16** | 83.35 | 48.44 | 58.05 |
| single: vqa_large_drop/blip_s0 | 70.86 | **69.71** | 85.93 | 51.32 | 62.52 |
| single: vqa_large_drop/blip_s1 | 70.86 | **69.81** | 86.04 | 51.78 | 62.51 |
| single: vqa_b64_crossfit_drop/blip_s0 | 69.43 | **68.56** | 85.18 | 51.17 | 60.80 |
| single: vqa_b64_crossfit_drop/blip_s1 | 69.47 | **68.07** | 84.63 | 51.26 | 60.18 |
| single: vqa_b64_crossfit/blip_s0 | 69.59 | **67.99** | 83.71 | 49.65 | 61.17 |
| single: vqa_b64_crossfit/blip_s1 | 69.66 | **68.18** | 84.69 | 49.25 | 60.92 |
| baseline seeds averaged (2) | 67.55 | **67.27** | 84.39 | 49.77 | 59.15 |
| caption-model seeds averaged (6) | 71.56 | **70.21** | 85.94 | 52.29 | 63.27 |
| all 8 models, equal weight | 71.36 | **70.17** | 86.06 | 53.22 | 62.84 |
| weighted: 0.9 × caption models + 0.1 × baselines (w chosen on dev) | 71.60 | **70.24** | 86.01 | 52.86 | 63.12 |
