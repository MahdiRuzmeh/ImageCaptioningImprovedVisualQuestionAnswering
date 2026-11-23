# VQA ensemble (test = 10,001 val2014 questions)

| Combination | Dev | **Test** | Yes/no | Number | Other |
|---|---:|---:|---:|---:|---:|
| single: vqa_b64_baseline/nocap_s0 | 66.58 | **66.62** | 83.74 | 48.64 | 58.65 |
| single: vqa_b64_baseline/nocap_s1 | 67.19 | **66.16** | 83.35 | 48.44 | 58.05 |
| single: vqa_large_drop/blip_s0 | 70.86 | **69.71** | 85.93 | 51.32 | 62.52 |
| single: vqa_large_drop/blip_s1 | 70.86 | **69.81** | 86.04 | 51.78 | 62.51 |
| baseline seeds averaged (2) | 67.55 | **67.27** | 84.39 | 49.77 | 59.15 |
| caption-model seeds averaged (2) | 71.72 | **70.27** | 86.23 | 52.73 | 63.04 |
| all 4 models, equal weight | 70.63 | **69.85** | 85.78 | 53.15 | 62.42 |
| weighted: 1.0 × caption models + 0.0 × baselines (w chosen on dev) | 71.72 | **70.27** | 86.23 | 52.73 | 63.04 |
