# VQA ensemble (test = 10,001 val2014 questions)

| Combination | Dev | **Test** | Yes/no | Number | Other |
|---|---:|---:|---:|---:|---:|
| single: vqa_strong_baseline/nocap_s0 | 66.48 | **65.47** | 82.20 | 47.87 | 57.67 |
| single: vqa_strong_baseline/nocap_s1 | 65.93 | **66.01** | 83.25 | 47.67 | 58.04 |
| single: vqa_large_drop/blip_s0 | 70.86 | **69.71** | 85.93 | 51.32 | 62.52 |
| single: vqa_large_drop/blip_s1 | 70.86 | **69.81** | 86.04 | 51.78 | 62.51 |
| single: vqa_b64_crossfit_drop/blip_s0 | 69.43 | **68.56** | 85.18 | 51.17 | 60.80 |
| single: vqa_b64_crossfit_drop/blip_s1 | 69.47 | **68.07** | 84.63 | 51.26 | 60.18 |
| single: vqa_b64_crossfit/blip_s0 | 69.59 | **67.99** | 83.71 | 49.65 | 61.17 |
| single: vqa_b64_crossfit/blip_s1 | 69.66 | **68.18** | 84.69 | 49.25 | 60.92 |
| single: vqa_gate_aux/blip_s0 | 69.41 | **68.09** | 84.89 | 50.59 | 60.22 |
| single: vqa_gate_aux/blip_s1 | 69.11 | **68.16** | 84.42 | 51.02 | 60.61 |
| single: vqa_gate_noaux/blip_s0 | 69.44 | **68.50** | 84.44 | 51.55 | 61.14 |
| single: vqa_gate_noaux/blip_s1 | 69.74 | **68.56** | 84.83 | 51.00 | 61.10 |
| baseline seeds averaged (2) | 66.80 | **66.82** | 83.70 | 48.87 | 59.02 |
| caption-model seeds averaged (10) | 71.07 | **70.07** | 85.84 | 52.78 | 62.92 |
| all 12 models, equal weight | 70.86 | **70.07** | 85.99 | 53.28 | 62.68 |
| weighted: 1.0 × caption models + 0.0 × baselines (w chosen on dev) | 71.07 | **70.07** | 85.84 | 52.78 | 62.92 |
