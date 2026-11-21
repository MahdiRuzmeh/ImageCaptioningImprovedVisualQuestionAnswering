# VQA ensemble (test = 10,001 val2014 questions)

| Combination | Dev | **Test** | Yes/no | Number | Other |
|---|---:|---:|---:|---:|---:|
| single: vqa_strong_baseline/nocap_s0 | 66.48 | **65.47** | 82.20 | 47.87 | 57.67 |
| single: vqa_strong_baseline/nocap_s1 | 65.93 | **66.01** | 83.25 | 47.67 | 58.04 |
| single: vqa_gate_aux/blip_s0 | 69.41 | **68.09** | 84.89 | 50.59 | 60.22 |
| single: vqa_gate_aux/blip_s1 | 69.11 | **68.16** | 84.42 | 51.02 | 60.61 |
| single: vqa_b64_crossfit_drop/blip_s0 | 69.43 | **68.56** | 85.18 | 51.17 | 60.80 |
| single: vqa_b64_crossfit_drop/blip_s1 | 69.47 | **68.07** | 84.63 | 51.26 | 60.18 |
| baseline seeds averaged (2) | 66.80 | **66.82** | 83.70 | 48.87 | 59.02 |
| caption-model seeds averaged (4) | 70.43 | **69.22** | 85.60 | 51.56 | 61.70 |
| all 6 models, equal weight | 70.14 | **69.32** | 85.73 | 52.19 | 61.64 |
| weighted: 0.9 × caption models + 0.1 × baselines (w chosen on dev) | 70.45 | **69.30** | 85.69 | 51.94 | 61.70 |
