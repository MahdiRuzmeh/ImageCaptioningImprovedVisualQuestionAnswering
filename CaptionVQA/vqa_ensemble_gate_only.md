# VQA ensemble (test = 10,001 val2014 questions)

| Combination | Dev | **Test** | Yes/no | Number | Other |
|---|---:|---:|---:|---:|---:|
| single: vqa_strong_baseline/nocap_s0 | 66.48 | **65.47** | 82.20 | 47.87 | 57.67 |
| single: vqa_strong_baseline/nocap_s1 | 65.93 | **66.01** | 83.25 | 47.67 | 58.04 |
| single: vqa_gate_aux/blip_s0 | 69.41 | **68.09** | 84.89 | 50.59 | 60.22 |
| single: vqa_gate_aux/blip_s1 | 69.11 | **68.16** | 84.42 | 51.02 | 60.61 |
| single: vqa_gate_noaux/blip_s0 | 69.44 | **68.50** | 84.44 | 51.55 | 61.14 |
| single: vqa_gate_noaux/blip_s1 | 69.74 | **68.56** | 84.83 | 51.00 | 61.10 |
| baseline seeds averaged (2) | 66.80 | **66.82** | 83.70 | 48.87 | 59.02 |
| caption-model seeds averaged (4) | 70.25 | **69.31** | 85.46 | 52.55 | 61.73 |
| all 6 models, equal weight | 69.96 | **69.41** | 85.90 | 52.69 | 61.55 |
| weighted: 0.9 × caption models + 0.1 × baselines (w chosen on dev) | 70.31 | **69.43** | 85.52 | 52.61 | 61.91 |
