# VQA ensemble (test = 10,001 val2014 questions)

| Combination | Dev | **Test** | Yes/no | Number | Other |
|---|---:|---:|---:|---:|---:|
| single: nocap_s0 | 66.48 | **65.47** | 82.20 | 47.87 | 57.67 |
| single: nocap_s1 | 65.93 | **66.01** | 83.25 | 47.67 | 58.04 |
| single: blip_s0 | 68.96 | **67.71** | 83.53 | 49.11 | 60.89 |
| single: blip_s1 | 69.26 | **67.81** | 83.18 | 49.07 | 61.36 |
| baseline seeds averaged (2) | 66.80 | **66.82** | 83.70 | 48.87 | 59.02 |
| caption-model seeds averaged (2) | 69.64 | **68.57** | 84.02 | 50.46 | 61.89 |
| all 4 models, equal weight | 69.76 | **68.99** | 84.75 | 52.20 | 61.70 |
| weighted: 0.7 × caption models + 0.3 × baselines (w chosen on dev) | 70.26 | **69.11** | 84.69 | 51.77 | 62.11 |
