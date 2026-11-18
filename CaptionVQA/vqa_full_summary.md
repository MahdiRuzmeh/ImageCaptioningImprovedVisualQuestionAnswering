# VQA: baseline vs + BLIP captions

Train 443,757 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (no caption) | 3 | **60.03 ± 0.45** | 77.59 ± 0.98 | 40.47 ± 0.40 | 52.15 ± 0.29 | — | — |
| + BLIP captions | 3 | **66.08 ± 0.20** | 82.30 ± 0.41 | 48.11 ± 1.01 | 58.77 ± 0.13 | 47.20 ± 0.47 | 0.55 ± 0.00 |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

**Gain of BLIP captions over the baseline** (mean): overall +6.05, yes/no +4.72, number +7.64, other +6.62

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | Baseline (no caption) | + BLIP captions | BLIP − baseline |
|---|---:|---:|---:|---:|
| how many | 977 | 45.3 | 53.9 | +8.6 |
| is the | 811 | 74.8 | 81.0 | +6.2 |
| what | 759 | 45.7 | 51.1 | +5.4 |
| what color is the | 638 | 73.3 | 81.5 | +8.1 |
| what is the | 551 | 46.8 | 53.6 | +6.8 |
| none of the above | 390 | 55.2 | 56.9 | +1.7 |
| is this a | 355 | 76.9 | 83.3 | +6.4 |
| is this | 351 | 78.4 | 86.6 | +8.1 |
| what is | 293 | 34.7 | 44.7 | +9.9 |
| what kind of | 270 | 53.4 | 58.4 | +5.0 |
| are the | 258 | 74.1 | 78.0 | +3.9 |
| is there a | 229 | 75.6 | 84.7 | +9.1 |
| what type of | 181 | 50.5 | 56.7 | +6.2 |
| where is the | 166 | 33.2 | 41.3 | +8.1 |
| what color are the | 160 | 66.8 | 71.4 | +4.7 |
| is it | 159 | 90.7 | 91.3 | +0.6 |
| what are the | 153 | 46.8 | 56.5 | +9.7 |
| are there | 142 | 79.1 | 83.9 | +4.8 |
| is | 141 | 79.9 | 77.4 | -2.5 |
| is there | 141 | 76.6 | 82.1 | +5.5 |
| does the | 136 | 79.8 | 80.6 | +0.8 |
| are these | 127 | 78.3 | 78.1 | -0.2 |
| what is on the | 115 | 39.6 | 51.6 | +12.0 |
| is the man | 113 | 73.6 | 78.2 | +4.6 |
| what is the man | 112 | 61.5 | 69.2 | +7.7 |
| how | 111 | 29.5 | 31.8 | +2.3 |
| are | 110 | 74.3 | 78.7 | +4.4 |
| which | 108 | 43.6 | 43.3 | -0.4 |
| does this | 106 | 77.2 | 71.8 | -5.3 |
| how many people are | 102 | 52.0 | 58.8 | +6.8 |
| what is in the | 85 | 50.0 | 57.9 | +7.8 |
| what is this | 76 | 63.9 | 72.6 | +8.8 |
| what are | 74 | 58.1 | 67.4 | +9.3 |
| do | 73 | 77.8 | 73.8 | -3.9 |
| what does the | 72 | 21.6 | 26.2 | +4.7 |
| what color | 68 | 71.9 | 75.0 | +3.1 |
| what time | 68 | 24.3 | 26.2 | +2.0 |
| why | 66 | 12.7 | 16.2 | +3.5 |
| where are the | 65 | 34.6 | 45.5 | +10.9 |
| what color is | 61 | 76.3 | 86.6 | +10.2 |
| are there any | 59 | 74.9 | 87.9 | +13.1 |
| how many people are in | 59 | 49.1 | 64.0 | +14.9 |
| are they | 57 | 74.2 | 80.8 | +6.6 |
| who is | 51 | 29.9 | 34.1 | +4.2 |
| can you | 48 | 59.6 | 77.2 | +17.6 |
| is this an | 46 | 74.5 | 84.9 | +10.4 |
| what brand | 46 | 39.5 | 47.4 | +7.9 |
| what is the person | 45 | 59.9 | 69.9 | +10.1 |
| what sport is | 43 | 88.8 | 94.2 | +5.3 |
| is he | 42 | 85.6 | 89.0 | +3.3 |
| what animal is | 41 | 80.0 | 83.6 | +3.6 |
| what room is | 41 | 95.3 | 94.1 | -1.1 |
| has | 40 | 73.8 | 72.8 | -1.0 |
| what is the woman | 40 | 54.8 | 72.9 | +18.2 |
| is the woman | 39 | 71.5 | 81.8 | +10.3 |
| was | 39 | 75.3 | 82.8 | +7.5 |
| what is the color of the | 39 | 82.1 | 83.0 | +0.9 |
| what number is | 38 | 6.3 | 8.9 | +2.6 |
| what is the name | 37 | 9.4 | 16.5 | +7.1 |
| do you | 35 | 75.6 | 72.1 | -3.5 |
| is that a | 35 | 74.7 | 81.6 | +7.0 |
| is the person | 33 | 78.2 | 85.1 | +6.9 |
| is this person | 30 | 62.8 | 70.4 | +7.7 |
| could | 27 | 76.4 | 83.0 | +6.5 |
| why is the | 18 | 23.9 | 24.4 | +0.6 |
