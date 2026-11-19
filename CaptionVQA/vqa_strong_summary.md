# VQA: baseline vs + BLIP captions

Train 643,110 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (no caption) | 2 | **65.74 ± 0.31** | 82.70 ± 0.67 | 47.76 ± 0.40 | 57.88 ± 0.23 | — | — |
| + BLIP captions | 2 | **67.80 ± 0.06** | 83.45 ± 0.23 | 48.93 ± 0.09 | 61.17 ± 0.32 | 60.21 ± 0.31 | — |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

**Gain of BLIP captions over the baseline** (mean): overall +2.06, yes/no +0.75, number +1.17, other +3.30

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | Baseline (no caption) | + BLIP captions | BLIP − baseline |
|---|---:|---:|---:|---:|
| how many | 977 | 53.3 | 54.7 | +1.4 |
| is the | 811 | 81.9 | 81.8 | -0.1 |
| what | 759 | 48.4 | 51.8 | +3.4 |
| what color is the | 638 | 79.1 | 84.6 | +5.5 |
| what is the | 551 | 53.4 | 57.2 | +3.8 |
| none of the above | 390 | 61.5 | 61.1 | -0.4 |
| is this a | 355 | 82.9 | 85.4 | +2.4 |
| is this | 351 | 86.5 | 85.8 | -0.7 |
| what is | 293 | 45.2 | 46.9 | +1.7 |
| what kind of | 270 | 59.7 | 60.7 | +1.0 |
| are the | 258 | 78.9 | 79.7 | +0.8 |
| is there a | 229 | 81.6 | 84.0 | +2.4 |
| what type of | 181 | 55.4 | 60.7 | +5.4 |
| where is the | 166 | 38.0 | 44.2 | +6.2 |
| what color are the | 160 | 72.5 | 75.7 | +3.2 |
| is it | 159 | 90.7 | 93.8 | +3.1 |
| what are the | 153 | 56.5 | 59.9 | +3.4 |
| are there | 142 | 81.4 | 81.7 | +0.3 |
| is | 141 | 79.6 | 81.5 | +1.9 |
| is there | 141 | 84.6 | 85.5 | +0.9 |
| does the | 136 | 81.2 | 80.9 | -0.3 |
| are these | 127 | 82.4 | 83.0 | +0.5 |
| what is on the | 115 | 49.0 | 51.2 | +2.2 |
| is the man | 113 | 83.9 | 84.2 | +0.3 |
| what is the man | 112 | 65.2 | 68.1 | +2.9 |
| how | 111 | 34.8 | 36.2 | +1.4 |
| are | 110 | 76.2 | 77.3 | +1.1 |
| which | 108 | 49.5 | 47.9 | -1.6 |
| does this | 106 | 76.7 | 75.8 | -0.9 |
| how many people are | 102 | 59.7 | 57.1 | -2.6 |
| what is in the | 85 | 60.0 | 59.7 | -0.3 |
| what is this | 76 | 64.0 | 72.5 | +8.5 |
| what are | 74 | 60.2 | 60.6 | +0.4 |
| do | 73 | 78.9 | 77.9 | -1.0 |
| what does the | 72 | 22.8 | 24.2 | +1.4 |
| what color | 68 | 74.5 | 79.2 | +4.7 |
| what time | 68 | 24.7 | 26.8 | +2.1 |
| why | 66 | 23.3 | 24.7 | +1.4 |
| where are the | 65 | 38.5 | 46.8 | +8.3 |
| what color is | 61 | 77.6 | 82.3 | +4.7 |
| are there any | 59 | 85.6 | 89.3 | +3.7 |
| how many people are in | 59 | 63.9 | 68.8 | +4.9 |
| are they | 57 | 76.3 | 82.5 | +6.2 |
| who is | 51 | 47.9 | 46.6 | -1.4 |
| can you | 48 | 71.5 | 67.9 | -3.5 |
| is this an | 46 | 81.6 | 85.1 | +3.5 |
| what brand | 46 | 39.3 | 44.2 | +4.9 |
| what is the person | 45 | 62.7 | 74.0 | +11.3 |
| what sport is | 43 | 92.0 | 96.3 | +4.3 |
| is he | 42 | 89.5 | 88.2 | -1.3 |
| what animal is | 41 | 85.4 | 84.6 | -0.7 |
| what room is | 41 | 96.6 | 94.4 | -2.2 |
| has | 40 | 83.8 | 77.9 | -5.9 |
| what is the woman | 40 | 58.8 | 70.0 | +11.2 |
| is the woman | 39 | 87.2 | 81.2 | -6.0 |
| was | 39 | 75.8 | 87.3 | +11.5 |
| what is the color of the | 39 | 85.8 | 87.4 | +1.7 |
| what number is | 38 | 8.3 | 6.2 | -2.1 |
| what is the name | 37 | 12.2 | 17.2 | +5.0 |
| do you | 35 | 72.4 | 72.6 | +0.1 |
| is that a | 35 | 80.9 | 85.9 | +5.0 |
| is the person | 33 | 75.8 | 87.0 | +11.2 |
| is this person | 30 | 74.8 | 73.7 | -1.2 |
| could | 27 | 74.6 | 79.6 | +5.0 |
| why is the | 18 | 28.9 | 28.9 | +0.0 |
