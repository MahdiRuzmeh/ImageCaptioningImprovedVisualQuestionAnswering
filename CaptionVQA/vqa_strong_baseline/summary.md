# VQA: baseline vs + BLIP captions

Train 643,110 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (no caption) | 2 | **65.74 ± 0.31** | 82.70 ± 0.67 | 47.76 ± 0.40 | 57.88 ± 0.23 | — | — |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | Baseline (no caption) |
|---|---:|---:|
| how many | 977 | 53.3 |
| is the | 811 | 81.9 |
| what | 759 | 48.4 |
| what color is the | 638 | 79.1 |
| what is the | 551 | 53.4 |
| none of the above | 390 | 61.5 |
| is this a | 355 | 82.9 |
| is this | 351 | 86.5 |
| what is | 293 | 45.2 |
| what kind of | 270 | 59.7 |
| are the | 258 | 78.9 |
| is there a | 229 | 81.6 |
| what type of | 181 | 55.4 |
| where is the | 166 | 38.0 |
| what color are the | 160 | 72.5 |
| is it | 159 | 90.7 |
| what are the | 153 | 56.5 |
| are there | 142 | 81.4 |
| is | 141 | 79.6 |
| is there | 141 | 84.6 |
| does the | 136 | 81.2 |
| are these | 127 | 82.4 |
| what is on the | 115 | 49.0 |
| is the man | 113 | 83.9 |
| what is the man | 112 | 65.2 |
| how | 111 | 34.8 |
| are | 110 | 76.2 |
| which | 108 | 49.5 |
| does this | 106 | 76.7 |
| how many people are | 102 | 59.7 |
| what is in the | 85 | 60.0 |
| what is this | 76 | 64.0 |
| what are | 74 | 60.2 |
| do | 73 | 78.9 |
| what does the | 72 | 22.8 |
| what color | 68 | 74.5 |
| what time | 68 | 24.7 |
| why | 66 | 23.3 |
| where are the | 65 | 38.5 |
| what color is | 61 | 77.6 |
| are there any | 59 | 85.6 |
| how many people are in | 59 | 63.9 |
| are they | 57 | 76.3 |
| who is | 51 | 47.9 |
| can you | 48 | 71.5 |
| is this an | 46 | 81.6 |
| what brand | 46 | 39.3 |
| what is the person | 45 | 62.7 |
| what sport is | 43 | 92.0 |
| is he | 42 | 89.5 |
| what animal is | 41 | 85.4 |
| what room is | 41 | 96.6 |
| has | 40 | 83.8 |
| what is the woman | 40 | 58.8 |
| is the woman | 39 | 87.2 |
| was | 39 | 75.8 |
| what is the color of the | 39 | 85.8 |
| what number is | 38 | 8.3 |
| what is the name | 37 | 12.2 |
| do you | 35 | 72.4 |
| is that a | 35 | 80.9 |
| is the person | 33 | 75.8 |
| is this person | 30 | 74.8 |
| could | 27 | 74.6 |
| why is the | 18 | 28.9 |
