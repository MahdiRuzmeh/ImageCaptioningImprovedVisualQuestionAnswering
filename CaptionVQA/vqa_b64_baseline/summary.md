# VQA: baseline vs + BLIP captions

Train 643,110 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (no caption) | 2 | **66.39 ± 0.33** | 83.54 ± 0.28 | 48.54 ± 0.14 | 58.35 ± 0.42 | — | — |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | Baseline (no caption) |
|---|---:|---:|
| how many | 977 | 54.0 |
| is the | 811 | 81.9 |
| what | 759 | 49.3 |
| what color is the | 638 | 79.9 |
| what is the | 551 | 53.9 |
| none of the above | 390 | 60.8 |
| is this a | 355 | 84.2 |
| is this | 351 | 87.1 |
| what is | 293 | 45.4 |
| what kind of | 270 | 60.2 |
| are the | 258 | 79.7 |
| is there a | 229 | 85.6 |
| what type of | 181 | 57.5 |
| where is the | 166 | 38.8 |
| what color are the | 160 | 73.8 |
| is it | 159 | 91.9 |
| what are the | 153 | 56.3 |
| are there | 142 | 82.9 |
| is | 141 | 82.6 |
| is there | 141 | 85.4 |
| does the | 136 | 82.4 |
| are these | 127 | 82.7 |
| what is on the | 115 | 49.4 |
| is the man | 113 | 82.8 |
| what is the man | 112 | 63.5 |
| how | 111 | 37.7 |
| are | 110 | 74.2 |
| which | 108 | 47.3 |
| does this | 106 | 77.2 |
| how many people are | 102 | 59.5 |
| what is in the | 85 | 59.2 |
| what is this | 76 | 66.2 |
| what are | 74 | 58.2 |
| do | 73 | 79.6 |
| what does the | 72 | 23.3 |
| what color | 68 | 79.7 |
| what time | 68 | 26.8 |
| why | 66 | 22.4 |
| where are the | 65 | 38.0 |
| what color is | 61 | 80.5 |
| are there any | 59 | 82.0 |
| how many people are in | 59 | 62.8 |
| are they | 57 | 79.6 |
| who is | 51 | 41.7 |
| can you | 48 | 72.3 |
| is this an | 46 | 84.0 |
| what brand | 46 | 42.7 |
| what is the person | 45 | 68.3 |
| what sport is | 43 | 93.1 |
| is he | 42 | 88.8 |
| what animal is | 41 | 88.2 |
| what room is | 41 | 95.4 |
| has | 40 | 81.5 |
| what is the woman | 40 | 61.0 |
| is the woman | 39 | 79.0 |
| was | 39 | 85.5 |
| what is the color of the | 39 | 84.9 |
| what number is | 38 | 7.0 |
| what is the name | 37 | 13.0 |
| do you | 35 | 74.4 |
| is that a | 35 | 80.4 |
| is the person | 33 | 79.4 |
| is this person | 30 | 79.2 |
| could | 27 | 73.1 |
| why is the | 18 | 24.4 |
