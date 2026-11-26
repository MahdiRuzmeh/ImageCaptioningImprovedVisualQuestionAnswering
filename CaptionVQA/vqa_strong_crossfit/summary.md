# VQA: baseline vs + BLIP captions

Train 643,110 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| + BLIP captions | 2 | **67.80 ± 0.06** | 83.45 ± 0.23 | 48.93 ± 0.09 | 61.17 ± 0.32 | 60.21 ± 0.31 | — |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | + BLIP captions |
|---|---:|---:|
| how many | 977 | 54.7 |
| is the | 811 | 81.8 |
| what | 759 | 51.8 |
| what color is the | 638 | 84.6 |
| what is the | 551 | 57.2 |
| none of the above | 390 | 61.1 |
| is this a | 355 | 85.4 |
| is this | 351 | 85.8 |
| what is | 293 | 46.9 |
| what kind of | 270 | 60.7 |
| are the | 258 | 79.7 |
| is there a | 229 | 84.0 |
| what type of | 181 | 60.7 |
| where is the | 166 | 44.2 |
| what color are the | 160 | 75.7 |
| is it | 159 | 93.8 |
| what are the | 153 | 59.9 |
| are there | 142 | 81.7 |
| is | 141 | 81.5 |
| is there | 141 | 85.5 |
| does the | 136 | 80.9 |
| are these | 127 | 83.0 |
| what is on the | 115 | 51.2 |
| is the man | 113 | 84.2 |
| what is the man | 112 | 68.1 |
| how | 111 | 36.2 |
| are | 110 | 77.3 |
| which | 108 | 47.9 |
| does this | 106 | 75.8 |
| how many people are | 102 | 57.1 |
| what is in the | 85 | 59.7 |
| what is this | 76 | 72.5 |
| what are | 74 | 60.6 |
| do | 73 | 77.9 |
| what does the | 72 | 24.2 |
| what color | 68 | 79.2 |
| what time | 68 | 26.8 |
| why | 66 | 24.7 |
| where are the | 65 | 46.8 |
| what color is | 61 | 82.3 |
| are there any | 59 | 89.3 |
| how many people are in | 59 | 68.8 |
| are they | 57 | 82.5 |
| who is | 51 | 46.6 |
| can you | 48 | 67.9 |
| is this an | 46 | 85.1 |
| what brand | 46 | 44.2 |
| what is the person | 45 | 74.0 |
| what sport is | 43 | 96.3 |
| is he | 42 | 88.2 |
| what animal is | 41 | 84.6 |
| what room is | 41 | 94.4 |
| has | 40 | 77.9 |
| what is the woman | 40 | 70.0 |
| is the woman | 39 | 81.2 |
| was | 39 | 87.3 |
| what is the color of the | 39 | 87.4 |
| what number is | 38 | 6.2 |
| what is the name | 37 | 17.2 |
| do you | 35 | 72.6 |
| is that a | 35 | 85.9 |
| is the person | 33 | 87.0 |
| is this person | 30 | 73.7 |
| could | 27 | 79.6 |
| why is the | 18 | 28.9 |
