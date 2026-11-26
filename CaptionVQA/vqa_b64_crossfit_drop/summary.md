# VQA: baseline vs + BLIP captions

Train 643,110 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| + BLIP captions | 2 | **68.31 ± 0.35** | 84.90 ± 0.39 | 51.21 ± 0.06 | 60.49 ± 0.44 | 64.48 ± 0.02 | — |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | + BLIP captions |
|---|---:|---:|
| how many | 977 | 57.5 |
| is the | 811 | 83.6 |
| what | 759 | 50.3 |
| what color is the | 638 | 84.2 |
| what is the | 551 | 56.6 |
| none of the above | 390 | 61.6 |
| is this a | 355 | 86.3 |
| is this | 351 | 87.8 |
| what is | 293 | 46.8 |
| what kind of | 270 | 60.3 |
| are the | 258 | 79.4 |
| is there a | 229 | 87.7 |
| what type of | 181 | 59.8 |
| where is the | 166 | 42.7 |
| what color are the | 160 | 75.8 |
| is it | 159 | 94.3 |
| what are the | 153 | 60.5 |
| are there | 142 | 83.0 |
| is | 141 | 83.9 |
| is there | 141 | 85.7 |
| does the | 136 | 82.5 |
| are these | 127 | 81.7 |
| what is on the | 115 | 49.8 |
| is the man | 113 | 83.9 |
| what is the man | 112 | 64.7 |
| how | 111 | 37.8 |
| are | 110 | 75.1 |
| which | 108 | 46.8 |
| does this | 106 | 80.0 |
| how many people are | 102 | 63.5 |
| what is in the | 85 | 61.1 |
| what is this | 76 | 70.9 |
| what are | 74 | 60.4 |
| do | 73 | 76.8 |
| what does the | 72 | 26.2 |
| what color | 68 | 78.8 |
| what time | 68 | 24.4 |
| why | 66 | 28.2 |
| where are the | 65 | 38.7 |
| what color is | 61 | 83.5 |
| are there any | 59 | 89.1 |
| how many people are in | 59 | 69.3 |
| are they | 57 | 83.3 |
| who is | 51 | 43.4 |
| can you | 48 | 73.3 |
| is this an | 46 | 90.7 |
| what brand | 46 | 42.8 |
| what is the person | 45 | 70.9 |
| what sport is | 43 | 94.2 |
| is he | 42 | 92.1 |
| what animal is | 41 | 86.6 |
| what room is | 41 | 94.3 |
| has | 40 | 83.8 |
| what is the woman | 40 | 65.1 |
| is the woman | 39 | 83.7 |
| was | 39 | 91.2 |
| what is the color of the | 39 | 86.2 |
| what number is | 38 | 6.6 |
| what is the name | 37 | 14.5 |
| do you | 35 | 74.7 |
| is that a | 35 | 86.7 |
| is the person | 33 | 81.7 |
| is this person | 30 | 75.8 |
| could | 27 | 78.9 |
| why is the | 18 | 26.1 |
