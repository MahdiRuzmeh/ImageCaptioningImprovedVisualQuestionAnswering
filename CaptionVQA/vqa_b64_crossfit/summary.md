# VQA: baseline vs + BLIP captions

Train 643,110 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| + BLIP captions | 2 | **68.08 ± 0.13** | 84.20 ± 0.69 | 49.47 ± 0.25 | 61.03 ± 0.19 | 60.48 ± 0.06 | — |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | + BLIP captions |
|---|---:|---:|
| how many | 977 | 55.2 |
| is the | 811 | 81.9 |
| what | 759 | 51.4 |
| what color is the | 638 | 84.4 |
| what is the | 551 | 57.4 |
| none of the above | 390 | 61.1 |
| is this a | 355 | 86.6 |
| is this | 351 | 87.9 |
| what is | 293 | 49.1 |
| what kind of | 270 | 61.1 |
| are the | 258 | 79.8 |
| is there a | 229 | 87.3 |
| what type of | 181 | 59.7 |
| where is the | 166 | 43.0 |
| what color are the | 160 | 73.6 |
| is it | 159 | 94.7 |
| what are the | 153 | 60.1 |
| are there | 142 | 83.4 |
| is | 141 | 80.2 |
| is there | 141 | 85.3 |
| does the | 136 | 80.9 |
| are these | 127 | 82.3 |
| what is on the | 115 | 50.3 |
| is the man | 113 | 82.3 |
| what is the man | 112 | 67.9 |
| how | 111 | 36.1 |
| are | 110 | 78.6 |
| which | 108 | 46.6 |
| does this | 106 | 73.4 |
| how many people are | 102 | 61.3 |
| what is in the | 85 | 59.0 |
| what is this | 76 | 73.6 |
| what are | 74 | 62.6 |
| do | 73 | 79.0 |
| what does the | 72 | 27.4 |
| what color | 68 | 77.6 |
| what time | 68 | 27.2 |
| why | 66 | 21.9 |
| where are the | 65 | 42.7 |
| what color is | 61 | 81.7 |
| are there any | 59 | 87.7 |
| how many people are in | 59 | 66.6 |
| are they | 57 | 83.9 |
| who is | 51 | 44.8 |
| can you | 48 | 70.0 |
| is this an | 46 | 89.2 |
| what brand | 46 | 42.6 |
| what is the person | 45 | 72.4 |
| what sport is | 43 | 95.6 |
| is he | 42 | 89.3 |
| what animal is | 41 | 87.6 |
| what room is | 41 | 94.3 |
| has | 40 | 81.9 |
| what is the woman | 40 | 70.8 |
| is the woman | 39 | 81.2 |
| was | 39 | 87.6 |
| what is the color of the | 39 | 87.3 |
| what number is | 38 | 10.9 |
| what is the name | 37 | 15.5 |
| do you | 35 | 73.6 |
| is that a | 35 | 88.3 |
| is the person | 33 | 83.0 |
| is this person | 30 | 79.8 |
| could | 27 | 80.7 |
| why is the | 18 | 25.6 |
