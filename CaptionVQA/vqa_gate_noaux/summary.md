# VQA: baseline vs + BLIP captions

Train 643,110 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (UpDn: 1 = image; MCAN: r, 1 = use caption) |
|---|---:|---:|---:|---:|---:|---:|---:|
| + BLIP captions | 2 | **68.53 ± 0.04** | 84.63 ± 0.28 | 51.27 ± 0.39 | 61.12 ± 0.03 | 64.90 ± 0.14 | 1.00 ± 0.00 |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | + BLIP captions |
|---|---:|---:|
| how many | 977 | 57.6 |
| is the | 811 | 82.9 |
| what | 759 | 51.6 |
| what color is the | 638 | 84.7 |
| what is the | 551 | 58.4 |
| none of the above | 390 | 60.5 |
| is this a | 355 | 87.1 |
| is this | 351 | 88.1 |
| what is | 293 | 47.9 |
| what kind of | 270 | 61.4 |
| are the | 258 | 80.8 |
| is there a | 229 | 88.5 |
| what type of | 181 | 59.7 |
| where is the | 166 | 43.1 |
| what color are the | 160 | 74.7 |
| is it | 159 | 94.1 |
| what are the | 153 | 57.8 |
| are there | 142 | 84.1 |
| is | 141 | 80.3 |
| is there | 141 | 87.5 |
| does the | 136 | 82.8 |
| are these | 127 | 80.9 |
| what is on the | 115 | 51.7 |
| is the man | 113 | 84.5 |
| what is the man | 112 | 65.8 |
| how | 111 | 38.5 |
| are | 110 | 76.8 |
| which | 108 | 47.5 |
| does this | 106 | 74.9 |
| how many people are | 102 | 59.8 |
| what is in the | 85 | 60.2 |
| what is this | 76 | 69.9 |
| what are | 74 | 63.0 |
| do | 73 | 78.3 |
| what does the | 72 | 25.6 |
| what color | 68 | 79.5 |
| what time | 68 | 25.2 |
| why | 66 | 26.6 |
| where are the | 65 | 43.5 |
| what color is | 61 | 83.3 |
| are there any | 59 | 87.9 |
| how many people are in | 59 | 68.3 |
| are they | 57 | 79.7 |
| who is | 51 | 44.3 |
| can you | 48 | 71.0 |
| is this an | 46 | 84.1 |
| what brand | 46 | 43.8 |
| what is the person | 45 | 71.0 |
| what sport is | 43 | 92.7 |
| is he | 42 | 91.3 |
| what animal is | 41 | 87.6 |
| what room is | 41 | 94.4 |
| has | 40 | 76.9 |
| what is the woman | 40 | 69.6 |
| is the woman | 39 | 80.8 |
| was | 39 | 90.4 |
| what is the color of the | 39 | 86.2 |
| what number is | 38 | 10.1 |
| what is the name | 37 | 13.1 |
| do you | 35 | 76.3 |
| is that a | 35 | 87.1 |
| is the person | 33 | 83.6 |
| is this person | 30 | 79.2 |
| could | 27 | 88.9 |
| why is the | 18 | 28.9 |
