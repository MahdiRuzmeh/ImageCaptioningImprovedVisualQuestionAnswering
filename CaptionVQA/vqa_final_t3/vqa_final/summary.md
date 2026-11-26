# VQA: baseline vs + BLIP captions

Train 658,111 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (UpDn: 1 = image; MCAN: r, 1 = use caption) |
|---|---:|---:|---:|---:|---:|---:|---:|
| + BLIP captions | 2 | **78.83 ± 0.04** | 93.53 ± 0.67 | 60.22 ± 0.58 | 72.86 ± 0.26 | 73.58 ± 0.31 | — |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | + BLIP captions |
|---|---:|---:|
| how many | 977 | 66.8 |
| is the | 811 | 92.1 |
| what | 759 | 62.7 |
| what color is the | 638 | 90.3 |
| what is the | 551 | 70.1 |
| none of the above | 390 | 75.6 |
| is this a | 355 | 96.0 |
| is this | 351 | 94.5 |
| what is | 293 | 67.4 |
| what kind of | 270 | 72.8 |
| are the | 258 | 89.9 |
| is there a | 229 | 95.9 |
| what type of | 181 | 71.7 |
| where is the | 166 | 52.9 |
| what color are the | 160 | 86.2 |
| is it | 159 | 98.0 |
| what are the | 153 | 73.1 |
| are there | 142 | 93.9 |
| is | 141 | 92.8 |
| is there | 141 | 93.5 |
| does the | 136 | 92.4 |
| are these | 127 | 93.9 |
| what is on the | 115 | 66.9 |
| is the man | 113 | 93.7 |
| what is the man | 112 | 78.1 |
| how | 111 | 50.2 |
| are | 110 | 86.2 |
| which | 108 | 66.2 |
| does this | 106 | 90.7 |
| how many people are | 102 | 67.7 |
| what is in the | 85 | 78.6 |
| what is this | 76 | 79.1 |
| what are | 74 | 77.8 |
| do | 73 | 89.2 |
| what does the | 72 | 34.8 |
| what color | 68 | 89.6 |
| what time | 68 | 30.2 |
| why | 66 | 40.4 |
| where are the | 65 | 58.3 |
| what color is | 61 | 89.3 |
| are there any | 59 | 95.7 |
| how many people are in | 59 | 72.1 |
| are they | 57 | 90.7 |
| who is | 51 | 64.5 |
| can you | 48 | 80.3 |
| is this an | 46 | 94.7 |
| what brand | 46 | 56.3 |
| what is the person | 45 | 77.3 |
| what sport is | 43 | 94.9 |
| is he | 42 | 96.2 |
| what animal is | 41 | 91.8 |
| what room is | 41 | 99.5 |
| has | 40 | 90.8 |
| what is the woman | 40 | 78.4 |
| is the woman | 39 | 92.9 |
| was | 39 | 94.4 |
| what is the color of the | 39 | 93.6 |
| what number is | 38 | 23.2 |
| what is the name | 37 | 33.2 |
| do you | 35 | 95.1 |
| is that a | 35 | 92.6 |
| is the person | 33 | 89.4 |
| is this person | 30 | 85.7 |
| could | 27 | 94.4 |
| why is the | 18 | 32.2 |
