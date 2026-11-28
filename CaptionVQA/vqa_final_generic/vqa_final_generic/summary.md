# VQA: baseline vs + BLIP captions

Train 658,111 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (UpDn: 1 = image; MCAN: r, 1 = use caption) |
|---|---:|---:|---:|---:|---:|---:|---:|
| + BLIP captions | 2 | **78.76 ± 0.06** | 93.96 ± 0.11 | 61.31 ± 0.18 | 72.09 ± 0.01 | 77.37 ± 0.12 | — |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | + BLIP captions |
|---|---:|---:|
| how many | 977 | 68.1 |
| is the | 811 | 91.9 |
| what | 759 | 62.2 |
| what color is the | 638 | 89.7 |
| what is the | 551 | 68.8 |
| none of the above | 390 | 73.8 |
| is this a | 355 | 96.0 |
| is this | 351 | 96.5 |
| what is | 293 | 66.2 |
| what kind of | 270 | 72.1 |
| are the | 258 | 92.7 |
| is there a | 229 | 95.4 |
| what type of | 181 | 72.8 |
| where is the | 166 | 52.6 |
| what color are the | 160 | 85.9 |
| is it | 159 | 97.8 |
| what are the | 153 | 73.8 |
| are there | 142 | 95.4 |
| is | 141 | 92.5 |
| is there | 141 | 93.7 |
| does the | 136 | 94.9 |
| are these | 127 | 93.1 |
| what is on the | 115 | 68.7 |
| is the man | 113 | 91.2 |
| what is the man | 112 | 76.6 |
| how | 111 | 50.0 |
| are | 110 | 89.3 |
| which | 108 | 62.5 |
| does this | 106 | 92.2 |
| how many people are | 102 | 72.0 |
| what is in the | 85 | 76.4 |
| what is this | 76 | 79.8 |
| what are | 74 | 75.0 |
| do | 73 | 87.5 |
| what does the | 72 | 31.9 |
| what color | 68 | 91.2 |
| what time | 68 | 27.9 |
| why | 66 | 42.3 |
| where are the | 65 | 58.8 |
| what color is | 61 | 89.6 |
| are there any | 59 | 92.5 |
| how many people are in | 59 | 77.7 |
| are they | 57 | 90.6 |
| who is | 51 | 59.6 |
| can you | 48 | 89.5 |
| is this an | 46 | 94.7 |
| what brand | 46 | 52.9 |
| what is the person | 45 | 79.2 |
| what sport is | 43 | 96.3 |
| is he | 42 | 91.5 |
| what animal is | 41 | 93.0 |
| what room is | 41 | 97.7 |
| has | 40 | 89.4 |
| what is the woman | 40 | 73.5 |
| is the woman | 39 | 86.9 |
| was | 39 | 93.3 |
| what is the color of the | 39 | 91.8 |
| what number is | 38 | 10.9 |
| what is the name | 37 | 22.7 |
| do you | 35 | 98.6 |
| is that a | 35 | 92.9 |
| is the person | 33 | 90.3 |
| is this person | 30 | 91.8 |
| could | 27 | 96.3 |
| why is the | 18 | 35.6 |
