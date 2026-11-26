# VQA: baseline vs + BLIP captions

Train 643,110 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (UpDn: 1 = image; MCAN: r, 1 = use caption) |
|---|---:|---:|---:|---:|---:|---:|---:|
| + BLIP captions | 2 | **69.76 ± 0.07** | 85.99 ± 0.08 | 51.55 ± 0.33 | 62.52 ± 0.01 | 63.44 ± 0.41 | — |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | + BLIP captions |
|---|---:|---:|
| how many | 977 | 58.1 |
| is the | 811 | 84.2 |
| what | 759 | 51.9 |
| what color is the | 638 | 85.0 |
| what is the | 551 | 59.0 |
| none of the above | 390 | 62.3 |
| is this a | 355 | 88.3 |
| is this | 351 | 88.0 |
| what is | 293 | 50.0 |
| what kind of | 270 | 65.0 |
| are the | 258 | 81.7 |
| is there a | 229 | 89.8 |
| what type of | 181 | 59.4 |
| where is the | 166 | 43.2 |
| what color are the | 160 | 78.9 |
| is it | 159 | 95.1 |
| what are the | 153 | 60.5 |
| are there | 142 | 86.7 |
| is | 141 | 84.3 |
| is there | 141 | 87.6 |
| does the | 136 | 85.3 |
| are these | 127 | 82.8 |
| what is on the | 115 | 55.8 |
| is the man | 113 | 89.0 |
| what is the man | 112 | 68.7 |
| how | 111 | 35.8 |
| are | 110 | 74.9 |
| which | 108 | 54.3 |
| does this | 106 | 80.7 |
| how many people are | 102 | 62.1 |
| what is in the | 85 | 63.6 |
| what is this | 76 | 73.2 |
| what are | 74 | 66.1 |
| do | 73 | 74.5 |
| what does the | 72 | 29.9 |
| what color | 68 | 81.5 |
| what time | 68 | 26.1 |
| why | 66 | 24.8 |
| where are the | 65 | 46.2 |
| what color is | 61 | 82.5 |
| are there any | 59 | 92.5 |
| how many people are in | 59 | 67.3 |
| are they | 57 | 84.6 |
| who is | 51 | 47.5 |
| can you | 48 | 64.2 |
| is this an | 46 | 89.5 |
| what brand | 46 | 44.8 |
| what is the person | 45 | 74.1 |
| what sport is | 43 | 92.1 |
| is he | 42 | 90.8 |
| what animal is | 41 | 86.2 |
| what room is | 41 | 96.7 |
| has | 40 | 77.2 |
| what is the woman | 40 | 72.8 |
| is the woman | 39 | 74.9 |
| was | 39 | 88.3 |
| what is the color of the | 39 | 87.4 |
| what number is | 38 | 7.5 |
| what is the name | 37 | 17.7 |
| do you | 35 | 83.6 |
| is that a | 35 | 86.9 |
| is the person | 33 | 84.7 |
| is this person | 30 | 71.3 |
| could | 27 | 85.2 |
| why is the | 18 | 29.7 |
