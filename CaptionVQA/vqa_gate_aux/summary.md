# VQA: baseline vs + BLIP captions

Train 643,110 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (UpDn: 1 = image; MCAN: r, 1 = use caption) |
|---|---:|---:|---:|---:|---:|---:|---:|
| + BLIP captions | 2 | **68.13 ± 0.04** | 84.65 ± 0.33 | 50.81 ± 0.31 | 60.42 ± 0.26 | 64.51 ± 0.67 | 0.57 ± 0.01 |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | + BLIP captions |
|---|---:|---:|
| how many | 977 | 56.6 |
| is the | 811 | 82.6 |
| what | 759 | 50.6 |
| what color is the | 638 | 84.6 |
| what is the | 551 | 56.5 |
| none of the above | 390 | 61.5 |
| is this a | 355 | 85.6 |
| is this | 351 | 89.0 |
| what is | 293 | 47.7 |
| what kind of | 270 | 60.5 |
| are the | 258 | 80.6 |
| is there a | 229 | 87.4 |
| what type of | 181 | 57.7 |
| where is the | 166 | 43.2 |
| what color are the | 160 | 75.7 |
| is it | 159 | 94.7 |
| what are the | 153 | 57.1 |
| are there | 142 | 84.8 |
| is | 141 | 81.9 |
| is there | 141 | 86.1 |
| does the | 136 | 82.3 |
| are these | 127 | 83.3 |
| what is on the | 115 | 48.9 |
| is the man | 113 | 82.4 |
| what is the man | 112 | 64.5 |
| how | 111 | 40.0 |
| are | 110 | 76.8 |
| which | 108 | 51.1 |
| does this | 106 | 78.6 |
| how many people are | 102 | 60.3 |
| what is in the | 85 | 60.6 |
| what is this | 76 | 67.8 |
| what are | 74 | 60.3 |
| do | 73 | 77.9 |
| what does the | 72 | 27.6 |
| what color | 68 | 77.4 |
| what time | 68 | 25.4 |
| why | 66 | 25.8 |
| where are the | 65 | 38.8 |
| what color is | 61 | 84.7 |
| are there any | 59 | 88.5 |
| how many people are in | 59 | 71.0 |
| are they | 57 | 78.9 |
| who is | 51 | 41.9 |
| can you | 48 | 72.3 |
| is this an | 46 | 88.0 |
| what brand | 46 | 42.8 |
| what is the person | 45 | 69.0 |
| what sport is | 43 | 93.8 |
| is he | 42 | 90.5 |
| what animal is | 41 | 90.4 |
| what room is | 41 | 94.1 |
| has | 40 | 73.2 |
| what is the woman | 40 | 68.4 |
| is the woman | 39 | 81.2 |
| was | 39 | 90.9 |
| what is the color of the | 39 | 88.7 |
| what number is | 38 | 7.5 |
| what is the name | 37 | 12.6 |
| do you | 35 | 73.4 |
| is that a | 35 | 90.9 |
| is the person | 33 | 83.2 |
| is this person | 30 | 72.0 |
| could | 27 | 80.0 |
| why is the | 18 | 28.1 |
