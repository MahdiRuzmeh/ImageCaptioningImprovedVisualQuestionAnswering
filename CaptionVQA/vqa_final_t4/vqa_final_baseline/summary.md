# VQA: baseline vs + BLIP captions

Train 658,111 questions (train2014), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (UpDn: 1 = image; MCAN: r, 1 = use caption) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (no caption) | 2 | **78.76 ± 0.17** | 93.93 ± 0.08 | 62.19 ± 0.08 | 71.87 ± 0.26 | — | — |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | Baseline (no caption) |
|---|---:|---:|
| how many | 977 | 69.2 |
| is the | 811 | 91.8 |
| what | 759 | 62.1 |
| what color is the | 638 | 90.4 |
| what is the | 551 | 68.0 |
| none of the above | 390 | 75.2 |
| is this a | 355 | 95.7 |
| is this | 351 | 95.9 |
| what is | 293 | 66.6 |
| what kind of | 270 | 71.0 |
| are the | 258 | 91.5 |
| is there a | 229 | 97.3 |
| what type of | 181 | 71.7 |
| where is the | 166 | 51.1 |
| what color are the | 160 | 86.2 |
| is it | 159 | 96.7 |
| what are the | 153 | 72.1 |
| are there | 142 | 95.0 |
| is | 141 | 92.2 |
| is there | 141 | 93.9 |
| does the | 136 | 92.4 |
| are these | 127 | 94.9 |
| what is on the | 115 | 67.2 |
| is the man | 113 | 92.8 |
| what is the man | 112 | 76.8 |
| how | 111 | 51.9 |
| are | 110 | 88.7 |
| which | 108 | 63.8 |
| does this | 106 | 91.5 |
| how many people are | 102 | 70.7 |
| what is in the | 85 | 75.6 |
| what is this | 76 | 77.6 |
| what are | 74 | 74.3 |
| do | 73 | 92.1 |
| what does the | 72 | 31.6 |
| what color | 68 | 88.5 |
| what time | 68 | 29.5 |
| why | 66 | 41.6 |
| where are the | 65 | 59.2 |
| what color is | 61 | 86.6 |
| are there any | 59 | 92.5 |
| how many people are in | 59 | 80.9 |
| are they | 57 | 88.9 |
| who is | 51 | 61.3 |
| can you | 48 | 88.8 |
| is this an | 46 | 95.1 |
| what brand | 46 | 51.2 |
| what is the person | 45 | 75.3 |
| what sport is | 43 | 96.3 |
| is he | 42 | 91.9 |
| what animal is | 41 | 92.7 |
| what room is | 41 | 97.6 |
| has | 40 | 86.4 |
| what is the woman | 40 | 75.4 |
| is the woman | 39 | 90.0 |
| was | 39 | 92.3 |
| what is the color of the | 39 | 93.5 |
| what number is | 38 | 17.8 |
| what is the name | 37 | 31.4 |
| do you | 35 | 95.0 |
| is that a | 35 | 92.9 |
| is the person | 33 | 90.8 |
| is this person | 30 | 86.2 |
| could | 27 | 96.3 |
| why is the | 18 | 33.9 |
