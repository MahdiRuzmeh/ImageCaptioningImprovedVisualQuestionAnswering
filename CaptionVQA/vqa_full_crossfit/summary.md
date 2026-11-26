# VQA: baseline vs + BLIP captions

Train 443,757 questions (train2014 images the captioner never saw), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| + BLIP captions | 3 | **66.08 ± 0.20** | 82.30 ± 0.41 | 48.11 ± 1.01 | 58.77 ± 0.13 | 47.20 ± 0.47 | 0.55 ± 0.00 |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | + BLIP captions |
|---|---:|---:|
| how many | 977 | 53.9 |
| is the | 811 | 81.0 |
| what | 759 | 51.1 |
| what color is the | 638 | 81.5 |
| what is the | 551 | 53.6 |
| none of the above | 390 | 56.9 |
| is this a | 355 | 83.3 |
| is this | 351 | 86.6 |
| what is | 293 | 44.7 |
| what kind of | 270 | 58.4 |
| are the | 258 | 78.0 |
| is there a | 229 | 84.7 |
| what type of | 181 | 56.7 |
| where is the | 166 | 41.3 |
| what color are the | 160 | 71.4 |
| is it | 159 | 91.3 |
| what are the | 153 | 56.5 |
| are there | 142 | 83.9 |
| is | 141 | 77.4 |
| is there | 141 | 82.1 |
| does the | 136 | 80.6 |
| are these | 127 | 78.1 |
| what is on the | 115 | 51.6 |
| is the man | 113 | 78.2 |
| what is the man | 112 | 69.2 |
| how | 111 | 31.8 |
| are | 110 | 78.7 |
| which | 108 | 43.3 |
| does this | 106 | 71.8 |
| how many people are | 102 | 58.8 |
| what is in the | 85 | 57.9 |
| what is this | 76 | 72.6 |
| what are | 74 | 67.4 |
| do | 73 | 73.8 |
| what does the | 72 | 26.2 |
| what color | 68 | 75.0 |
| what time | 68 | 26.2 |
| why | 66 | 16.2 |
| where are the | 65 | 45.5 |
| what color is | 61 | 86.6 |
| are there any | 59 | 87.9 |
| how many people are in | 59 | 64.0 |
| are they | 57 | 80.8 |
| who is | 51 | 34.1 |
| can you | 48 | 77.2 |
| is this an | 46 | 84.9 |
| what brand | 46 | 47.4 |
| what is the person | 45 | 69.9 |
| what sport is | 43 | 94.2 |
| is he | 42 | 89.0 |
| what animal is | 41 | 83.6 |
| what room is | 41 | 94.1 |
| has | 40 | 72.8 |
| what is the woman | 40 | 72.9 |
| is the woman | 39 | 81.8 |
| was | 39 | 82.8 |
| what is the color of the | 39 | 83.0 |
| what number is | 38 | 8.9 |
| what is the name | 37 | 16.5 |
| do you | 35 | 72.1 |
| is that a | 35 | 81.6 |
| is the person | 33 | 85.1 |
| is this person | 30 | 70.4 |
| could | 27 | 83.0 |
| why is the | 18 | 24.4 |
