# VQA: baseline vs + BLIP captions

Train 443,757 questions (train2014 images the captioner never saw), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (no caption) | 3 | **60.03 ± 0.45** | 77.59 ± 0.98 | 40.47 ± 0.40 | 52.15 ± 0.29 | — | — |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | Baseline (no caption) |
|---|---:|---:|
| how many | 977 | 45.3 |
| is the | 811 | 74.8 |
| what | 759 | 45.7 |
| what color is the | 638 | 73.3 |
| what is the | 551 | 46.8 |
| none of the above | 390 | 55.2 |
| is this a | 355 | 76.9 |
| is this | 351 | 78.4 |
| what is | 293 | 34.7 |
| what kind of | 270 | 53.4 |
| are the | 258 | 74.1 |
| is there a | 229 | 75.6 |
| what type of | 181 | 50.5 |
| where is the | 166 | 33.2 |
| what color are the | 160 | 66.8 |
| is it | 159 | 90.7 |
| what are the | 153 | 46.8 |
| are there | 142 | 79.1 |
| is | 141 | 79.9 |
| is there | 141 | 76.6 |
| does the | 136 | 79.8 |
| are these | 127 | 78.3 |
| what is on the | 115 | 39.6 |
| is the man | 113 | 73.6 |
| what is the man | 112 | 61.5 |
| how | 111 | 29.5 |
| are | 110 | 74.3 |
| which | 108 | 43.6 |
| does this | 106 | 77.2 |
| how many people are | 102 | 52.0 |
| what is in the | 85 | 50.0 |
| what is this | 76 | 63.9 |
| what are | 74 | 58.1 |
| do | 73 | 77.8 |
| what does the | 72 | 21.6 |
| what color | 68 | 71.9 |
| what time | 68 | 24.3 |
| why | 66 | 12.7 |
| where are the | 65 | 34.6 |
| what color is | 61 | 76.3 |
| are there any | 59 | 74.9 |
| how many people are in | 59 | 49.1 |
| are they | 57 | 74.2 |
| who is | 51 | 29.9 |
| can you | 48 | 59.6 |
| is this an | 46 | 74.5 |
| what brand | 46 | 39.5 |
| what is the person | 45 | 59.9 |
| what sport is | 43 | 88.8 |
| is he | 42 | 85.6 |
| what animal is | 41 | 80.0 |
| what room is | 41 | 95.3 |
| has | 40 | 73.8 |
| what is the woman | 40 | 54.8 |
| is the woman | 39 | 71.5 |
| was | 39 | 75.3 |
| what is the color of the | 39 | 82.1 |
| what number is | 38 | 6.3 |
| what is the name | 37 | 9.4 |
| do you | 35 | 75.6 |
| is that a | 35 | 74.7 |
| is the person | 33 | 78.2 |
| is this person | 30 | 62.8 |
| could | 27 | 76.4 |
| why is the | 18 | 23.9 |
