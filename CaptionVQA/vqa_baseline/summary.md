# VQA: baseline vs + BLIP captions

Train 30,952 questions (train2014 images the captioner never saw), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (no caption) | 3 | **46.56 ± 0.44** | 67.80 ± 0.40 | 28.31 ± 0.50 | 35.53 ± 0.55 | — | — |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | Baseline (no caption) |
|---|---:|---:|
| how many | 977 | 32.0 |
| is the | 811 | 65.8 |
| what | 759 | 31.3 |
| what color is the | 638 | 51.7 |
| what is the | 551 | 29.4 |
| none of the above | 390 | 42.9 |
| is this a | 355 | 65.8 |
| is this | 351 | 68.0 |
| what is | 293 | 20.0 |
| what kind of | 270 | 36.2 |
| are the | 258 | 64.6 |
| is there a | 229 | 61.8 |
| what type of | 181 | 31.5 |
| where is the | 166 | 22.5 |
| what color are the | 160 | 43.0 |
| is it | 159 | 72.7 |
| what are the | 153 | 29.8 |
| are there | 142 | 60.9 |
| is | 141 | 70.3 |
| is there | 141 | 69.4 |
| does the | 136 | 68.2 |
| are these | 127 | 66.4 |
| what is on the | 115 | 27.3 |
| is the man | 113 | 63.0 |
| what is the man | 112 | 46.8 |
| how | 111 | 15.8 |
| are | 110 | 59.9 |
| which | 108 | 33.5 |
| does this | 106 | 72.3 |
| how many people are | 102 | 37.3 |
| what is in the | 85 | 30.3 |
| what is this | 76 | 43.2 |
| what are | 74 | 28.6 |
| do | 73 | 69.7 |
| what does the | 72 | 17.3 |
| what color | 68 | 48.2 |
| what time | 68 | 18.4 |
| why | 66 | 11.5 |
| where are the | 65 | 19.6 |
| what color is | 61 | 61.6 |
| are there any | 59 | 65.5 |
| how many people are in | 59 | 32.4 |
| are they | 57 | 64.7 |
| who is | 51 | 24.6 |
| can you | 48 | 52.7 |
| is this an | 46 | 75.4 |
| what brand | 46 | 27.8 |
| what is the person | 45 | 50.2 |
| what sport is | 43 | 83.5 |
| is he | 42 | 73.5 |
| what animal is | 41 | 68.6 |
| what room is | 41 | 91.3 |
| has | 40 | 71.8 |
| what is the woman | 40 | 37.3 |
| is the woman | 39 | 64.4 |
| was | 39 | 68.5 |
| what is the color of the | 39 | 53.6 |
| what number is | 38 | 6.1 |
| what is the name | 37 | 9.6 |
| do you | 35 | 64.4 |
| is that a | 35 | 55.8 |
| is the person | 33 | 64.4 |
| is this person | 30 | 57.7 |
| could | 27 | 56.4 |
| why is the | 18 | 13.1 |
