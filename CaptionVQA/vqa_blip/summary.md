# VQA: baseline vs + BLIP captions

Train 30,952 questions (train2014 images the captioner never saw), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| + BLIP captions | 3 | **59.79 ± 0.50** | 78.49 ± 0.82 | 45.03 ± 1.82 | 49.73 ± 1.01 | 32.30 ± 0.04 | 0.55 ± 0.05 |
| + oracle captions (ceiling) | 1 | **83.30** | 100.00 | 86.44 | 69.82 | 18.60 | 0.47 |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | + BLIP captions | + oracle captions (ceiling) |
|---|---:|---:|---:|
| how many | 977 | 51.7 | 95.0 |
| is the | 811 | 75.1 | 97.2 |
| what | 759 | 39.5 | 60.5 |
| what color is the | 638 | 80.6 | 95.8 |
| what is the | 551 | 43.5 | 66.1 |
| none of the above | 390 | 49.9 | 74.8 |
| is this a | 355 | 78.8 | 97.4 |
| is this | 351 | 81.3 | 97.7 |
| what is | 293 | 34.6 | 63.8 |
| what kind of | 270 | 47.8 | 62.1 |
| are the | 258 | 73.0 | 98.4 |
| is there a | 229 | 80.1 | 99.8 |
| what type of | 181 | 45.9 | 58.3 |
| where is the | 166 | 28.8 | 52.4 |
| what color are the | 160 | 74.9 | 95.6 |
| is it | 159 | 93.1 | 100.0 |
| what are the | 153 | 45.8 | 64.4 |
| are there | 142 | 81.1 | 99.3 |
| is | 141 | 76.1 | 98.3 |
| is there | 141 | 77.6 | 100.0 |
| does the | 136 | 71.8 | 98.5 |
| are these | 127 | 72.4 | 99.2 |
| what is on the | 115 | 37.6 | 65.3 |
| is the man | 113 | 69.7 | 97.9 |
| what is the man | 112 | 62.1 | 79.4 |
| how | 111 | 18.2 | 44.8 |
| are | 110 | 70.4 | 100.0 |
| which | 108 | 38.2 | 77.3 |
| does this | 106 | 74.7 | 100.0 |
| how many people are | 102 | 52.8 | 97.8 |
| what is in the | 85 | 49.4 | 74.7 |
| what is this | 76 | 64.5 | 71.3 |
| what are | 74 | 43.2 | 65.5 |
| do | 73 | 68.1 | 98.6 |
| what does the | 72 | 19.8 | 33.5 |
| what color | 68 | 74.0 | 96.2 |
| what time | 68 | 22.1 | 34.1 |
| why | 66 | 13.6 | 38.6 |
| where are the | 65 | 29.8 | 45.5 |
| what color is | 61 | 85.7 | 98.9 |
| are there any | 59 | 86.6 | 100.0 |
| how many people are in | 59 | 56.2 | 98.0 |
| are they | 57 | 78.4 | 94.7 |
| who is | 51 | 27.8 | 63.7 |
| can you | 48 | 62.1 | 100.0 |
| is this an | 46 | 83.6 | 100.0 |
| what brand | 46 | 23.3 | 44.8 |
| what is the person | 45 | 55.7 | 80.7 |
| what sport is | 43 | 91.3 | 92.1 |
| is he | 42 | 86.4 | 100.0 |
| what animal is | 41 | 80.7 | 90.5 |
| what room is | 41 | 97.7 | 97.3 |
| has | 40 | 69.9 | 97.5 |
| what is the woman | 40 | 57.7 | 71.8 |
| is the woman | 39 | 72.9 | 94.9 |
| was | 39 | 69.4 | 97.4 |
| what is the color of the | 39 | 88.9 | 99.2 |
| what number is | 38 | 6.8 | 61.3 |
| what is the name | 37 | 10.5 | 29.2 |
| do you | 35 | 72.8 | 100.0 |
| is that a | 35 | 75.0 | 92.0 |
| is the person | 33 | 76.3 | 97.0 |
| is this person | 30 | 77.8 | 100.0 |
| could | 27 | 64.3 | 100.0 |
| why is the | 18 | 20.6 | 23.9 |
