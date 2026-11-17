# VQA: baseline vs + BLIP captions

Train 30,952 questions (train2014 images the captioner never saw), test 10,001 questions (val2014). Official VQA v2 accuracy (%), mean ± std over seeds, best dev epoch.

## Overall and by answer type

| Model | Seeds | Overall | Yes/no | Number | Other | Shuffled captions | Gate (1 = image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (no caption) | 3 | **46.56 ± 0.44** | 67.80 ± 0.40 | 28.31 ± 0.50 | 35.53 ± 0.55 | — | — |
| + BLIP captions | 3 | **59.79 ± 0.50** | 78.49 ± 0.82 | 45.03 ± 1.82 | 49.73 ± 1.01 | 32.30 ± 0.04 | 0.55 ± 0.05 |
| + oracle captions (ceiling) | 1 | **83.30** | 100.00 | 86.44 | 69.82 | 18.60 | 0.47 |

Test questions per answer type: number 1,359, other 4,921, yes/no 3,721

**Gain of BLIP captions over the baseline** (mean): overall +13.24, yes/no +10.69, number +16.73, other +14.20

## By question type (65 VQA v2 types, mean over seeds)

| Question type | Test n | Baseline (no caption) | + BLIP captions | + oracle captions (ceiling) | BLIP − baseline |
|---|---:|---:|---:|---:|---:|
| how many | 977 | 32.0 | 51.7 | 95.0 | +19.6 |
| is the | 811 | 65.8 | 75.1 | 97.2 | +9.3 |
| what | 759 | 31.3 | 39.5 | 60.5 | +8.2 |
| what color is the | 638 | 51.7 | 80.6 | 95.8 | +28.9 |
| what is the | 551 | 29.4 | 43.5 | 66.1 | +14.1 |
| none of the above | 390 | 42.9 | 49.9 | 74.8 | +6.9 |
| is this a | 355 | 65.8 | 78.8 | 97.4 | +13.0 |
| is this | 351 | 68.0 | 81.3 | 97.7 | +13.4 |
| what is | 293 | 20.0 | 34.6 | 63.8 | +14.6 |
| what kind of | 270 | 36.2 | 47.8 | 62.1 | +11.6 |
| are the | 258 | 64.6 | 73.0 | 98.4 | +8.4 |
| is there a | 229 | 61.8 | 80.1 | 99.8 | +18.3 |
| what type of | 181 | 31.5 | 45.9 | 58.3 | +14.4 |
| where is the | 166 | 22.5 | 28.8 | 52.4 | +6.2 |
| what color are the | 160 | 43.0 | 74.9 | 95.6 | +31.9 |
| is it | 159 | 72.7 | 93.1 | 100.0 | +20.4 |
| what are the | 153 | 29.8 | 45.8 | 64.4 | +16.1 |
| are there | 142 | 60.9 | 81.1 | 99.3 | +20.3 |
| is | 141 | 70.3 | 76.1 | 98.3 | +5.8 |
| is there | 141 | 69.4 | 77.6 | 100.0 | +8.2 |
| does the | 136 | 68.2 | 71.8 | 98.5 | +3.6 |
| are these | 127 | 66.4 | 72.4 | 99.2 | +6.1 |
| what is on the | 115 | 27.3 | 37.6 | 65.3 | +10.3 |
| is the man | 113 | 63.0 | 69.7 | 97.9 | +6.6 |
| what is the man | 112 | 46.8 | 62.1 | 79.4 | +15.2 |
| how | 111 | 15.8 | 18.2 | 44.8 | +2.4 |
| are | 110 | 59.9 | 70.4 | 100.0 | +10.4 |
| which | 108 | 33.5 | 38.2 | 77.3 | +4.8 |
| does this | 106 | 72.3 | 74.7 | 100.0 | +2.3 |
| how many people are | 102 | 37.3 | 52.8 | 97.8 | +15.6 |
| what is in the | 85 | 30.3 | 49.4 | 74.7 | +19.1 |
| what is this | 76 | 43.2 | 64.5 | 71.3 | +21.4 |
| what are | 74 | 28.6 | 43.2 | 65.5 | +14.5 |
| do | 73 | 69.7 | 68.1 | 98.6 | -1.6 |
| what does the | 72 | 17.3 | 19.8 | 33.5 | +2.5 |
| what color | 68 | 48.2 | 74.0 | 96.2 | +25.7 |
| what time | 68 | 18.4 | 22.1 | 34.1 | +3.7 |
| why | 66 | 11.5 | 13.6 | 38.6 | +2.1 |
| where are the | 65 | 19.6 | 29.8 | 45.5 | +10.2 |
| what color is | 61 | 61.6 | 85.7 | 98.9 | +24.2 |
| are there any | 59 | 65.5 | 86.6 | 100.0 | +21.1 |
| how many people are in | 59 | 32.4 | 56.2 | 98.0 | +23.8 |
| are they | 57 | 64.7 | 78.4 | 94.7 | +13.7 |
| who is | 51 | 24.6 | 27.8 | 63.7 | +3.2 |
| can you | 48 | 52.7 | 62.1 | 100.0 | +9.4 |
| is this an | 46 | 75.4 | 83.6 | 100.0 | +8.1 |
| what brand | 46 | 27.8 | 23.3 | 44.8 | -4.6 |
| what is the person | 45 | 50.2 | 55.7 | 80.7 | +5.5 |
| what sport is | 43 | 83.5 | 91.3 | 92.1 | +7.8 |
| is he | 42 | 73.5 | 86.4 | 100.0 | +12.9 |
| what animal is | 41 | 68.6 | 80.7 | 90.5 | +12.0 |
| what room is | 41 | 91.3 | 97.7 | 97.3 | +6.4 |
| has | 40 | 71.8 | 69.9 | 97.5 | -1.8 |
| what is the woman | 40 | 37.3 | 57.7 | 71.8 | +20.3 |
| is the woman | 39 | 64.4 | 72.9 | 94.9 | +8.5 |
| was | 39 | 68.5 | 69.4 | 97.4 | +0.9 |
| what is the color of the | 39 | 53.6 | 88.9 | 99.2 | +35.3 |
| what number is | 38 | 6.1 | 6.8 | 61.3 | +0.6 |
| what is the name | 37 | 9.6 | 10.5 | 29.2 | +0.9 |
| do you | 35 | 64.4 | 72.8 | 100.0 | +8.4 |
| is that a | 35 | 55.8 | 75.0 | 92.0 | +19.2 |
| is the person | 33 | 64.4 | 76.3 | 97.0 | +11.8 |
| is this person | 30 | 57.7 | 77.8 | 100.0 | +20.1 |
| could | 27 | 56.4 | 64.3 | 100.0 | +7.9 |
| why is the | 18 | 13.1 | 20.6 | 23.9 | +7.4 |
