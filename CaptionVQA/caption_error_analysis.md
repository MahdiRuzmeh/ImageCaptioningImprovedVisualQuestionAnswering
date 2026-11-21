# Why are the captions wrong? (10,001 test questions, cross-fitted BLIP-base captions)

Caption right = the majority human answer is in the caption (yes/no: right polarity), as in the reports.
Caption soft = best VQA score among the 10 human answers that appear in the caption (accepts other human answers).
VQA columns: official accuracy, mean of 2 seeds (baseline = 1b-A, + captions = 1b-B with dropout).

## 1. By question group

| Group | n | Caption right | Caption soft | VQA without caption | VQA + caption | Gain |
|---|---:|---:|---:|---:|---:|---:|
| open: color | 994 | 70.3 | 80.1 | 78.2 | 80.9 | +2.8 |
| open: count 0-3 | 804 | 55.7 | 65.7 | 65.2 | 70.0 | +4.7 |
| open: count 4+ / other number | 434 | 13.8 | 22.2 | 26.0 | 25.6 | -0.4 |
| open: read text / time | 601 | 24.6 | 29.2 | 30.6 | 31.3 | +0.7 |
| open: other (objects, places, activities ...) | 3,437 | 48.0 | 56.5 | 56.1 | 58.3 | +2.1 |
| yes/no: is there / are there | 568 | 76.4 | 76.4 | 84.5 | 86.2 | +1.7 |
| yes/no: other | 3,163 | 66.8 | 66.8 | 83.2 | 84.5 | +1.3 |
| **all** | 10,001 | 55.5 | 60.8 | 66.4 | 68.3 | +1.9 |

## 2. Why open-question captions are wrong (3,264 questions)

| Reason | n | Share | VQA without caption is right (score ≥ 0.6) |
|---|---:|---:|---:|
| caption names something else | 1,415 | 43.4% | 21.8% |
| another human answer is in the caption | 1,079 | 33.1% | 49.2% |
| caption gives a different number | 421 | 12.9% | 20.7% |
| caption gives no number | 186 | 5.7% | 15.1% |
| caption gives a different color | 124 | 3.8% | 40.3% |
| caption gives no color | 34 | 1.0% | 26.5% |
| only singular/plural differs | 5 | 0.2% | 40.0% |

Wrong open captions by group:

| Group | wrong | of all wrong open |
|---|---:|---:|
| open: color | 295 | 9.0% |
| open: count 0-3 | 356 | 10.9% |
| open: count 4+ / other number | 374 | 11.5% |
| open: read text / time | 453 | 13.9% |
| open: other (objects, places, activities ...) | 1,786 | 54.7% |

## 3. Yes/no captions with the wrong polarity (1,184 of 3,731)

| Error | n | VQA without caption is right |
|---|---:|---:|
| caption says no, answer yes | 712 | 76.3% |
| caption says yes, answer no | 472 | 75.0% |

Answer "yes" for 49.5% of yes/no questions; captions are negative for 56.9%.

## 4. Human agreement on the answer

| Votes for the majority answer (of 10) | n | Caption right | VQA without caption |
|---|---:|---:|---:|
| 8–10 (clear) | 5,748 | 69.9 | 75.1 |
| 5–7 | 2,350 | 48.2 | 69.9 |
| ≤ 4 (ambiguous) | 1,903 | 21.3 | 35.7 |

## 5. Captioner vs VQA without caption: who knows the answer?

| | n | Share |
|---|---:|---:|
| both | 4,427 | 44.3% |
| only the caption (new information for VQA) | 1,126 | 11.3% |
| only the VQA model (caption can only mislead) | 1,913 | 19.1% |
| neither | 2,535 | 25.3% |

## 6. Examples of wrong open captions (first 3 per reason)

**caption names something else**

- Q: What fruit is on the shirt by the cat's head? — answer *pineapple* — caption: "there is no fruit on the shirt by the cat ' s head."
- Q: Why is she carrying an umbrella inside? — answer *for picture* — caption: "she is carrying an umbrella inside."
- Q: What is se posing next to? — answer *cabinet* — caption: "the woman is posing next to a dresser."

**another human answer is in the caption**

- Q: What kind of shoes are pictured? — answer *converse* — caption: "tennis shoes are pictured."
- Q: Where is the fire extinguisher? — answer *missing* — caption: "the fire extinguisher is on the wall."
- Q: What kind of sign is this? — answer *stop* — caption: "this sign is a street sign."

**caption gives a different number**

- Q: How many people are carrying surfboards? — answer *1* — caption: "four people are carrying surfboards."
- Q: How many horses are in this scene? — answer *5* — caption: "there are four horses."
- Q: How many boats are there? — answer *9* — caption: "there are seven boats."

**caption gives a different color**

- Q: What color is the seahorse? — answer *yellow and orange* — caption: "the seahorse is yellow."
- Q: What color is the little box on top of the shelf? — answer *green* — caption: "the little box on top of the shelf is red."
- Q: What color are the shoes? — answer *black* — caption: "the shoes are blue."

**caption gives no color**

- Q: What kind of bird is this? — answer *yellow* — caption: "this bird is a canary."
- Q: Who is one of the sponsors? — answer *orange* — caption: "one of the sponsors is a tennis player."
- Q: What coat of paint does the plane have? — answer *gray* — caption: "the plane has no coat of paint."

**only singular/plural differs**

- Q: What shape do the beams on the barn doors make? — answer *rectangle* — caption: "the beams on the barn doors are in the shape of rectangles."
- Q: What is the predominant vegetable on the table? — answer *carrots* — caption: "the predominant vegetable on the table is a carrot."
- Q: Which animal is these? — answer *cat* — caption: "these are cats."

**caption gives no number**

- Q: What's this man's number? — answer *2* — caption: "this man ' s number is a number."
- Q: What is Beltre's number? — answer *17* — caption: "the beltre ' s number is a number."
- Q: What is the number on the bus? — answer *1038* — caption: "the number forty is on the bus."

