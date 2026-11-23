# Why are the captions wrong? (10,001 test questions, cross-fitted BLIP-base captions)

Caption right = the majority human answer is in the caption (yes/no: right polarity), as in the reports.
Caption soft = best VQA score among the 10 human answers that appear in the caption (accepts other human answers).
VQA columns: official accuracy, mean of 2 seeds (baseline = 1b-A, + captions = 1b-B with dropout).

## 1. By question group

| Group | n | Caption right | Caption soft | VQA without caption | VQA + caption | Gain |
|---|---:|---:|---:|---:|---:|---:|
| open: color | 994 | 74.2 | 83.1 | 78.2 | 81.9 | +3.8 |
| open: count 0-3 | 804 | 58.8 | 68.4 | 65.2 | 70.4 | +5.1 |
| open: count 4+ / other number | 434 | 15.0 | 24.7 | 26.0 | 25.7 | -0.2 |
| open: read text / time | 601 | 26.6 | 31.8 | 30.6 | 33.9 | +3.2 |
| open: other (objects, places, activities ...) | 3,437 | 51.6 | 60.3 | 56.1 | 60.5 | +4.3 |
| yes/no: is there / are there | 568 | 81.3 | 81.3 | 84.5 | 88.8 | +4.3 |
| yes/no: other | 3,163 | 70.2 | 70.2 | 83.2 | 85.3 | +2.1 |
| **all** | 10,001 | 58.9 | 64.3 | 66.4 | 69.8 | +3.4 |

## 2. Why open-question captions are wrong (3,059 questions)

| Reason | n | Share | VQA without caption is right (score ≥ 0.6) |
|---|---:|---:|---:|
| caption names something else | 1,269 | 41.5% | 21.0% |
| another human answer is in the caption | 1,082 | 35.4% | 45.5% |
| caption gives a different number | 416 | 13.6% | 20.0% |
| caption gives no number | 151 | 4.9% | 13.2% |
| caption gives a different color | 102 | 3.3% | 38.2% |
| caption gives no color | 32 | 1.0% | 31.2% |
| only singular/plural differs | 7 | 0.2% | 28.6% |

Wrong open captions by group:

| Group | wrong | of all wrong open |
|---|---:|---:|
| open: color | 256 | 8.4% |
| open: count 0-3 | 331 | 10.8% |
| open: count 4+ / other number | 369 | 12.1% |
| open: read text / time | 441 | 14.4% |
| open: other (objects, places, activities ...) | 1,662 | 54.3% |

## 3. Yes/no captions with the wrong polarity (1,047 of 3,731)

| Error | n | VQA without caption is right |
|---|---:|---:|
| caption says no, answer yes | 626 | 78.0% |
| caption says yes, answer no | 421 | 75.5% |

Answer "yes" for 49.5% of yes/no questions; captions are negative for 56.0%.

## 4. Human agreement on the answer

| Votes for the majority answer (of 10) | n | Caption right | VQA without caption |
|---|---:|---:|---:|
| 8–10 (clear) | 5,748 | 73.9 | 75.1 |
| 5–7 | 2,350 | 51.4 | 69.9 |
| ≤ 4 (ambiguous) | 1,903 | 23.1 | 35.7 |

## 5. Captioner vs VQA without caption: who knows the answer?

| | n | Share |
|---|---:|---:|
| both | 4,621 | 46.2% |
| only the caption (new information for VQA) | 1,274 | 12.7% |
| only the VQA model (caption can only mislead) | 1,719 | 17.2% |
| neither | 2,387 | 23.9% |

## 6. Examples of wrong open captions (first 3 per reason)

**caption names something else**

- Q: What fruit is on the shirt by the cat's head? — answer *pineapple* — caption: "there is no fruit on the shirt by the cat ' s head."
- Q: Why is she carrying an umbrella inside? — answer *for picture* — caption: "she is carrying an umbrella inside."
- Q: Why is he at the beach in long pants? — answer *surfing* — caption: "he is at the beach in long pants to surf."

**another human answer is in the caption**

- Q: What kind of shoes are pictured? — answer *converse* — caption: "sneakers are pictured."
- Q: Where is the fire extinguisher? — answer *missing* — caption: "the fire extinguisher is on the wall."
- Q: What kind of sign is this? — answer *stop* — caption: "this is a street sign."

**caption gives a different number**

- Q: How many people are carrying surfboards? — answer *1* — caption: "four people are carrying surfboards."
- Q: How many horses are in this scene? — answer *5* — caption: "there are six horses."
- Q: How many surfers do you see? — answer *1* — caption: "two surfers can be seen."

**caption gives a different color**

- Q: What color is the little box on top of the shelf? — answer *green* — caption: "the little box on top of the shelf is red."
- Q: What color is the beak? — answer *white* — caption: "the beak is orange."
- Q: What color is the bird's bill? — answer *orange* — caption: "the bird ' s bill is white."

**caption gives no color**

- Q: What kind of bird is this? — answer *yellow* — caption: "this bird is a canary."
- Q: Who is one of the sponsors? — answer *orange* — caption: "one of the sponsors is in the audience."
- Q: What kind of horses are these? — answer *brown* — caption: "these are foals."

**caption gives no number**

- Q: How many spots does the dog have? — answer *3* — caption: "the dog has a lot of spots."
- Q: What's this man's number? — answer *2* — caption: "this man ' s number is unknown."
- Q: How many broccoli trees are there? — answer *3* — caption: "there are no broccoli trees."

**only singular/plural differs**

- Q: What does the sheep have on its head? — answer *tag* — caption: "the sheep has tags on its head."
- Q: What is open in the truck? — answer *door* — caption: "doors are open in the truck."
- Q: What is the man holding? — answer *controllers* — caption: "the man is holding a controller."

