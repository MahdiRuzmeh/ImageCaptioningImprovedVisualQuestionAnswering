# QuestionDependentCaptionGenerator

Rule-based generator baraye sakht-e **question-dependent caption** az VQA v2.

Har sample: `(soal, javab)` → caption mesl `"The car is red."`

## Files

| File | Kar |
|------|-----|
| `caption_rules.py` | Rule engine + helper ha |
| `generate.py` | CLI baraye generate + save JSON |

## Data (pishfarz)

Mesl `SimpleVQA/configs/default.yaml`:

- Input: `../dataset/v2_OpenEnded_mscoco_*_questions.json` + `v2_mscoco_*_annotations.json`
- Output: `../dataset/v2_question_dependent_captions_{train,val}2014.json`

## Run

```bash
cd QuestionDependentCaptions
python generate.py --split train
python generate.py --split val
python generate.py --split train --max-items 1000   # smoke test
```

## Output row

```json
{
  "question_id": 262148000,
  "image_id": 262148,
  "question": "What color is the car?",
  "answer": "red",
  "caption": "The car is red.",
  "rule": "what_color"
}
```

## Notes

- Javab = mode answer (10 annotator) — hamoon logic `SimpleVQA/train.py`
- `rule_counts` to `info` baraye didan fallback ha
- Baraye train captioner: dataset loader joda lazem hast `(image_id, question, caption)`
