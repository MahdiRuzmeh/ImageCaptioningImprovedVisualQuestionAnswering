# Running caption generation on Kaggle

[`kaggle_runner.py`](kaggle_runner.py) continues the LLM caption generation
(`generate.py --llm`) on a Kaggle GPU using Ollama. It stops cleanly before
Kaggle's session time limit, so the dataset can be finished over several
sessions.

As of 2026-11-07: 406,573 rows in total, 237,259 `llm_fallback` done,
**100,947 `needs_llm` remaining**.

## What the script does

1. Uses the repo you cloned to `/tmp/repo` and checked out at
   `feat/kaggle-runner`, and rebuilds the local folder layout under
   `/kaggle/working/src`
   (`QuestionDependentCaptionGenerator/` and `dataset/`), so `generate.py`
   resumes from `outputs/` exactly as it does on the PC.
2. If several copies of the outputs are attached as inputs (your upload and
   earlier notebook versions), it resumes from the one with the **fewest
   `needs_llm` rows**. Only the `vqa_v2_*` files are copied into `outputs/`.
3. Installs Ollama, starts it (`OLLAMA_NUM_PARALLEL=8`, spread across both
   T4s), and pulls `qwen2.5:3b-instruct-q4_K_M`.
4. Runs `generate.py --split train --llm --batch-size 5 --llm-parallel 8
   --checkpoint-every 50`.
5. After `RUN_HOURS` (9 h) it sends Ctrl+C (SIGINT). `generate.py` then writes
   its checkpoint and exits. The script waits up to 30 minutes for that save.
6. Deletes everything except `outputs/` from `/kaggle/working`, so the
   notebook's output contains only the results (not the VQA JSONs).
7. Prints the remaining rule counts and the total wall time.

## One-time setup

1. **The code comes from git.** The repo is public, so you don't upload
   any code. The notebook clones the repo and checks out
   `feat/kaggle-runner`. Make sure any code changes you want are **pushed**
   to that branch before you start a run.

2. **Make a Kaggle Dataset** (keep it private), for example
   `vqa-caption-gen`, with only the files that are gitignored. You can
   upload them all flat into the dataset, with no folders and no zip:

   ```
   # from QuestionDependentCaptionGenerator/outputs/
   vqa_v2_question_dependent_captions_train2014.json
   vqa_v2_question_classification_result.json
   vqa_v2_question_dependent_captions_train2014_not_directly_visual.json
   vqa_v2_question_dependent_captions_train2014_validation_audit.jsonl
   vqa_v2_question_dependent_captions_train2014.json.llm_failures
   # from src/dataset/
   v2_OpenEnded_mscoco_train2014_questions.json
   v2_mscoco_train2014_annotations.json
   ```

3. **Verify your phone number** in Kaggle account settings. Without it you
   can't turn Internet on, and the script needs Internet to clone the repo and install Ollama.

## Session 1

1. Create a new notebook.
2. Right sidebar: **Accelerator → GPU T4 x2**, **Internet → On**.
3. **Add Input →** your dataset.
4. Put these three lines into one code cell:

   ```
   !git clone https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /tmp/repo
   !git -C /tmp/repo checkout feat/kaggle-runner
   !python -u /tmp/repo/QuestionDependentCaptionGenerator/kaggle/kaggle_runner.py
   ```

   The repo goes to `/tmp`, not `/kaggle/working`, so it isn't saved as
   notebook output.
5. Optional: run the cell interactively for about 10 minutes to check that
   Ollama starts and the first batches finish. Then stop it with the
   **Interrupt** button.
6. Click **Save Version → Save & Run All (Commit)**. It runs in the
   background, so you can close the browser.
7. Once the version has finished (about 9.5 h), open it, go to **Output**,
   and check the last log lines for the remaining `needs_llm` count.

## Session 2 (and any later sessions)

1. Open the same notebook → **Add Input → Notebook Output →** the version
   from session 1. Keep the original dataset attached as well (it has the
   VQA JSONs).
2. **Save Version → Save & Run All (Commit)** again.
   The script lists every `outputs/` it finds and resumes from the most
   advanced one.
3. Repeat until the log shows `needs_llm` = 0, or no longer goes down.

## Getting the results back to the PC

1. Download the final version's output (the `outputs/` folder, under
   `src/QuestionDependentCaptionGenerator/outputs/`).
2. Back up your local `outputs/`, then replace it with the downloaded folder.
3. Optional: run your usual command locally to finish any leftovers.
   It resumes from that file.

## Settings (top of `kaggle_runner.py`)

| Setting | Default | Notes |
|---|---|---|
| `RUN_HOURS` | 9.0 | Hours before Ctrl+C. Setup (about 10 min) and the final save must also fit in Kaggle's session limit. |
| `SHUTDOWN_GRACE_MIN` | 30 | Minutes to wait for the final save before killing the process. |
| `MODEL` | `qwen2.5:3b-instruct-q4_K_M` | Keep the same as the PC run so all captions are consistent. |
| `BATCH_SIZE` | 5 | Same as the PC run. |
| `LLM_PARALLEL` | 8 | Concurrent Ollama requests. Lower it if you see timeouts or out-of-memory errors in `ollama.log`. |
| `CHECKPOINT_EVERY` | 50 | Batches between saves. The output file is about 150 MB, so saving every batch is slow. |

## Notes

- **Quota:** Kaggle gives about 30 GPU hours per week. Two 9.5 h runs fit in
  one week.
- **Speed:** a T4 isn't necessarily faster than a desktop GPU. The gain comes
  from running 8 requests at once. Use the session-1 log to estimate how many
  sessions you'll need.
- **If it gets killed:** the last periodic checkpoint (at most 50 batches
  old) is still in `outputs/`, so little work is lost.
- **If the run fails:** read the cell log, and `/kaggle/working/ollama.log`
  while the session is still running (the script deletes nothing outside
  `src/`, so `ollama.log` is also in the output).
