# Running caption generation on Kaggle

[`kaggle_runner.py`](kaggle_runner.py) continues the LLM caption generation
(`generate.py --llm`) on a Kaggle GPU using Ollama. It stops cleanly before
Kaggle's session time limit, so the dataset can be finished over several
sessions. All settings and input paths are in
[`kaggle_config.json`](kaggle_config.json).

As of 2026-11-07: 406,573 rows in total, 237,259 `llm_fallback` done,
**100,947 `needs_llm` remaining**.

## What the script does

1. Reads `kaggle_config.json` from the checkout it runs from, plus an
   optional override file (`--config`).
2. Copies `QuestionDependentCaptionGenerator/` from the checkout to
   `/kaggle/working/src/`, so `generate.py` resumes from `outputs/` exactly
   as it does on the PC.
3. Reads every input path from the config. There's no searching: if a path
   is wrong, the run stops at once with an error that names it. The VQA
   questions and annotations are passed straight to `generate.py` with
   `--questions` / `--annotations` and are only read, never copied.
4. Copies the `vqa_v2_*` files from `inputs.outputs_dir` into the writable
   `/kaggle/working/src/QuestionDependentCaptionGenerator/outputs/`. See
   [How resuming works](#how-resuming-works).
5. Installs Ollama and starts **one server per GPU** (see
   [Using both T4 GPUs](#using-both-t4-gpus)). It pulls the model and
   loads it on every GPU.
6. Runs `generate.py --llm` with the settings from the config.
7. After `run.hours` it sends Ctrl+C (SIGINT). `generate.py` then writes its
   checkpoint and exits. The script waits up to `run.shutdown_grace_min` for
   that save.
8. Deletes everything except `outputs/` from
   `/kaggle/working/src/QuestionDependentCaptionGenerator`, so the notebook
   output is only the results plus the `ollama_gpu*.log` files. Prints how
   many requests each GPU served and the remaining `needs_llm` count.

## Using both T4 GPUs

A single Ollama server runs only one copy of the model, so on a T4 x2
machine the second GPU would sit idle. The runner does this instead:

```
generate.py --llm-parallel 8
        │  all requests go to ollama.host
        ▼
balancing proxy  127.0.0.1:11434   (inside kaggle_runner.py)
        │  each request goes to the server with fewer requests in progress
   ┌────┴─────────────────────┐
   ▼                          ▼
ollama serve :11435      ollama serve :11436
CUDA_VISIBLE_DEVICES=0   CUDA_VISIBLE_DEVICES=1
NUM_PARALLEL=4           NUM_PARALLEL=4
```

- Each GPU runs its own full copy of the 3B model (about 2.5 GB, which fits
  easily in 16 GB), so the two GPUs work at the same time. You get roughly
  twice the throughput of one GPU.
- `generate.py` is unchanged. It still writes one output file and one
  checkpoint, so resuming works exactly as before.
- Requests are split by load, so a slow GPU never gets a long queue. In a
  local test with two fake servers, 80 requests split exactly 40 / 40.
- At the end the log prints the split, for example
  `Load split: http://127.0.0.1:11435: 10234 requests (0 failed), …`.
  If one GPU shows 0, or many failures, check its `ollama_gpu*.log`.
- If a server returns an error or is down, the proxy passes the error back to
  `generate.py`, and its normal retry logic handles it.

## Config: `kaggle_config.json`

| Key | Default | Meaning |
|---|---|---|
| `repo.url` | this repo | Used only when the script is pasted into a cell. |
| `repo.branch` | `feat/kaggle-runner` | Same as above. |
| `repo.clone_dir` | `/tmp/repo` | Same as above. |
| `inputs.questions` | `/kaggle/input/datasets/mahdirouzmeh/questiondependentcaptions/v2_OpenEnded_mscoco_train2014_questions.json` | Absolute path to the VQA questions JSON. |
| `inputs.annotations` | `/kaggle/input/datasets/mahdirouzmeh/questiondependentcaptions/v2_mscoco_train2014_annotations.json` | Absolute path to the VQA annotations JSON. |
| `inputs.outputs_dir` | `/kaggle/input/datasets/mahdirouzmeh/questiondependentcaptions` | Folder holding `vqa_v2_question_dependent_captions_train2014.json` and the other `vqa_v2_*` files to resume from. In session 2, point it at the previous notebook's output. |
| `run.hours` | `9.0` | Hours before Ctrl+C. Setup (about 10 min) and the final save must also fit in Kaggle's session limit. |
| `run.shutdown_grace_min` | `30` | Minutes to wait for the final save before killing the process. |
| `generate.split` | `train` | `train` or `val`. Also sets the captions file name. |
| `generate.model` | `qwen2.5:3b-instruct-q4_K_M` | Keep the same as the PC run so all captions are consistent. |
| `generate.batch_size` | `5` | Same as the PC run. |
| `generate.llm_parallel` | `8` | Concurrent Ollama requests. Lower it if you see timeouts or out-of-memory errors in the `ollama_gpu*.log` files. |
| `generate.checkpoint_every` | `50` | Batches between saves. The file is about 150 MB, so saving every batch is slow. |
| `generate.extra_args` | `[]` | Extra `generate.py` flags, for example `["--min-consensus", "0.4"]`. |
| `ollama.host` | `http://127.0.0.1:11434` | Address of the load-balancing proxy. `generate.py` sends every request here (`--ollama-host`). |
| `ollama.gpus` | `"auto"` | `"auto"` starts one Ollama server per GPU found by `nvidia-smi`. Or give a list, for example `[0, 1]`, or `[0]` for one GPU. |
| `ollama.base_port` | `11435` | The server for GPU *k* listens on `base_port + k` (11435, 11436, …). |
| `ollama.num_parallel` | `null` | `OLLAMA_NUM_PARALLEL` per server. `null` means `ceil(llm_parallel / number of GPUs)`, so 8 becomes 4 + 4. |
| `ollama.keep_alive` | `"-1"` | Keeps the model loaded. |
| `paths.work_dir` | `/kaggle/working` | Where the results are written. |
| `paths.log_dir` | `/kaggle/working` | Gets one log per server: `ollama_gpu0.log`, `ollama_gpu1.log`. |

There are two ways to change a setting:

- **Permanently:** edit `kaggle_config.json`, commit, and push to
  `feat/kaggle-runner`.
- **For one notebook only:** write a small override file in the cell and pass
  it with `--config`. It only needs the keys you change:

  ```
  %%writefile /tmp/override.json
  {"run": {"hours": 8.5},
   "inputs": {"outputs_dir": "/kaggle/input/<previous-run>/src/QuestionDependentCaptionGenerator/outputs"}}
  ```
  ```
  !python -u /tmp/repo/QuestionDependentCaptionGenerator/kaggle/kaggle_runner.py --config /tmp/override.json
  ```

## Input paths

All three `inputs` paths are absolute and are used exactly as written. To see
the real paths of the inputs attached to a notebook, run this in a cell:

```
!find /kaggle/input -name "*.json*" | head -50
```

## How resuming works

`/kaggle/input` is **read-only**, so nothing is ever written back to your
dataset:

```
/kaggle/input/.../questiondependentcaptions/     (read-only, your upload)
        │  copy vqa_v2_* files at the start of the run
        ▼
/kaggle/working/src/QuestionDependentCaptionGenerator/outputs/   (writable)
        │  generate.py resumes from this copy and saves its checkpoints here
        ▼
notebook output of this version  (Kaggle keeps /kaggle/working)
        │  attach it as input to the next session
        ▼
next session: inputs.outputs_dir = that output's outputs/ folder
```

So each session reads the previous session's result and writes a new one.
Your original dataset is never changed and stays as a backup.

## One-time setup

1. **Code:** the repo is public, so you don't upload any code. The notebook
   clones it and checks out `feat/kaggle-runner`. Push any code or config
   change to that branch before you start a run.
2. **VQA JSONs:** the config expects them in your dataset
   `mahdirouzmeh/questiondependentcaptions`. If they're in a different
   dataset, put their absolute paths in `inputs.questions` /
   `inputs.annotations`.
3. **Your outputs:** the private dataset
   `mahdirouzmeh/questiondependentcaptions` holds the 5 files from
   `QuestionDependentCaptionGenerator/outputs/`. Upload them as one zip,
   `caption_outputs.zip` (22 MB instead of 291 MB); Kaggle unzips it:

   ```
   vqa_v2_question_dependent_captions_train2014.json
   vqa_v2_question_classification_result.json
   vqa_v2_question_dependent_captions_train2014_not_directly_visual.json
   vqa_v2_question_dependent_captions_train2014_validation_audit.jsonl
   vqa_v2_question_dependent_captions_train2014.json.llm_failures
   ```

4. **Verify your phone number** in Kaggle account settings. Without it you
   can't turn Internet on, and the script needs Internet to clone the repo
   and install Ollama.

## Session 1

1. Create a new notebook.
2. Right sidebar: **Accelerator → GPU T4 x2**, **Internet → On**.
3. **Add Input →** `mahdirouzmeh/questiondependentcaptions` (plus the VQA
   dataset if it's a separate one).
4. First run a **dry run** interactively. It checks every path in a few
   seconds, without Ollama:

   ```
   !git clone https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /tmp/repo
   !git -C /tmp/repo checkout feat/kaggle-runner
   !python -u /tmp/repo/QuestionDependentCaptionGenerator/kaggle/kaggle_runner.py --dry-run
   ```

   Check the printed `Questions:`, `Annotations:` and `Resuming from:` lines.
   `Resuming from:` should show about 100947 `needs_llm`. If a path is
   wrong, the error names it; fix it in the config (or with `--config`).
5. Remove `--dry-run` from the last line. The repo goes to `/tmp`, not
   `/kaggle/working`, so it isn't saved as notebook output.
6. Optional: run the cell for about 10 minutes to check that Ollama starts
   and the first batches finish. Then stop it with **Interrupt**.
7. Click **Save Version → Save & Run All (Commit)**. It runs in the
   background, so you can close the browser.
8. When it has finished (about 9.5 h), open the version and check the last
   log line, `Remaining needs_llm: …`.

## Session 2 (and any later sessions)

1. Open the same notebook → **Add Input → Notebook Output →** the version
   from session 1. Keep the other inputs attached.
2. Find where that output is mounted:

   ```
   !find /kaggle/input -name "vqa_v2_question_dependent_captions_train2014.json"
   ```

   Use the folder that is **not** your original dataset. It ends with
   `src/QuestionDependentCaptionGenerator/outputs`.
3. Point `inputs.outputs_dir` at that folder with an override cell above the
   run cell:

   ```
   %%writefile /tmp/override.json
   {"inputs": {"outputs_dir": "<folder from step 2>"}}
   ```

   Then add `--config /tmp/override.json` to the `kaggle_runner.py` line.
   Run with `--dry-run` once and check `Resuming from:`. Its `needs_llm`
   count must be lower than in session 1.
4. **Save Version → Save & Run All (Commit)**.
5. Repeat for each session, each time pointing `outputs_dir` at the newest
   version's output, until `Remaining needs_llm` is 0.

## Getting the results back to the PC

1. Download the final version's output:
   `src/QuestionDependentCaptionGenerator/outputs/`.
2. Back up your local `outputs/`, then replace it with the downloaded folder.
3. Optional: run your usual command locally to finish any leftovers.
   It resumes from that file.

## Notes

- **Quota:** Kaggle gives about 30 GPU hours per week. Two 9.5 h runs fit in
  one week.
- **Speed:** a T4 isn't necessarily faster than a desktop GPU. The gain comes
  from running 8 requests at once. Use the session-1 log to estimate how many
  sessions you'll need.
- **If it gets killed:** the last periodic checkpoint (at most
  `checkpoint_every` batches old) is still in `outputs/`.
- **If the run fails:** read the cell log and the `ollama_gpu*.log` files in
  the output.

## Re-checking the question classifier (`kaggle_reclassify_runner.py`)

The first classification run kept every non-blacklisted question as
`DIRECTLY_VISUAL`: Ollama couldn't be reached, so every LLM call failed and
was counted as `PARSE_FAIL_KEEP` (406,573 of 406,573). The
[`reclassify_captions.py`](../reclassify_captions.py) script asks the LLM again for every
row of the finished captions file. It uses the same prompt and few-shots
(`question_classifier.py`, prompt `v20`):

- `DIRECTLY_VISUAL` rows stay in `vqa_v2_question_dependent_captions_train2014.json`
  (captions unchanged).
- `NOT_DIRECTLY_VISUAL` rows are removed and written to
  `vqa_v2_question_dependent_captions_train2014_not_directly_visual_reclassified.json`
  with their reason (`NEEDS_OCR` / `NEEDS_KNOWLEDGE` / `NEEDS_OPINION`).
- `vqa_v2_question_classification_result.json` is updated the same way.
  This is needed: `generate.py` only resumes when the captions row count
  matches that file's `post_filter_count`. Otherwise it rebuilds every row.

A failed request is never counted as a label this time. Connection, HTTP
and timeout errors are retried 3 times, and a row that still fails stays
unclassified for the next run. The script checks Ollama before it starts and
stops after 5 batches in a row with no reply.

Settings are in [`kaggle_reclassify_config.json`](kaggle_reclassify_config.json): 2 GPUs
(`"gpus": "auto"`), `llm_parallel` 8 (4 per GPU), `batch_size` 10, model
`qwen2.5:3b-instruct-q4_K_M`, `run.hours` 11.

`run.hours` counts from when the runner starts, so the Ollama install and
model load (about 10 minutes) are included. At 11 h it sends Ctrl+C, the
script finishes the requests in flight and exits, and about 50 minutes are
left before Kaggle's 12-hour limit.

`batch_size` is the number of questions saved to the progress file at once.
Each question is its own Ollama request, as in the first run. A window of
16 requests stays in flight, so both GPUs stay busy between batches.

**Session 1.** Upload the finished
`vqa_v2_question_dependent_captions_train2014.json` and
`vqa_v2_question_classification_result.json` to the dataset, and check the two
`inputs` paths. Then:

```
!git clone https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /tmp/repo
!git -C /tmp/repo checkout feat/kaggle-runner
!python -u /tmp/repo/QuestionDependentCaptionGenerator/kaggle/kaggle_reclassify_runner.py --dry-run
```

Remove `--dry-run` and use **Save & Run All (Commit)**. Output goes to
`/kaggle/working/outputs/`.

**If it isn't finished** (the log ends with `NOT finished`), attach that
version's output and continue from its progress file:

```
%%writefile /tmp/override.json
{"inputs": {"resume_dir": "/kaggle/input/<previous-run>/outputs"}}
```

Then add `--config /tmp/override.json` to the runner line. The final files are
written only when every row is classified (`DONE` in the log). Copy the three
JSONs into the local `outputs/` folder (back it up first).

To run it on the PC instead:

```
python reclassify_captions.py --input outputs/vqa_v2_question_dependent_captions_train2014.json --classification-result outputs/vqa_v2_question_classification_result.json --output-dir outputs_reclassified --llm-parallel 4 --batch-size 10
```

## Finding and regenerating wrong captions (`kaggle_judge_runner.py`)

[`kaggle_judge_runner.py`](kaggle_judge_runner.py) judges every caption with
`qwen2.5:7b-instruct-q4_K_M` and regenerates the ones whose meaning is wrong.
See [`../caption_errors/README.md`](../caption_errors/README.md) for the fields
it writes. All settings are in
[`kaggle_judge_config.json`](kaggle_judge_config.json).

Notebook (GPU T4 x2, Internet ON), one cell, then **Save Version → Save & Run All (Commit)**:

```
!git clone https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /tmp/repo
!git -C /tmp/repo checkout feat/kaggle-runner
!python -u /tmp/repo/QuestionDependentCaptionGenerator/kaggle/kaggle_judge_runner.py
```

### Step 1: calibrate (`run.mode = "calibrate"`, the default)

Judges only the 298 hand-labelled rows in
`audit/DevAuditor/caption_error_audit_manual.json` (part of the repo, nothing
to attach). Prints WRONG precision / recall and every disagreement. Takes a
few minutes. Switch to `full` only when the target (recall ≥ 0.90,
precision ≥ 0.80) is met.

### Step 2: full run (`run.mode = "full"`)

Attach the reclassified captions JSON as a dataset and set `inputs.captions`
to its path (`!find /kaggle/input -name "*.json"` shows it).
`inputs.expected_rows` (283707) stops the run if the wrong file is attached.

```
%%writefile /tmp/override.json
{"run": {"mode": "full", "hours": 5}}
```
```
!python -u /tmp/repo/QuestionDependentCaptionGenerator/kaggle/kaggle_judge_runner.py --config /tmp/override.json
```

### Start with the "Need to Manual validate" rows (optional)

```
%%writefile /tmp/override.json
{"run": {"mode": "full", "scope": "manual", "hours": 2}}
```

These ~7.5k rows are ~73% wrong (hand-labelled sample) and take about 1 h.
They hold only ~23% of all wrong captions (~5.5k of ~23.5k); the rest sit in
`Ready to Use` rows. When the log says `DONE`, continue with `"scope": "all"`
and `inputs.resume_dir` set to that output. Rows already judged are skipped.

### Time limit

`run.hours` counts from when the runner starts, setup included. The job stops
starting new batches 5 min before the limit, finishes what is in flight and
saves. At the limit it gets Ctrl+C, and after `run.shutdown_grace_min` it is
killed. Lower `run.hours` to save weekly GPU quota; nothing finished is lost.

### Both GPUs

As above: one Ollama server per T4, each with one copy of the model and
`ceil(llm_parallel / 2)` parallel slots (default `llm_parallel` 8 → 4 per GPU).
Parallel slots share the model weights, so this is better than loading the
model twice on one GPU. If both GPUs have memory left (`ollama_gpu*.log`, or
`!nvidia-smi` in another notebook), try `llm_parallel` 12.

### Every session ends with a captions JSON

`/kaggle/working/outputs/vqa_v2_question_dependent_captions_train2014.json`
holds every row:
- rows judged so far have `caption_quality_verdict` / `verdict_reason`, and the
  new caption if they were regenerated;
- rows not reached yet have `caption_quality_verdict: null`.

`info.caption_error_fix.progress.complete` is `true` only in the last session.

Next to it, `vqa_v2_question_dependent_captions_train2014_suspicious.json`
lists every row whose final caption the code check marks as possibly wrong
(`suspicious_reasons`). This is the input for a later pass with a bigger
model. Use the file from the last session.

### Next session (resume)

1. From the previous session's **Output** tab, download
   `outputs/vqa_v2_question_dependent_captions_train2014_caption_check_progress.jsonl`.
2. Add that file to your dataset (**New Version**). Leave the original
   captions JSON in the dataset **unchanged**, because the run always starts
   from it. Never replace it with a session's output; the runner refuses an
   output file as `inputs.captions`.
3. Set `inputs.resume_progress` in `kaggle_judge_config.json` to the file's
   path in the dataset (check with
   `!find /kaggle/input -name "*caption_check_progress.jsonl"`), commit and
   push. Or set it for one notebook only:

   ```
   %%writefile /tmp/override.json
   {"inputs": {"resume_progress": "/kaggle/input/datasets/<you>/<dataset>/vqa_v2_question_dependent_captions_train2014_caption_check_progress.jsonl"}}
   ```
4. Run the notebook. The log should say `Resuming from … (N progress lines)`.

Instead of the dataset, you can also attach the previous notebook's output
(**Add Input → Notebook output**) and set `inputs.resume_dir` to its
`outputs` folder. If both are set, `resume_progress` wins.

Each row is judged once. A run with a different model or prompt version
refuses the progress file instead of mixing results. The end of the log says
how many rows are left and the ETA.

| Key | Default | Meaning |
|---|---|---|
| `inputs.captions` | dataset path | Reclassified captions JSON (input of the full run). |
| `inputs.expected_rows` | `283707` | Safety check on the row count; `null` to skip. |
| `inputs.audit` | `audit/DevAuditor/caption_error_audit_manual.json` | Gold set for `calibrate` (relative to `QuestionDependentCaptionGenerator/`). |
| `inputs.resume_progress` | dataset path | The previous session's `*_caption_check_progress.jsonl` (e.g. uploaded to your dataset). `null` for a fresh start. |
| `inputs.resume_dir` | `null` | Alternative: the folder holding that file (e.g. a notebook output). |
| `run.mode` | `"calibrate"` | `"calibrate"` or `"full"`. |
| `run.scope` | `"all"` | Full mode only. `"all"`: every row. `"manual"`: only rows whose `caption_status` is `Need to Manual validate` in the input (~7.5k rows, ~1 h). A later `"all"` run with `resume_dir` skips rows already judged. |
| `run.hours` | `11.0` | Max hours from runner start. |
| `run.shutdown_grace_min` | `10` | Minutes after Ctrl+C before the job is killed. |
| `judge.model` | `qwen2.5:7b-instruct-q4_K_M` | Judge and regeneration model (~4.7 GB per GPU). |
| `judge.llm_parallel` | `8` | Batches in flight in total (split over the GPUs). |
| `judge.batch_size` | `10` | Captions per judge request. |
| `judge.extra_args` | `[]` | Extra `judge_captions.py` flags. |
| `regen.batch_size` | `5` | WRONG captions per regeneration request (packed like generation). |
| `regen.rounds` | `2` | Regeneration attempts per WRONG caption; still WRONG after that → `Need to Manual validate`. |
| `regen.temperature` | `0.3` | Sampling temperature for regeneration (judge always uses 0). |
| `regen.single_fallback` | `true` | Redo a caption one by one when it fails the checks against its pack (batch contamination, polarity, spurious negation, format). |

Testing other regeneration settings in one notebook, for example:

```
%%writefile /tmp/override.json
{"run": {"mode": "full", "hours": 5}, "regen": {"batch_size": 3, "temperature": 0.2}}
```

You can change `regen.batch_size`, `regen.temperature` and
`regen.single_fallback` between sessions. The model, the prompt versions and
`regen.rounds` must stay the same, or the run refuses the progress file.
