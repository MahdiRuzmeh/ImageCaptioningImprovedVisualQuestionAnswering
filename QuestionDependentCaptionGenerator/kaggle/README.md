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
3. Finds the VQA questions and annotations files (see
   [Input paths](#input-paths)). It passes them straight to `generate.py`
   with `--questions` / `--annotations`, so they're never copied.
4. Finds every copy of the captions JSON among the inputs (your upload,
   earlier notebook versions) and resumes from the one with the **fewest
   `needs_llm` rows**. Only the `vqa_v2_*` files are copied into `outputs/`.
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
| `inputs.search_root` | `/kaggle/input` | Where attached Kaggle inputs are searched. |
| `inputs.questions` | `v2_OpenEnded_mscoco_train2014_questions.json` | A bare file name or an absolute path; see [Input paths](#input-paths). |
| `inputs.annotations` | `v2_mscoco_train2014_annotations.json` | Same as above. |
| `inputs.outputs_dirs` | `["/kaggle/input/datasets/mahdirouzmeh/questiondependentcaptions"]` | Folders that hold `vqa_v2_question_dependent_captions_train2014.json` (your uploaded outputs). The run stops with an error if a listed folder doesn't have it. |
| `inputs.outputs_search` | `true` | Also search `search_root` for other copies of the captions file, such as a previous notebook version you attached in session 2. The copy with the fewest `needs_llm` rows wins. Set to `false` to use only `outputs_dirs`. |
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
   "inputs": {"questions": "/kaggle/input/vqa-v2/v2_OpenEnded_mscoco_train2014_questions.json"}}
  ```
  ```
  !python -u /tmp/repo/QuestionDependentCaptionGenerator/kaggle/kaggle_runner.py --config /tmp/override.json
  ```

## Input paths

`inputs.questions` and `inputs.annotations` can each be set in two ways:

- **A bare file name** (the default). The script searches for that name at
  any depth under `/kaggle/input`. Just attach any Kaggle dataset that has
  the file, for example a public VQA v2 dataset already on Kaggle.
- **An absolute path**, for example
  `/kaggle/input/vqa-v2/v2_mscoco_train2014_annotations.json`. Use this when
  the dataset uses a different file name, or has several copies. The script
  stops with an error if the path doesn't exist.

To find the exact paths of an attached dataset, run this in a cell:

```
!find /kaggle/input -name "*.json" | head -50
```

## One-time setup

1. **Code:** the repo is public, so you don't upload any code. The notebook
   clones it and checks out `feat/kaggle-runner`. Push any code or config
   change to that branch before you start a run.
2. **VQA JSONs:** attach the Kaggle dataset that already has
   `v2_OpenEnded_mscoco_train2014_questions.json` and
   `v2_mscoco_train2014_annotations.json`. If its file names are different,
   set their absolute paths in the config.
3. **Your outputs:** create a private Kaggle Dataset (for example
   `vqa-caption-gen`) and upload the 5 files from
   `QuestionDependentCaptionGenerator/outputs/`, flat, with no zip:

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
3. **Add Input →** the VQA dataset and your outputs dataset.
4. First run a **dry run** interactively. It checks every path in a few
   seconds, without Ollama:

   ```
   !git clone https://github.com/MahdiRuzmeh/ImageCaptioningImprovedVisualQuestionAnswering.git /tmp/repo
   !git -C /tmp/repo checkout feat/kaggle-runner
   !python -u /tmp/repo/QuestionDependentCaptionGenerator/kaggle/kaggle_runner.py --dry-run
   ```

   Check the printed `Questions:`, `Annotations:` and `Resuming from:` lines.
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
   from session 1. Keep the other two datasets attached.
2. **Save Version → Save & Run All (Commit)** again. The log lists every
   copy of the outputs it found, with its `needs_llm` count, and resumes from
   the most advanced one.
3. Repeat until `Remaining needs_llm` is 0, or no longer goes down.

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
