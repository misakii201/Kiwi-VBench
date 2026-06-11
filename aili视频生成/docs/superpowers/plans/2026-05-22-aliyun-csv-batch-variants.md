# Aliyun CSV Batch Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the main CSV automation workflow: 100 prompts -> Aliyun 1:1 masters -> 9 ratios x 2 quality tiers -> `results.csv`, plus a one-run sample script that saves 18 test records.

**Architecture:** Keep the existing Aliyun client/scheduler/downloader core. Add `aliyun_video.batch_variants` for CSV jobs, variant matrix, orchestration, and CSV results. Add `aliyun_video.variants` for reusable ffmpeg crop derivation. Add CLI scripts for production batch and the one-run 18-row sample.

**Tech Stack:** Python standard library `csv/json/pathlib/dataclasses`, existing `imageio_ffmpeg`, existing Aliyun modules, `unittest`.

---

### Task 1: CSV Prompt Jobs and Variant Matrix

**Files:**
- Create: `aliyun_video/batch_variants.py`
- Create: `aliyun_video/variants.py`
- Test: `tests/test_batch_variants.py`

- [ ] **Step 1: Write failing tests**

Test `load_prompt_csv` parses `id,prompt,duration,watermark,seed`. Test `build_variant_matrix` returns exactly 18 outputs for 9 ratios x `1080P/720P`.

- [ ] **Step 2: Run failing tests**

Run: `python -m unittest tests.test_batch_variants -v`

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement minimal modules**

Move reusable crop concepts from `scripts/derive_crop_variants.py` into `aliyun_video.variants`, including 1080P and 720P matrices.

- [ ] **Step 4: Verify green**

Run: `python -m unittest tests.test_batch_variants -v`

Expected: PASS.

### Task 2: Orchestration and Results CSV

**Files:**
- Modify: `aliyun_video/batch_variants.py`
- Modify: `tests/test_batch_variants.py`

- [ ] **Step 1: Write failing tests**

Use fake submit/download/derive callables to verify one successful prompt writes 18 `results.csv` rows and failed master generation writes one failed row.

- [ ] **Step 2: Run failing tests**

Run: `python -m unittest tests.test_batch_variants -v`

Expected: FAIL because orchestration function is missing.

- [ ] **Step 3: Implement orchestration**

Add `run_batch_variant_workflow(...)` with injectable callables for API generation and ffmpeg derivation.

- [ ] **Step 4: Verify green**

Run: `python -m unittest tests.test_batch_variants -v`

Expected: PASS.

### Task 3: CLI Scripts

**Files:**
- Create: `scripts/generate_batch_variants.py`
- Create: `scripts/write_18_row_sample.py`
- Modify: `tests/test_batch_variants.py`

- [ ] **Step 1: Write failing CLI tests**

Verify production CLI `--dry-run` prints 18 planned outputs for one prompt and no API calls. Verify sample script writes a CSV with exactly 18 result rows.

- [ ] **Step 2: Run failing tests**

Run: `python -m unittest tests.test_batch_variants -v`

Expected: FAIL because scripts are missing.

- [ ] **Step 3: Implement scripts**

`generate_batch_variants.py` supports `--csv`, `--output-dir`, `--dry-run`, and real execution. `write_18_row_sample.py` writes a self-contained sample under `outputs/sample_18_rows`.

- [ ] **Step 4: Verify green**

Run: `python -m unittest tests.test_batch_variants -v`

Expected: PASS.

### Task 4: Cleanup and Docs

**Files:**
- Delete: `aliyun_video/prompt_latents.py`
- Delete: `scripts/generate_prompt_latents.py`
- Delete: `tests/test_prompt_latents.py`
- Delete: `docs/superpowers/specs/2026-05-22-csv-prompt-latents-design.md`
- Delete: `docs/superpowers/plans/2026-05-22-csv-prompt-latents.md`
- Modify: `README.md`

- [ ] **Step 1: Remove latent-only files**

Delete the temporary CSV latent layer from the previous direction.

- [ ] **Step 2: Update README**

Document the new main CSV workflow and the 18-row sample script.

- [ ] **Step 3: Full verification**

Run: `python -m unittest discover -s tests -v`

Expected: PASS.
