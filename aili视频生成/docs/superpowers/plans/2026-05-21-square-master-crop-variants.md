# Square Master Crop Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one square basketball master video and derive nine aspect-ratio variants from the same source.

**Architecture:** Reuse the existing Aliyun generation CLI for the master video. Add a small FFmpeg-based derivation script that computes center-crop filters for each target ratio and scales the result to the required output size.

**Tech Stack:** Python 3, existing HappyHorse framework, `imageio-ffmpeg` for FFmpeg discovery, standard-library `unittest`.

---

### Task 1: Square Master Config

**Files:**
- Create: `configs/jobs.basketball.square_master.yaml`

- [ ] Create a config with one `1080P` `1:1` basketball job using the centered small-subject prompt.
- [ ] Run `python scripts/generate_videos.py --config configs/jobs.basketball.square_master.yaml --dry-run` and verify it prints one `1:1` task.

### Task 2: FFmpeg Variant Deriver

**Files:**
- Create: `scripts/derive_crop_variants.py`
- Create: `tests/test_crop_variants.py`

- [ ] Write failing tests for crop filter generation for `16:9`, `9:16`, and `1:1`.
- [ ] Implement variant definitions and `build_filter`.
- [ ] Run `python -m unittest tests.test_crop_variants -v` and verify it passes.

### Task 3: Execute Experiment

**Files:**
- Output: `outputs/square_crop_experiment/*.mp4`

- [ ] Generate the `1:1` master through Aliyun.
- [ ] Run `python scripts/derive_crop_variants.py --input outputs/basketball_square_master_1080P_1x1.mp4 --output-dir outputs/square_crop_experiment`.
- [ ] Verify the nine generated MP4 files exist and FFmpeg reports the intended dimensions.

