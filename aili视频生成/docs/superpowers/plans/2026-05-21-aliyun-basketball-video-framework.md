# Aliyun Basketball Video Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python command-line framework that batch-generates basketball videos through Aliyun HappyHorse, supports multiple resolution/aspect variants, and keeps task query traffic within RPS=20.

**Architecture:** The framework uses dataclasses for configuration and task records, a small HTTP client for HappyHorse create/query calls, a scheduler for variant expansion and polling, and a downloader for successful MP4 URLs. The CLI wires these modules together around a YAML configuration file.

**Tech Stack:** Python 3, standard-library `urllib`, standard-library `unittest`, optional `PyYAML` for `.yaml` config files.

---

### Task 1: Project Skeleton And Models

**Files:**
- Create: `src/aliyun_video/__init__.py`
- Create: `src/aliyun_video/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for task variant naming and defaults**

```python
from aliyun_video.models import PromptJob, VideoVariant, expand_jobs


def test_expand_jobs_creates_stable_basketball_variant_ids():
    jobs = [PromptJob(id="dunk", prompt="篮球运动员扣篮", duration=5)]
    variants = [
        VideoVariant(resolution="720P", ratio="16:9"),
        VideoVariant(resolution="1080P", ratio="9:16"),
    ]

    expanded = expand_jobs(jobs, variants)

    assert [task.id for task in expanded] == ["dunk_720P_16x9", "dunk_1080P_9x16"]
    assert expanded[0].prompt == "篮球运动员扣篮"
    assert expanded[1].duration == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_models -v`
Expected: import failure because `aliyun_video.models` does not exist.

- [ ] **Step 3: Implement dataclasses and `expand_jobs`**

Create `VideoVariant`, `PromptJob`, `VideoTask`, `TaskRecord`, and `expand_jobs`. Validate resolution, ratio, duration, and prompt presence in dataclass `__post_init__` methods.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_models -v`
Expected: pass.

### Task 2: Config Loader

**Files:**
- Create: `src/aliyun_video/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for config parsing and endpoint validation**

```python
import os
import tempfile
import unittest
from pathlib import Path

from aliyun_video.config import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_basketball_config_from_json_compatible_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.yaml"
            path.write_text(
                """
region: cn-beijing
model: happyhorse-1.0-t2v
rps: 20
poll_interval_seconds: 15
output_dir: outputs
variants:
  - resolution: 720P
    ratio: "16:9"
jobs:
  - id: fast_break
    prompt: "篮球快攻上篮，电影感镜头"
    duration: 5
    watermark: false
""".strip(),
                encoding="utf-8",
            )

            config = load_config(path, env={"DASHSCOPE_API_KEY": "sk-test"})

        self.assertEqual(config.api_key, "sk-test")
        self.assertEqual(config.create_url, "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis")
        self.assertEqual(config.query_base_url, "https://dashscope.aliyuncs.com/api/v1/tasks")
        self.assertEqual(config.rps, 20)
        self.assertEqual(config.jobs[0].id, "fast_break")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_config -v`
Expected: import failure because `aliyun_video.config` does not exist.

- [ ] **Step 3: Implement `load_config`**

Load YAML with `yaml.safe_load`, read API key from config or `DASHSCOPE_API_KEY`, select region endpoints, and return an `AppConfig` dataclass.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_config -v`
Expected: pass.

### Task 3: HTTP Client And Scheduler

**Files:**
- Create: `src/aliyun_video/client.py`
- Create: `src/aliyun_video/scheduler.py`
- Create: `tests/test_client_scheduler.py`

- [ ] **Step 1: Write failing tests for request payloads and RPS limiter**

Test that `HappyHorseClient.create_task` sends the async header, model, prompt, resolution, ratio, duration, watermark, and seed. Test that `RateLimiter.acquire` sleeps when more than 20 acquisitions are attempted inside one second.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_client_scheduler -v`
Expected: import failure.

- [ ] **Step 3: Implement client and scheduler units**

Use injectable transport functions so tests do not hit the network. Implement `RateLimiter`, `TaskRunner.submit_tasks`, and `TaskRunner.poll_until_complete`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_client_scheduler -v`
Expected: pass.

### Task 4: Downloader, CLI, Docs, And Example

**Files:**
- Create: `src/aliyun_video/downloader.py`
- Create: `scripts/generate_videos.py`
- Create: `configs/jobs.basketball.example.yaml`
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `README.md`
- Create: `tests/test_downloader.py`

- [ ] **Step 1: Write failing downloader test**

Test that the downloader writes a deterministic MP4 filename containing job id, resolution, and ratio.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_downloader -v`
Expected: import failure.

- [ ] **Step 3: Implement downloader, CLI, and documentation**

The CLI should accept `--config`, `--dry-run`, and `--max-wait-seconds`. In dry-run mode it prints the expanded basketball tasks without calling the API.

- [ ] **Step 4: Run full verification**

Run: `python -m unittest discover -s tests -v`
Run: `python scripts/generate_videos.py --config configs/jobs.basketball.example.yaml --dry-run`
Expected: tests pass and dry-run prints expanded basketball tasks.

