# Aliyun CSV Batch Variants Design

## Goal

Build one automated workflow for a CSV of prompt rows:

1. Read 100 prompt rows from CSV.
2. Submit each prompt to Aliyun HappyHorse as a `1080P` `1:1` master video.
3. Download successful master videos.
4. Derive 9 aspect-ratio variants for each master.
5. Produce two quality tiers for each variant: `1080P` and `720P`.
6. Save one final CSV row per derived output video, yielding up to `100 * 9 * 2 = 1800` records.

## Input

The primary input is a CSV file with required columns:

- `id`: stable prompt id used in output filenames.
- `prompt`: prompt text sent to Aliyun.

Optional columns:

- `duration`: seconds, default `5`.
- `watermark`: boolean, default `false`.
- `seed`: optional Aliyun seed.

Example:

```csv
id,prompt,duration,watermark,seed
001,"cinematic basketball fast break, centered subject, enough empty space for crops",5,false,123
```

## Output Layout

Each run writes to one output root, defaulting to `outputs/batch_<timestamp>`:

```text
outputs/batch_<timestamp>/
  masters/
    001_1080P_1x1.mp4
  variants/
    001_16x9_1080P.mp4
    001_16x9_720P.mp4
    ...
  api_tasks.jsonl
  results.csv
```

`api_tasks.jsonl` records Aliyun submission and polling results for audit/debugging.

`results.csv` records final outputs. Successful rows include:

- `prompt_id`
- `prompt`
- `status`
- `aliyun_task_id`
- `master_video`
- `variant_video`
- `ratio`
- `quality`
- `width`
- `height`
- `duration`
- `seed`
- `error`

Failed prompt rows are also recorded with `status=FAILED` and an `error` message. A failed master generation does not create variant rows.

## Variant Matrix

The workflow derives the same 9 ratios already used by `scripts/derive_crop_variants.py`:

- `16x9`
- `9x16`
- `1x1`
- `4x3`
- `3x4`
- `4x5`
- `5x4`
- `9x21`
- `21x9`

For each ratio, two output quality tiers are generated:

- `1080P`: existing target dimensions such as `1920x1080`, `1080x1920`, `1080x1080`.
- `720P`: proportional 720-height/720-width equivalent such as `1280x720`, `720x1280`, `720x720`.

## Architecture

The implementation should keep the existing Aliyun core and refactor crop logic into reusable functions:

- `aliyun_video.client`: keep API submit/query behavior.
- `aliyun_video.scheduler`: keep polling and rate limiting.
- `aliyun_video.downloader`: keep master video download behavior.
- `aliyun_video.models`: keep/extend task models for CSV prompt jobs.
- `aliyun_video.variants`: new reusable crop/variant module extracted from `scripts/derive_crop_variants.py`.
- `scripts/generate_batch_variants.py`: new end-to-end CLI entrypoint.

The old YAML-based `scripts/generate_videos.py` may remain as a low-level legacy utility, but the README should present `generate_batch_variants.py` as the main workflow.

## Redundancy Cleanup

The CSV latent feature added earlier is outside this Aliyun batch-video workflow. To avoid confusing future runs, remove:

- `aliyun_video/prompt_latents.py`
- `scripts/generate_prompt_latents.py`
- `tests/test_prompt_latents.py`
- CSV latent README section
- CSV latent spec/plan docs

Keep historical generated videos and existing test fixtures untouched.

## Error Handling

The workflow should fail fast on malformed CSV headers or invalid prompt rows. During generation, per-prompt Aliyun failures are recorded in `results.csv`; successful prompts continue through download and variant derivation.

If ffmpeg variant derivation fails for one prompt, record failed variant rows for that prompt and continue with the next prompt.

## Testing

Tests should cover:

- CSV prompt loading and validation.
- Building exactly one `1080P` `1:1` Aliyun master task per prompt.
- Building the 9-ratio x 2-quality variant matrix.
- Writing `results.csv` with one row per successful derived variant.
- Recording failed prompt rows when master generation fails.
- CLI dry-run shows the derived plan without calling Aliyun or ffmpeg.
