# Aliyun CSV Batch Video Variant Generator

Python workflow for reading prompt rows from CSV, generating `1080P` `1:1` master videos with Aliyun Model Studio HappyHorse, and deriving 9 aspect ratios in both `1080P` and `720P`.

## Setup

Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
$env:DASHSCOPE_API_KEY="your_api_key_here"
```

Linux server:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export DASHSCOPE_API_KEY="your_api_key_here"
```

Generated videos and run outputs are written under `outputs/`. That directory is ignored by git and should be created on the server during runs, not uploaded with the source code.

## Input CSV

Create a prompt CSV:

```csv
id,prompt,duration,watermark,seed
001,"cinematic basketball fast break, centered subject, enough empty space for crop variants",5,false,123
```

An example file is provided at `configs/prompts.example.csv`.

Required columns:

- `id`
- `prompt`

Optional columns:

- `duration`, default `5`
- `watermark`, default `false`
- `seed`

## Main Dry Run

Preview all derived output rows without calling Aliyun or ffmpeg:

```powershell
python scripts/generate_batch_variants.py --csv prompts.csv --dry-run
```

Linux:

```bash
python scripts/generate_batch_variants.py --csv prompts.csv --dry-run
```

For 100 prompt rows, the dry-run prints 1800 planned outputs: `100 * 9 ratios * 2 quality tiers`.

## Generate Batch Variants

```powershell
python scripts/generate_batch_variants.py --csv prompts.csv --output-dir outputs/batch_001
```

Linux:

```bash
python scripts/generate_batch_variants.py --csv prompts.csv --output-dir outputs/batch_001
```

The workflow:

1. Submits one `1080P` `1:1` master task to Aliyun per prompt.
2. Downloads successful master MP4 files to `masters/`.
3. Derives 9 aspect ratios for each master.
4. Saves each ratio in `1080P` and `720P`.
5. Writes one final row per derived MP4 to `results.csv`.

Output layout:

```text
outputs/batch_001/
  masters/
    001_1080P_1x1.mp4
  variants/
    001_16x9_1080P.mp4
    001_16x9_720P.mp4
    ...
  api_tasks.jsonl
  results.csv
```

`results.csv` includes prompt id, prompt text, task id, master path, variant path, ratio, quality, width, height, duration, seed, status, and error.

## One-Run 18-Row Sample

Write one sample prompt's 9 ratios x 2 quality rows without calling Aliyun or ffmpeg:

```powershell
python scripts/write_18_row_sample.py --output-dir outputs/sample_18_rows
```

This creates placeholder MP4 files and `outputs/sample_18_rows/results.csv` with exactly 18 rows.

## Legacy Utilities

The older YAML utility still exists for low-level Aliyun task experiments:

```powershell
python scripts/generate_videos.py --config configs/jobs.basketball.example.yaml --dry-run
```

The standalone crop utility derives 1080P variants from an existing master:

```powershell
python scripts/derive_crop_variants.py --input master.mp4 --output-dir outputs/crops --prefix sample
```

## Notes

- Query polling is globally limited by `rps` and defaults to `20`, matching the HappyHorse task query default limit.
- Successful `video_url` links are downloaded immediately because Aliyun returns links with a limited validity window.
- For `eu-central-1`, add `--workspace-id`.
- Run `python -m unittest discover -s tests -v` after deployment if you want to verify the server environment before a real batch.
