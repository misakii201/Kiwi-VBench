# Aliyun Basketball Video Framework Design

## Goal

Build a Python framework for generating basketball-themed videos with Aliyun Model Studio HappyHorse text-to-video. The first version supports batch prompts, multiple resolution and aspect-ratio variants, asynchronous task polling, query RPS control at 20, and immediate local download of successful MP4 results.

## API Facts

- Create endpoint: `POST https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis` for Beijing.
- Required headers: `Authorization: Bearer <api key>`, `Content-Type: application/json`, `X-DashScope-Async: enable`.
- Model: `happyhorse-1.0-t2v`.
- Parameters: `resolution` is `720P` or `1080P`; `ratio` includes `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `4:5`, `5:4`, `9:21`, `21:9`; `duration` is an integer from 3 to 15 seconds.
- Query endpoint: `GET /api/v1/tasks/{task_id}`.
- Query API default RPS is 20.
- Successful results contain a `video_url` that is valid for 24 hours, so the framework downloads videos immediately.

## Architecture

The package is split into small modules. `client.py` owns HTTP calls. `config.py` loads and validates job configuration. `scheduler.py` expands basketball prompt jobs into variant tasks, submits them, polls statuses with a global rate limiter, and records results. `downloader.py` downloads successful videos. `models.py` contains dataclasses shared across modules.

The command line entry point `scripts/generate_videos.py` reads a YAML config, loads the API key from `DASHSCOPE_API_KEY` unless explicitly configured, submits tasks, polls them, downloads videos, and writes JSONL status records.

## First Version Scope

- Include a basketball-focused example config.
- Support Beijing, Singapore, US Virginia, and Germany endpoint selection, with Germany requiring `workspace_id`.
- Keep dependencies light: Python standard library for HTTP/downloads, `PyYAML` for YAML config.
- Test config validation, task expansion, rate limiting, endpoint selection, and downloader file naming.

