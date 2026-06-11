#!/usr/bin/env python3
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

TARGET = "/kwkj-k8s/cy123/daVinci-MagiHuman-main/daVinci-MagiHuman-main/t5"
TOKEN = os.environ.get("HF_TOKEN", "")

path = snapshot_download(
    repo_id="google/t5gemma-9b-9b-ul2",
    local_dir=TARGET,
    token=TOKEN,
    resume_download=True,
    max_workers=2,
)
print("Download complete:", path)
