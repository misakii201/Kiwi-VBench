import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
 
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
 
os.environ.setdefault("DAVINCI_DISABLE_MAGI_COMPILE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
 
from inference.common import EvaluationConfig, parse_config
from inference.infra import initialize_infra
from inference.infra.checkpoint import load_model_checkpoint
from inference.infra.distributed import get_dp_rank
from inference.model.dit import DiTModel
from inference.pipeline.data_proxy import MagiDataProxy
from inference.pipeline.prompt_process import get_padded_t5_gemma_embedding
from inference.utils import print_rank_0
 
 
@dataclass
class TrainBatch:
    video_x0: torch.Tensor
    audio_x0: torch.Tensor
    audio_len: List[int]
    prompts: List[str]
    txt_feat: Optional[torch.Tensor] = None
    txt_feat_lens: Optional[List[int]] = None
 
 
@dataclass
class TrainEvalInput:
    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: List[int]
    txt_feat: torch.Tensor
    txt_feat_len: List[int]
 
 
class RandomLatentDataset(Dataset):
    def __init__(
        self,
        *,
        length: int,
        z_dim: int,
        latent_frames: int,
        latent_h: int,
        latent_w: int,
        audio_frames: int,
        device: str = "cpu",
    ):
        self.length = length
        self.z_dim = z_dim
        self.latent_frames = latent_frames
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.audio_frames = audio_frames
        self.device = device
 
    def __len__(self):
        return self.length
 
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video = torch.randn(self.z_dim, self.latent_frames, self.latent_h, self.latent_w, device=self.device)
        audio = torch.randn(self.audio_frames, 64, device=self.device)
        return {"video_x0": video, "audio_x0": audio, "audio_len": self.audio_frames, "prompt": "dummy"}
 
 
class LatentManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        *,
        temporal_crop_latent_frames: Optional[int] = None,
        expected_latent_h: Optional[int] = None,
        expected_latent_w: Optional[int] = None,
    ):
        self.manifest_path = manifest_path
        self.items: List[Dict[str, Any]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))
        self._manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        self.temporal_crop_latent_frames = temporal_crop_latent_frames
        self.expected_latent_h = int(expected_latent_h) if expected_latent_h is not None else None
        self.expected_latent_w = int(expected_latent_w) if expected_latent_w is not None else None
 
    def __len__(self):
        return len(self.items)
 
    def _load_tensor(self, path: str) -> torch.Tensor:
        if not os.path.isabs(path):
            path = os.path.join(self._manifest_dir, path)
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict) and "tensor" in obj and isinstance(obj["tensor"], torch.Tensor):
            return obj["tensor"]
        raise ValueError(f"Unsupported tensor file: {path}")

    def _load_txt_feat(self, item: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        txt_path = item.get("txt_latent_path")
        if not txt_path:
            return None, None
        feat = self._load_tensor(str(txt_path))
        if feat.dim() == 3 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        if feat.dim() != 2:
            raise ValueError(f"txt_latent_path must store (L,D) or (1,L,D): {txt_path}")
        txt_len = int(item.get("txt_feat_len", feat.shape[0]))
        return feat.to(torch.float32), txt_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        video = self._load_tensor(item["video_latent_path"])
        audio = self._load_tensor(item["audio_latent_path"])
        prompt = item.get("prompt", "")
        txt_feat, txt_feat_len = self._load_txt_feat(item)
 
        if video.dim() == 5:
            if video.shape[0] != 1:
                raise ValueError("video_latent_path must store either (C,T,H,W) or (1,C,T,H,W)")
            video = video[0]
        if video.dim() != 4:
            raise ValueError("video_latent_path must store either (C,T,H,W) or (1,C,T,H,W)")
 
        if audio.dim() == 3:
            if audio.shape[0] != 1:
                raise ValueError("audio_latent_path must store either (T,C) or (1,T,C)")
            audio = audio[0]
        if audio.dim() != 2:
            raise ValueError("audio_latent_path must store either (T,C) or (1,T,C)")
        if audio.shape[-1] != 64:
            raise ValueError("audio latent last dim must be 64")
 
        audio_len = int(item.get("audio_len", audio.shape[0]))
        audio_len = min(audio_len, audio.shape[0])

        if self.temporal_crop_latent_frames is not None:
            target_t = int(self.temporal_crop_latent_frames)
            if target_t > 0 and video.shape[1] > target_t:
                t_total = int(video.shape[1])
                start = int(torch.randint(0, t_total - target_t + 1, (1,)).item())
                video = video[:, start : start + target_t].contiguous()

                audio_target_len = 4 * (target_t - 1) + 1
                audio_start = start * 4
                if audio_start < audio.shape[0]:
                    audio_slice = audio[audio_start : min(audio_start + audio_target_len, audio.shape[0])]
                else:
                    audio_slice = audio[:0]
                if audio_slice.shape[0] < audio_target_len:
                    pad = torch.zeros(audio_target_len - audio_slice.shape[0], 64, dtype=audio.dtype)
                    audio_slice = torch.cat([audio_slice, pad], dim=0)
                audio = audio_slice.contiguous()
                audio_len = int(audio_target_len)
        out = {"video_x0": video, "audio_x0": audio, "audio_len": audio_len, "prompt": prompt}
        if txt_feat is not None:
            out["txt_feat"] = txt_feat
            out["txt_feat_len"] = txt_feat_len
        return out


class MultiLatentManifestDataset(Dataset):
    def __init__(
        self,
        manifest_paths: List[str],
        *,
        temporal_crop_latent_frames: Optional[int] = None,
        expected_latent_h: Optional[int] = None,
        expected_latent_w: Optional[int] = None,
    ):
        if not manifest_paths:
            raise ValueError("manifest_paths is empty")
        self.manifest_paths = [os.path.abspath(p) for p in manifest_paths]
        self.items: List[Dict[str, Any]] = []
        for p in self.manifest_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"manifest does not exist: {p}")
            base_dir = os.path.dirname(os.path.abspath(p))
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    obj["_manifest_dir"] = base_dir
                    self.items.append(obj)
        if not self.items:
            raise ValueError(f"all manifests are empty: {self.manifest_paths}")
        self.temporal_crop_latent_frames = temporal_crop_latent_frames
        self.expected_latent_h = int(expected_latent_h) if expected_latent_h is not None else None
        self.expected_latent_w = int(expected_latent_w) if expected_latent_w is not None else None

    def __len__(self):
        return len(self.items)

    def _load_tensor(self, path: str, manifest_dir: str) -> torch.Tensor:
        if not os.path.isabs(path):
            path = os.path.join(manifest_dir, path)
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict) and "tensor" in obj and isinstance(obj["tensor"], torch.Tensor):
            return obj["tensor"]
        raise ValueError(f"Unsupported tensor file: {path}")

    def _load_txt_feat(self, item: Dict[str, Any], manifest_dir: str) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        txt_path = item.get("txt_latent_path")
        if not txt_path:
            return None, None
        feat = self._load_tensor(str(txt_path), manifest_dir)
        if feat.dim() == 3 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        if feat.dim() != 2:
            raise ValueError(f"txt_latent_path must store (L,D) or (1,L,D): {txt_path}")
        txt_len = int(item.get("txt_feat_len", feat.shape[0]))
        return feat.to(torch.float32), txt_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        base_dir = str(item.get("_manifest_dir", ""))
        if not base_dir:
            base_dir = os.getcwd()
        video = self._load_tensor(item["video_latent_path"], base_dir)
        audio = self._load_tensor(item["audio_latent_path"], base_dir)
        prompt = item.get("prompt", "")
        txt_feat, txt_feat_len = self._load_txt_feat(item, base_dir)

        if video.dim() == 5:
            if video.shape[0] != 1:
                raise ValueError("video_latent_path must store either (C,T,H,W) or (1,C,T,H,W)")
            video = video[0]
        if video.dim() != 4:
            raise ValueError("video_latent_path must store either (C,T,H,W) or (1,C,T,H,W)")
        if self.expected_latent_h is not None and int(video.shape[-2]) != int(self.expected_latent_h):
            raise ValueError(
                f"Unexpected latent H={int(video.shape[-2])} (expected {int(self.expected_latent_h)}). "
                f"video_latent_path={item.get('video_latent_path')} manifest_dir={base_dir}"
            )
        if self.expected_latent_w is not None and int(video.shape[-1]) != int(self.expected_latent_w):
            raise ValueError(
                f"Unexpected latent W={int(video.shape[-1])} (expected {int(self.expected_latent_w)}). "
                f"video_latent_path={item.get('video_latent_path')} manifest_dir={base_dir}"
            )

        if audio.dim() == 3:
            if audio.shape[0] != 1:
                raise ValueError("audio_latent_path must store either (T,C) or (1,T,C)")
            audio = audio[0]
        if audio.dim() != 2:
            raise ValueError("audio_latent_path must store either (T,C) or (1,T,C)")
        if audio.shape[-1] != 64:
            raise ValueError("audio latent last dim must be 64")

        audio_len = int(item.get("audio_len", audio.shape[0]))
        audio_len = min(audio_len, audio.shape[0])

        if self.temporal_crop_latent_frames is not None:
            target_t = int(self.temporal_crop_latent_frames)
            if target_t > 0 and video.shape[1] > target_t:
                t_total = int(video.shape[1])
                start = int(torch.randint(0, t_total - target_t + 1, (1,)).item())
                video = video[:, start : start + target_t].contiguous()

                audio_target_len = 4 * (target_t - 1) + 1
                audio_start = start * 4
                if audio_start < audio.shape[0]:
                    audio_slice = audio[audio_start : min(audio_start + audio_target_len, audio.shape[0])]
                else:
                    audio_slice = audio[:0]
                if audio_slice.shape[0] < audio_target_len:
                    pad = torch.zeros(audio_target_len - audio_slice.shape[0], 64, dtype=audio.dtype)
                    audio_slice = torch.cat([audio_slice, pad], dim=0)
                audio = audio_slice.contiguous()
                audio_len = int(audio_target_len)
        out = {"video_x0": video, "audio_x0": audio, "audio_len": audio_len, "prompt": prompt}
        if txt_feat is not None:
            out["txt_feat"] = txt_feat
            out["txt_feat_len"] = txt_feat_len
        return out


def collate_train_batch(examples: List[Dict[str, Any]]) -> TrainBatch:
    video = torch.stack([ex["video_x0"] for ex in examples], dim=0)
    audio_list = [ex["audio_x0"] for ex in examples]
    max_len = max(audio.shape[0] for audio in audio_list)
    audio = torch.zeros(len(examples), max_len, audio_list[0].shape[-1], dtype=audio_list[0].dtype)
    for i, a in enumerate(audio_list):
        audio[i, : a.shape[0]] = a
    audio_len = [int(ex.get("audio_len", audio_list[i].shape[0])) for i, ex in enumerate(examples)]
    prompts = [str(ex.get("prompt", "")) for ex in examples]
    txt_feats = [ex.get("txt_feat") for ex in examples]
    txt_feat = None
    txt_feat_lens = None
    if txt_feats and all(t is not None for t in txt_feats):
        txt_feat = torch.stack(txt_feats, dim=0)
        txt_feat_lens = [int(ex.get("txt_feat_len", txt_feats[i].shape[0])) for i, ex in enumerate(examples)]
    return TrainBatch(
        video_x0=video,
        audio_x0=audio,
        audio_len=audio_len,
        prompts=prompts,
        txt_feat=txt_feat,
        txt_feat_lens=txt_feat_lens,
    )
 
 
def apply_shift(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    if shift == 1.0:
        return sigma
    return shift * sigma / (1 + (shift - 1) * sigma)
 

def crop_video_to_patch_multiple(video: torch.Tensor, *, t_patch_size: int, patch_size: int) -> torch.Tensor:
    if video.dim() != 5:
        raise ValueError(f"video must be 5D (B,C,T,H,W), got shape={tuple(video.shape)}")
    if t_patch_size <= 0 or patch_size <= 0:
        raise ValueError(f"Invalid patch sizes: t_patch_size={t_patch_size}, patch_size={patch_size}")
    b, c, t, h, w = video.shape
    t2 = (t // t_patch_size) * t_patch_size
    h2 = (h // patch_size) * patch_size
    w2 = (w // patch_size) * patch_size
    if t2 <= 0 or h2 <= 0 or w2 <= 0:
        raise ValueError(f"Cannot crop to non-positive shape from {tuple(video.shape)}")
    if t2 == t and h2 == h and w2 == w:
        return video
    return video[:, :, :t2, :h2, :w2].contiguous()


def random_crop_video_audio(
    video: torch.Tensor,
    audio: torch.Tensor,
    audio_len: List[int],
    *,
    latent_frames: int,
    latent_h: int,
    latent_w: int,
    t_patch_size: int,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    if video.dim() != 5:
        return video, audio, audio_len
    if audio.dim() != 3:
        return video, audio, audio_len
    b, c, t, h, w = video.shape
    if b <= 0 or t <= 0 or h <= 0 or w <= 0:
        return video, audio, audio_len

    t_crop = int(latent_frames) if int(latent_frames) > 0 else t
    h_crop = int(latent_h) if int(latent_h) > 0 else h
    w_crop = int(latent_w) if int(latent_w) > 0 else w

    t_crop = max(int(t_patch_size), min(t_crop, t))
    h_crop = max(int(patch_size), min(h_crop, h))
    w_crop = max(int(patch_size), min(w_crop, w))

    t_crop = (t_crop // int(t_patch_size)) * int(t_patch_size)
    h_crop = (h_crop // int(patch_size)) * int(patch_size)
    w_crop = (w_crop // int(patch_size)) * int(patch_size)

    if t_crop <= 0 or h_crop <= 0 or w_crop <= 0:
        return video, audio, audio_len
    if t_crop == t and h_crop == h and w_crop == w:
        return video, audio, audio_len

    device = video.device
    new_video = torch.empty((b, c, t_crop, h_crop, w_crop), device=device, dtype=video.dtype)

    audio_target_len = 4 * (t_crop - 1) + 1
    new_audio = torch.zeros((b, audio_target_len, audio.shape[-1]), device=device, dtype=audio.dtype)
    new_audio_len: List[int] = []

    for i in range(b):
        t0 = int(torch.randint(0, t - t_crop + 1, (1,), device=device).item()) if t > t_crop else 0
        h0 = int(torch.randint(0, h - h_crop + 1, (1,), device=device).item()) if h > h_crop else 0
        w0 = int(torch.randint(0, w - w_crop + 1, (1,), device=device).item()) if w > w_crop else 0
        new_video[i] = video[i, :, t0 : t0 + t_crop, h0 : h0 + h_crop, w0 : w0 + w_crop]

        a_len = int(audio_len[i]) if i < len(audio_len) else int(audio.shape[1])
        a_len = max(0, min(a_len, int(audio.shape[1])))
        a_start = int(t0) * 4
        a_end = min(a_start + audio_target_len, a_len)
        if a_start < a_end:
            new_audio[i, : a_end - a_start] = audio[i, a_start:a_end]
        new_audio_len.append(int(audio_target_len))

    return new_video, new_audio, new_audio_len


 
def sample_sigma(batch: int, device: torch.device, shift: float) -> torch.Tensor:
    sigma = torch.rand(batch, device=device)
    sigma = apply_shift(sigma, shift)
    return sigma
 
 
def build_txt_features(
    prompts: List[str],
    evaluation_config: EvaluationConfig,
    *,
    device: str,
    weight_dtype: torch.dtype,
    use_text_encoder: bool,
) -> Tuple[torch.Tensor, List[int]]:
    if (not use_text_encoder) or (not evaluation_config.txt_model_path):
        bsz = len(prompts)
        txt = torch.zeros(bsz, evaluation_config.t5_gemma_target_length, 3584, dtype=torch.float32, device=device)
        lens = [evaluation_config.t5_gemma_target_length for _ in range(bsz)]
        return txt, lens
 
    feats: List[torch.Tensor] = []
    for p in prompts:
        if not str(p).strip():
            feats.append(
                torch.zeros(1, evaluation_config.t5_gemma_target_length, 3584, dtype=torch.float32, device=device)
            )
            continue
        feat, _ = get_padded_t5_gemma_embedding(
            p,
            evaluation_config.txt_model_path,
            device,
            weight_dtype,
            evaluation_config.t5_gemma_target_length,
        )
        feats.append(feat)
    lens = [evaluation_config.t5_gemma_target_length for _ in prompts]
    return torch.cat(feats, dim=0), lens


def manifest_has_precomputed_txt(dataset: Dataset) -> bool:
    items = getattr(dataset, "items", None)
    if not items:
        return False
    for item in items[: min(32, len(items))]:
        if item.get("txt_latent_path"):
            return True
    return False


def apply_prompt_dropout_to_txt_feat(txt_feat: torch.Tensor, dropout: float) -> torch.Tensor:
    if dropout <= 0:
        return txt_feat
    out = txt_feat.clone()
    for i in range(out.shape[0]):
        if torch.rand(1).item() < dropout:
            out[i].zero_()
    return out


def build_dit_for_training(model_config, engine_config, device: torch.device, *, min_free_gb: float) -> DiTModel:
    model = DiTModel(model_config=model_config)
    if getattr(engine_config, "load", None):
        model = load_model_checkpoint(model, engine_config)
    assert_min_cuda_free_memory(min_free_gb, device=device)
    model.to(device=device, dtype=model_config.params_dtype)
    model.train()
    return model
 
 
def assert_min_cuda_free_memory(min_free_gb: float, *, device: Optional[torch.device] = None):
    if min_free_gb <= 0:
        return
    if not torch.cuda.is_available():
        return
    if device is None:
        idx = torch.cuda.current_device()
    else:
        if device.type != "cuda":
            return
        idx = device.index if device.index is not None else torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
    free_gb = free_bytes / 1024**3
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"CUDA device {idx} has only {free_gb:.2f} GiB free (total {total_bytes/1024**3:.2f} GiB). "
            f"Set CUDA_VISIBLE_DEVICES to select free GPUs or reduce --nproc_per_node. "
            f"You can also lower --min_free_gb to disable this check."
        )
 
 
def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, best_loss: float = float("inf")):
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-load-path", type=str, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--manifests", action="append", default=[])
    parser.add_argument("--manifest_glob", type=str, default=None)
    parser.add_argument("--all_datasets", action="store_true")
    parser.add_argument("--dataset_root", type=str, default=os.path.join(_PROJECT_ROOT, "dataset"))
    parser.add_argument("--manifest_mode", type=str, default="train", choices=["train", "all", "val"])
    parser.add_argument("--output_dir", type=str, default="output_train_ltx")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adafactor", choices=["adafactor", "adamw"])
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--use_text_encoder", action="store_true")
    parser.add_argument("--no_text_encoder", action="store_true")
    parser.add_argument(
        "--precomputed_txt",
        action="store_true",
        help="Load txt_feat from manifest txt_latent_path instead of running T5 each step.",
    )
    parser.add_argument(
        "--require_precomputed_txt",
        action="store_true",
        help="Fail if manifest rows are missing txt_latent_path while precomputed_txt is enabled.",
    )
    parser.add_argument("--txt_device", type=str, default="cpu")
    parser.add_argument("--prompt_dropout", type=float, default=0.0)
    parser.add_argument("--freeze_text_embedder", action="store_true")
    parser.add_argument("--text_lr_mult", type=float, default=0.1)
    parser.add_argument("--t5_target_length", type=int, default=None)
    parser.add_argument("--frame_receptive_field", type=int, default=None)
    parser.add_argument("--temporal_crop_latent_frames", type=int, default=None)
    parser.add_argument("--expected_latent_h", type=int, default=None)
    parser.add_argument("--expected_latent_w", type=int, default=None)
    parser.add_argument("--random_len", type=int, default=1000)
    parser.add_argument("--random_latent_frames", type=int, default=9)
    parser.add_argument("--random_latent_h", type=int, default=32)
    parser.add_argument("--random_latent_w", type=int, default=32)
    parser.add_argument("--random_audio_frames", type=int, default=101)
    parser.add_argument("--min_free_gb", type=float, default=8.0)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from (e.g. ckpt_step_2000.pt)")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    args, _ = parser.parse_known_args()
 
    old_argv = sys.argv
    if args.config_load_path:
        sys.argv = [old_argv[0], "--config-load-path", args.config_load_path]
    config = parse_config()
    sys.argv = old_argv

    if args.no_pretrained:
        config.engine_config.load = ""

    if args.amp_dtype == "bf16":
        config.arch_config.params_dtype = torch.bfloat16
    elif args.amp_dtype == "fp16":
        config.arch_config.params_dtype = torch.float16
    else:
        config.arch_config.params_dtype = torch.float32

    if args.frame_receptive_field is not None:
        config.evaluation_config.data_proxy_config.frame_receptive_field = int(args.frame_receptive_field)
    if args.t5_target_length is not None:
        config.evaluation_config.t5_gemma_target_length = int(args.t5_target_length)

    if args.no_text_encoder:
        use_text_encoder = False
    elif args.use_text_encoder:
        use_text_encoder = True
    else:
        use_text_encoder = bool(getattr(config.evaluation_config, "txt_model_path", ""))
    if (not use_text_encoder) and getattr(config.evaluation_config, "txt_model_path", "") and get_dp_rank() == 0:
        print_rank_0(
            "Warning: txt_model_path is set but text encoder is disabled. "
            "If you want the model to learn prompt-conditioned motion, enable --use_text_encoder (or omit both flags to default on)."
        )
    use_precomputed_txt = bool(args.precomputed_txt)
    if get_dp_rank() == 0:
        print_rank_0(
            f"use_text_encoder={use_text_encoder} use_precomputed_txt={use_precomputed_txt} "
            f"txt_device={args.txt_device}"
        )

    if torch.cuda.is_available():
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    assert_min_cuda_free_memory(args.min_free_gb)
 
    if args.batch_size > 1 and config.evaluation_config.data_proxy_config.frame_receptive_field != -1:
        config.evaluation_config.data_proxy_config.frame_receptive_field = -1
 
    model = build_dit_for_training(config.arch_config, config.engine_config, device, min_free_gb=args.min_free_gb)
    raw_model = model.module if hasattr(model, "module") else model
    if args.freeze_text_embedder and hasattr(raw_model, "adapter") and hasattr(raw_model.adapter, "text_embedder"):
        raw_model.adapter.text_embedder.requires_grad_(False)
    if dist.is_initialized() and dist.get_world_size() > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), broadcast_buffers=False)
 
    raw_model = model.module if hasattr(model, "module") else model
    text_param_ids: set[int] = set()
    if hasattr(raw_model, "adapter") and hasattr(raw_model.adapter, "text_embedder"):
        for p in raw_model.adapter.text_embedder.parameters():
            text_param_ids.add(id(p))

    main_params = []
    text_params = []
    for p in raw_model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in text_param_ids:
            text_params.append(p)
        else:
            main_params.append(p)

    param_groups = [{"params": main_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    if text_params:
        param_groups.append(
            {"params": text_params, "lr": float(args.lr) * float(args.text_lr_mult), "weight_decay": args.weight_decay}
        )

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    else:
        from transformers.optimization import Adafactor

        optimizer = Adafactor(param_groups, scale_parameter=False, relative_step=False, warmup_init=False)

    start_step = 0
    best_loss = float("inf")
    if args.resume_from and os.path.exists(args.resume_from):
        if get_dp_rank() == 0:
            print_rank_0(f"Resuming training from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location="cpu")
        if "model" in checkpoint:
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as e:
                if get_dp_rank() == 0:
                    print_rank_0(
                        f"Warning: failed to load optimizer state from {args.resume_from}: {e}. "
                        "Continue with freshly initialized optimizer."
                    )
        if "step" in checkpoint:
            start_step = checkpoint["step"]
        if "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]
 
    manifest_paths: List[str] = []
    if args.manifest:
        manifest_paths.append(args.manifest)
    if args.manifests:
        for v in args.manifests:
            if not v:
                continue
            parts = [p.strip() for p in str(v).split(",") if p.strip()]
            manifest_paths.extend(parts)
    if args.manifest_glob:
        manifest_paths.extend(sorted(glob.glob(args.manifest_glob)))

    if (not manifest_paths) and args.all_datasets:
        mode_name = f"latent_manifest_{args.manifest_mode}.jsonl"
        root = os.path.abspath(args.dataset_root)
        candidates = sorted(glob.glob(os.path.join(root, "*", mode_name)))
        if args.manifest_mode != "all":
            all_name = "latent_manifest_all.jsonl"
            fallbacks = sorted(glob.glob(os.path.join(root, "*", all_name)))
            non_empty = []
            for p in candidates:
                try:
                    if os.path.getsize(p) > 0:
                        non_empty.append(p)
                except OSError:
                    continue
            chosen = non_empty if non_empty else []
            if not chosen:
                chosen = fallbacks
            manifest_paths.extend(chosen)
        else:
            manifest_paths.extend(candidates)

    if manifest_paths:
        manifest_paths = [os.path.abspath(p) for p in manifest_paths]
        if len(manifest_paths) == 1:
            dataset = LatentManifestDataset(
                manifest_paths[0],
                temporal_crop_latent_frames=args.temporal_crop_latent_frames,
                expected_latent_h=args.expected_latent_h,
                expected_latent_w=args.expected_latent_w,
            )
        else:
            dataset = MultiLatentManifestDataset(
                manifest_paths,
                temporal_crop_latent_frames=args.temporal_crop_latent_frames,
                expected_latent_h=args.expected_latent_h,
                expected_latent_w=args.expected_latent_w,
            )
        if manifest_has_precomputed_txt(dataset):
            use_precomputed_txt = True
        elif args.require_precomputed_txt:
            raise RuntimeError("require_precomputed_txt is set but manifest has no txt_latent_path entries.")
        if get_dp_rank() == 0:
            print_rank_0(f"Loaded {len(manifest_paths)} manifest(s), total items={len(dataset)}")
            print_rank_0(f"use_precomputed_txt={use_precomputed_txt}")
            for p in manifest_paths[:10]:
                print_rank_0(f"  - {p}")
            if len(manifest_paths) > 10:
                print_rank_0(f"  ... (+{len(manifest_paths) - 10} more)")
    else:
        dataset = RandomLatentDataset(
            length=args.random_len,
            z_dim=config.evaluation_config.z_dim,
            latent_frames=args.random_latent_frames,
            latent_h=args.random_latent_h,
            latent_w=args.random_latent_w,
            audio_frames=args.random_audio_frames,
            device="cpu",
        )
 
    if get_dp_rank() == 0 and args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_train_batch,
    )
 
    data_proxy = MagiDataProxy(config.evaluation_config.data_proxy_config)
 
    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
 
    step = start_step
    running_loss = 0.0
    running_loss_video = 0.0
    running_loss_audio = 0.0
    t0 = time.time()
 
    optimizer.zero_grad(set_to_none=True)
    while step < args.num_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in dataloader:
            if step >= args.num_steps:
                break
 
            data_dtype = amp_dtype if amp_dtype != torch.float32 else torch.float32
            video_x0 = batch.video_x0.to(device, dtype=data_dtype, non_blocking=True)
            audio_x0 = batch.audio_x0.to(device, dtype=data_dtype, non_blocking=True)
            video_x0, audio_x0, audio_len = random_crop_video_audio(
                video_x0,
                audio_x0,
                list(batch.audio_len),
                latent_frames=int(args.random_latent_frames),
                latent_h=int(args.random_latent_h),
                latent_w=int(args.random_latent_w),
                t_patch_size=int(config.evaluation_config.data_proxy_config.t_patch_size),
                patch_size=int(config.evaluation_config.data_proxy_config.patch_size),
            )
            video_x0 = crop_video_to_patch_multiple(
                video_x0,
                t_patch_size=int(config.evaluation_config.data_proxy_config.t_patch_size),
                patch_size=int(config.evaluation_config.data_proxy_config.patch_size),
            )
 
            sigma = sample_sigma(video_x0.shape[0], device, config.evaluation_config.shift)
            sigma_video = sigma.view(-1, 1, 1, 1, 1)
            sigma_audio = sigma.view(-1, 1, 1)
 
            video_x1 = torch.randn_like(video_x0)
            audio_x1 = torch.randn_like(audio_x0)
 
            video_xt = (1 - sigma_video) * video_x0 + sigma_video * video_x1
            audio_xt = (1 - sigma_audio) * audio_x0 + sigma_audio * audio_x1
 
            v_target_video = video_x1 - video_x0
            v_target_audio = audio_x1 - audio_x0
 
            if use_precomputed_txt:
                if batch.txt_feat is None:
                    raise RuntimeError(
                        "use_precomputed_txt is enabled but batch has no txt_feat. "
                        "Run prepare_low_res_text_latents.py and ensure manifest has txt_latent_path."
                    )
                txt_feat = apply_prompt_dropout_to_txt_feat(
                    batch.txt_feat, float(args.prompt_dropout)
                )
                txt_lens = list(batch.txt_feat_lens or [])
                if isinstance(txt_feat, torch.Tensor) and txt_feat.device != device:
                    txt_feat = txt_feat.to(device, non_blocking=True)
            else:
                prompts = batch.prompts
                if use_text_encoder and float(args.prompt_dropout) > 0:
                    p = float(args.prompt_dropout)
                    dropped: List[str] = []
                    for s in prompts:
                        if torch.rand(1).item() < p:
                            dropped.append("")
                        else:
                            dropped.append(s)
                    prompts = dropped

                txt_feat, txt_lens = build_txt_features(
                    prompts,
                    config.evaluation_config,
                    device=args.txt_device,
                    weight_dtype=amp_dtype if amp_dtype != torch.float32 else torch.bfloat16,
                    use_text_encoder=use_text_encoder,
                )
                if isinstance(txt_feat, torch.Tensor) and txt_feat.device != device:
                    txt_feat = txt_feat.to(device, non_blocking=True)
 
            eval_input = TrainEvalInput(
                x_t=video_xt,
                audio_x_t=audio_xt,
                audio_feat_len=audio_len,
                txt_feat=txt_feat,
                txt_feat_len=txt_lens,
            )
            model_inputs = data_proxy.process_input(eval_input)
 
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                out = model(*model_inputs)
                v_pred_video, v_pred_audio = data_proxy.process_output(out)
                loss_video = torch.mean((v_pred_video.float() - v_target_video.float()) ** 2)
                loss_audio = torch.mean((v_pred_audio.float() - v_target_audio.float()) ** 2)
                
                # 如果输入音频是纯静音(全0)，则将 audio_loss 权重置为 0，防止破坏 Base 模型的生音能力
                # 乘以 0.0 可以保留计算图，避免 DDP 报错
                audio_is_silent = (audio_x0.abs().max() < 1e-5).float()
                loss_audio = loss_audio * (1.0 - audio_is_silent)
                
                loss = (loss_video + loss_audio) * 0.5
                loss = loss / args.grad_accum_steps
 
            loss.backward()
            running_loss += loss.item()
            running_loss_video += loss_video.item()
            running_loss_audio += loss_audio.item()
 
            if (step + 1) % args.grad_accum_steps == 0:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
 
            if step > start_step and step % args.log_every == 0 and get_dp_rank() == 0:
                elapsed = time.time() - t0
                avg_loss = running_loss / max(1, args.log_every)
                avg_loss_video = running_loss_video / max(1, args.log_every)
                avg_loss_audio = running_loss_audio / max(1, args.log_every)
                running_loss = 0.0
                running_loss_video = 0.0
                running_loss_audio = 0.0
                print_rank_0(f"step={step} loss={avg_loss:.6f} (v={avg_loss_video:.6f}, a={avg_loss_audio:.6f}) elapsed={elapsed:.1f}s")
                
                if args.wandb_project:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/loss_video": avg_loss_video,
                        "train/loss_audio": avg_loss_audio,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step": step,
                    }, step=step)

                t0 = time.time()
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_ckpt_path = os.path.join(args.output_dir, "ckpt_best.pt")
                    raw_model = model.module if hasattr(model, "module") else model
                    save_checkpoint(best_ckpt_path, raw_model, optimizer, step, best_loss)
                    print_rank_0(f"Saved new best model with loss {best_loss:.6f} to {best_ckpt_path}")
 
            if args.save_every > 0 and step > start_step and step % args.save_every == 0 and get_dp_rank() == 0:
                ckpt_path = os.path.join(args.output_dir, f"ckpt_step_{step}.pt")
                raw_model = model.module if hasattr(model, "module") else model
                save_checkpoint(ckpt_path, raw_model, optimizer, step, best_loss)
 
            step += 1
 
    if get_dp_rank() == 0 and step > start_step:
        ckpt_path = os.path.join(args.output_dir, f"ckpt_step_{step}.pt")
        raw_model = model.module if hasattr(model, "module") else model
        save_checkpoint(ckpt_path, raw_model, optimizer, step, best_loss)
 
    if get_dp_rank() == 0 and args.wandb_project:
        import wandb
        wandb.finish()
 
 
if __name__ == "__main__":
    full_argv = sys.argv
    cfg_load = None
    for i, a in enumerate(full_argv[1:]):
        if a == "--config-load-path" and i + 2 <= len(full_argv) - 1:
            cfg_load = full_argv[i + 2]
            break
        if a.startswith("--config-load-path="):
            cfg_load = a.split("=", 1)[1]
            break
    if cfg_load:
        sys.argv = [full_argv[0], "--config-load-path", cfg_load]
    else:
        sys.argv = [full_argv[0]]
    initialize_infra()
    sys.argv = full_argv
    main()
