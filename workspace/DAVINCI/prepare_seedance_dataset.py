import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
 
import av
import numpy as np
import pandas as pd
import torch
from PIL import Image
 
from inference.common import parse_config
from inference.model.sa_audio.sa_audio_model import SAAudioFeatureExtractor
from inference.model.vae2_2.vae2_2_model import get_vae2_2
 
 
@dataclass
class Record:
    video_path: str
    prompt: str
 
 
def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
 
 
def _read_csv(csv_path: str) -> List[Record]:
    df = pd.read_csv(csv_path)
    if "media_path" not in df.columns or "caption" not in df.columns:
        raise ValueError(f"CSV must contain columns: caption, media_path. Got: {list(df.columns)}")
 
    records: List[Record] = []
    for _, row in df.iterrows():
        video_path = str(row["media_path"]).strip()
        prompt = str(row["caption"]).strip()
        if not video_path or video_path.lower() == "nan":
            continue
        if not prompt or prompt.lower() == "nan":
            prompt = ""
        records.append(Record(video_path=video_path, prompt=prompt))
    return records
 
 
def _write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    _safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
 
 
def _split(records: List[Record], val_ratio: float, seed: int) -> Tuple[List[Record], List[Record]]:
    if val_ratio <= 0:
        return records, []
    if val_ratio >= 1:
        return [], records
 
    import random
 
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    val_n = int(round(len(records) * val_ratio))
    val_idx = set(indices[:val_n])
    train, val = [], []
    for i, r in enumerate(records):
        (val if i in val_idx else train).append(r)
    return train, val
 
 
def _center_crop(image: Image.Image, th: int, tw: int) -> Image.Image:
    w, h = image.size
    if w == tw and h == th:
        return image
    if h / w > th / tw:
        new_w = int(w)
        new_h = int(new_w * th / tw)
    else:
        new_h = int(h)
        new_w = int(new_h * tw / th)
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    return image.crop((left, top, right, bottom))
 
 
def _decode_video_frames(video_path: str, num_frames: int) -> List[Image.Image]:
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        frames: List[Image.Image] = []
        for frame in container.decode(stream):
            img = Image.fromarray(frame.to_rgb().to_ndarray())
            frames.append(img)
            if len(frames) >= num_frames:
                break
        if not frames:
            raise RuntimeError(f"No decoded frames: {video_path}")
        if len(frames) < num_frames:
            frames.extend([frames[-1]] * (num_frames - len(frames)))
        return frames
    finally:
        container.close()
 
 
def _extract_audio_mono(video_path: str, target_sample_rate: int, max_seconds: Optional[float]) -> Optional[torch.Tensor]:
    container = av.open(video_path)
    try:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            return None
        stream = audio_streams[0]
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=target_sample_rate)
        chunks: List[torch.Tensor] = []
        max_samples = None
        if max_seconds is not None:
            max_samples = int(max_seconds * target_sample_rate)
        total = 0
        for frame in container.decode(stream):
            resampled = resampler.resample(frame)
            if resampled is None:
                continue
            if not isinstance(resampled, list):
                resampled = [resampled]
            for af in resampled:
                arr = af.to_ndarray()
                if arr.ndim == 2:
                    arr = arr[0]
                t = torch.from_numpy(arr.astype("float32", copy=False))
                chunks.append(t)
                total += t.numel()
                if max_samples is not None and total >= max_samples:
                    break
            if max_samples is not None and total >= max_samples:
                break
        if not chunks:
            return None
        audio = torch.cat(chunks, dim=0)
        if max_samples is not None:
            audio = audio[:max_samples]
        return audio
    finally:
        container.close()
 
 
def _encode_audio_latent(audio_vae: SAAudioFeatureExtractor, audio_mono: torch.Tensor, device: torch.device) -> torch.Tensor:
    wav = audio_mono.to(device=device, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
    lat = audio_vae.encode(wav)
    if lat.dim() != 3:
        raise RuntimeError(f"Unexpected audio latent shape: {tuple(lat.shape)}")
    if lat.shape[-1] == 64:
        lat_t64 = lat[0]
    elif lat.shape[1] == 64:
        lat_t64 = lat[0].transpose(0, 1)
    else:
        raise RuntimeError(f"Cannot infer audio latent layout: {tuple(lat.shape)}")
    if lat_t64.dim() != 2 or lat_t64.shape[-1] != 64:
        raise RuntimeError(f"Unexpected audio latent final shape: {tuple(lat_t64.shape)}")
    return lat_t64.contiguous().to(dtype=torch.float32, device="cpu")
 
 
def _encode_video_latent(
    vae: Any,
    video_frames: List[Image.Image],
    *,
    device: torch.device,
    dtype: torch.dtype,
    height: int,
    width: int,
    pad_if_smaller: bool = False,
    pad_mode: str = "edge",
) -> torch.Tensor:
    pad_mode = str(pad_mode or "edge").strip().lower()
    if pad_mode not in ("edge", "reflect"):
        pad_mode = "edge"

    def _pad_to_target(img: Image.Image) -> Image.Image:
        w0, h0 = img.size
        if w0 == width and h0 == height:
            return img
        if (not pad_if_smaller) or w0 > width or h0 > height:
            return img
        pad_l = max(0, (width - w0) // 2)
        pad_r = max(0, width - w0 - pad_l)
        pad_t = max(0, (height - h0) // 2)
        pad_b = max(0, height - h0 - pad_t)
        if pad_l == 0 and pad_r == 0 and pad_t == 0 and pad_b == 0:
            return img
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return img
        padded = np.pad(
            arr,
            ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
            mode=pad_mode,
        )
        return Image.fromarray(padded)

    processed: List[Image.Image] = []
    for img in video_frames:
        img2 = _pad_to_target(img)
        if img2.size != (width, height):
            img2 = _center_crop(img2, height, width)
            img2 = img2.resize((width, height), resample=Image.BICUBIC)
        processed.append(img2)
 
    frames_np = np.stack([np.asarray(p, dtype=np.uint8) for p in processed], axis=0)
    x = torch.from_numpy(frames_np).to(dtype=torch.float32).div(255.0)
    x = x.mul(2.0).sub(1.0)
    x = x.permute(0, 3, 1, 2)
    x = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    x = x.to(device=device, dtype=dtype)
    with torch.no_grad():
        lat = vae.encode(x)
    if lat.dim() != 5:
        raise RuntimeError(f"Unexpected video latent shape: {tuple(lat.shape)}")
    return lat[0].contiguous().to(dtype=torch.float32, device="cpu")
 
 
def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")
 
 
def _resolve_vae_checkpoint(path: str) -> str:
    if not path:
        return path
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return path
    preferred = os.path.join(path, "Wan2.2_VAE.pth")
    if os.path.isfile(preferred):
        return preferred
    candidates: List[str] = []
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if not os.path.isfile(p):
            continue
        low = name.lower()
        if low.endswith((".pth", ".pt")) and ("vae" in low):
            candidates.append(p)
    candidates.sort()
    if candidates:
        return candidates[0]
    raise ValueError(f"Could not find VAE checkpoint under directory: {path}")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/kwkj-k8s/LTX-2/videos-lty/0331seedance/seedance原视频_25fps_121frames.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset", "seedance"),
    )
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--copy_csv", action="store_true")
    parser.add_argument("--video_mode", type=str, default="keep", choices=["keep", "symlink", "copy"])
    parser.add_argument("--encode_latents", action="store_true")
    parser.add_argument("--config-load-path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default="")
    parser.add_argument("--audio_model_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--pad_if_smaller", action="store_true")
    parser.add_argument("--pad_mode", type=str, default="edge", choices=["edge", "reflect"])
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--max_audio_seconds", type=float, default=0.0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--filter_face_lap_var_min", type=float, default=0.0)
    parser.add_argument("--filter_drop_no_face", action="store_true")
    parser.add_argument("--filter_drop_rect_mask", action="store_true")
    args = parser.parse_args()
 
    records = _read_csv(args.input_csv)
    if args.max_items and args.max_items > 0:
        records = records[: args.max_items]
 
    missing = [r for r in records if not os.path.exists(r.video_path)]
    kept = [r for r in records if os.path.exists(r.video_path)]
 
    _safe_mkdir(args.output_dir)
    if args.copy_csv:
        shutil.copy2(args.input_csv, os.path.join(args.output_dir, os.path.basename(args.input_csv)))
 
    if args.video_mode != "keep":
        videos_dir = os.path.join(args.output_dir, "videos")
        _safe_mkdir(videos_dir)
        for i, r in enumerate(kept):
            src = r.video_path
            dst = os.path.join(videos_dir, f"{i:06d}" + os.path.splitext(src)[1].lower())
            if os.path.exists(dst):
                r.video_path = dst
                continue
            if args.video_mode == "symlink":
                os.symlink(src, dst)
            elif args.video_mode == "copy":
                shutil.copy2(src, dst)
            else:
                raise ValueError(f"Unknown video_mode: {args.video_mode}")
            r.video_path = dst
 
    train_recs, val_recs = _split(kept, args.val_ratio, args.seed)
 
    train_items = [{"video_path": r.video_path, "prompt": r.prompt} for r in train_recs]
    val_items = [{"video_path": r.video_path, "prompt": r.prompt} for r in val_recs]
    all_items = [{"video_path": r.video_path, "prompt": r.prompt} for r in kept]
 
    _write_jsonl(os.path.join(args.output_dir, "manifest_all.jsonl"), all_items)
    _write_jsonl(os.path.join(args.output_dir, "manifest_train.jsonl"), train_items)
    _write_jsonl(os.path.join(args.output_dir, "manifest_val.jsonl"), val_items)
 
    if missing:
        _write_jsonl(
            os.path.join(args.output_dir, "missing.jsonl"),
            [{"video_path": r.video_path, "prompt": r.prompt} for r in missing],
        )
 
    meta = {
        "input_csv": args.input_csv,
        "output_dir": args.output_dir,
        "total_rows": len(records),
        "kept_rows": len(kept),
        "missing_rows": len(missing),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
 
    if not args.encode_latents:
        return
 
    cfg = None
    if args.config_load_path:
        cfg = parse_config()
    vae_model_path = args.vae_model_path or (cfg.evaluation_config.vae_model_path if cfg is not None else "")
    audio_model_path = args.audio_model_path or (cfg.evaluation_config.audio_model_path if cfg is not None else "")
    if not vae_model_path:
        raise ValueError("vae_model_path is empty. Provide --vae_model_path or --config-load-path with evaluation_config.vae_model_path")
    if not audio_model_path:
        raise ValueError(
            "audio_model_path is empty. Provide --audio_model_path or --config-load-path with evaluation_config.audio_model_path"
        )
 
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    weight_dtype = _parse_dtype(args.dtype)
    vae_ckpt = _resolve_vae_checkpoint(vae_model_path)
    vae = get_vae2_2(vae_ckpt, device=str(device), weight_dtype=weight_dtype)
    audio_vae = SAAudioFeatureExtractor(device=str(device), model_path=audio_model_path)
    target_sr = int(getattr(audio_vae, "sample_rate", 51200))
 
    latents_dir = os.path.join(args.output_dir, "latents")
    video_lat_dir = os.path.join(latents_dir, "video")
    audio_lat_dir = os.path.join(latents_dir, "audio")
    _safe_mkdir(video_lat_dir)
    _safe_mkdir(audio_lat_dir)
    output_dir_abs = os.path.abspath(args.output_dir)

    def _rel_to_output_dir(path: str) -> str:
        return os.path.relpath(os.path.abspath(path), output_dir_abs)
 
    start = max(0, int(args.start_index))
    end = int(args.end_index)
    if end < 0 or end > len(kept):
        end = len(kept)
    subset = kept[start:end]
 
    latent_items: List[Dict[str, Any]] = []
    max_audio_seconds = args.max_audio_seconds if args.max_audio_seconds > 0 else None
    for idx, r in enumerate(subset):
        global_idx = start + idx
        base = f"{global_idx:06d}"
        video_lat_path = os.path.join(video_lat_dir, base + ".pt")
        audio_lat_path = os.path.join(audio_lat_dir, base + ".pt")
        if args.skip_existing and os.path.exists(video_lat_path) and os.path.exists(audio_lat_path):
            audio_lat = torch.load(audio_lat_path, map_location="cpu")
            audio_len = int(audio_lat.shape[0]) if isinstance(audio_lat, torch.Tensor) else 0
            latent_items.append(
                {
                    "video_latent_path": _rel_to_output_dir(video_lat_path),
                    "audio_latent_path": _rel_to_output_dir(audio_lat_path),
                    "prompt": r.prompt,
                    "audio_len": audio_len,
                    "video_path": r.video_path,
                }
            )
            continue
 
        frames = _decode_video_frames(r.video_path, num_frames=args.num_frames)
        video_lat = _encode_video_latent(
            vae,
            frames,
            device=device,
            dtype=weight_dtype,
            height=args.height,
            width=args.width,
            pad_if_smaller=bool(args.pad_if_smaller),
            pad_mode=str(args.pad_mode),
        )
        torch.save(video_lat, video_lat_path)
 
        audio_mono = _extract_audio_mono(r.video_path, target_sample_rate=target_sr, max_seconds=max_audio_seconds)
        if audio_mono is None or audio_mono.numel() == 0:
            audio_lat = torch.zeros(args.num_frames, 64, dtype=torch.float32, device="cpu")
        else:
            audio_lat = _encode_audio_latent(audio_vae, audio_mono, device=device)
        torch.save(audio_lat, audio_lat_path)
 
        latent_items.append(
            {
                "video_latent_path": _rel_to_output_dir(video_lat_path),
                "audio_latent_path": _rel_to_output_dir(audio_lat_path),
                "prompt": r.prompt,
                "audio_len": int(audio_lat.shape[0]),
                "video_path": r.video_path,
            }
        )
 
    filtered_items = latent_items
    if (args.filter_face_lap_var_min and args.filter_face_lap_var_min > 0) or args.filter_drop_rect_mask:
        import cv2

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        def face_lap_var_inner(image_path: str) -> Optional[float]:
            img = cv2.imread(image_path)
            if img is None:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) == 0:
                return None
            x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
            roi = gray[y : y + h, x : x + w]
            ix0, ix1 = int(0.25 * w), int(0.75 * w)
            iy0, iy1 = int(0.25 * h), int(0.75 * h)
            inner = roi[iy0:iy1, ix0:ix1]
            if inner.size == 0:
                return None
            return float(cv2.Laplacian(inner, cv2.CV_64F).var())

        def has_rect_mask(image_path: str) -> bool:
            img = cv2.imread(image_path)
            if img is None:
                return False
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            ry0, ry1 = 0, int(0.55 * h)
            rx0, rx1 = int(0.20 * w), int(0.80 * w)
            roi = gray[ry0:ry1, rx0:rx1]
            if roi.size == 0:
                return False
            blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            edges = cv2.Canny(blurred, 40, 120)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) != 4:
                    continue
                if not cv2.isContourConvex(approx):
                    continue
                x, y, bw, bh = cv2.boundingRect(approx)
                if bw < 28 or bh < 28:
                    continue
                if bw > int(0.55 * roi.shape[1]) or bh > int(0.55 * roi.shape[0]):
                    continue
                ar = bw / max(1, bh)
                if ar < 0.7 or ar > 1.4:
                    continue
                cy = (y + bh / 2) / max(1, roi.shape[0])
                cx = (x + bw / 2) / max(1, roi.shape[1])
                if not (0.06 <= cy <= 0.55 and 0.20 <= cx <= 0.80):
                    continue
                patch = roi[y : y + bh, x : x + bw]
                if patch.size == 0:
                    continue
                inner = patch[int(0.15 * bh) : int(0.85 * bh), int(0.15 * bw) : int(0.85 * bw)]
                if inner.size == 0:
                    continue
                gx = cv2.Sobel(inner, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(inner, cv2.CV_32F, 0, 1, ksize=3)
                mag = cv2.magnitude(gx, gy)
                if float(mag.mean()) < 6.0:
                    return True
            return False

        kept_items: List[Dict[str, Any]] = []
        no_face = 0
        below = 0
        rect_mask = 0
        for it in latent_items:
            video_path = str(it.get("video_path", ""))
            if args.filter_drop_rect_mask and has_rect_mask(video_path):
                rect_mask += 1
                continue
            if args.filter_face_lap_var_min and args.filter_face_lap_var_min > 0:
                v = face_lap_var_inner(video_path)
                if v is None:
                    no_face += 1
                    if not args.filter_drop_no_face:
                        kept_items.append(it)
                    continue
                if v < float(args.filter_face_lap_var_min):
                    below += 1
                    continue
            kept_items.append(it)
        filtered_items = kept_items
        meta["filter_face_lap_var_min"] = float(args.filter_face_lap_var_min)
        meta["filter_drop_no_face"] = bool(args.filter_drop_no_face)
        meta["filter_drop_rect_mask"] = bool(args.filter_drop_rect_mask)
        meta["filter_no_face_rows"] = int(no_face)
        meta["filter_below_threshold_rows"] = int(below)
        meta["filter_rect_mask_rows"] = int(rect_mask)
        meta["filter_kept_rows"] = int(len(filtered_items))

    train_lat, val_lat = _split(filtered_items, args.val_ratio, args.seed)
    _write_jsonl(os.path.join(args.output_dir, "latent_manifest_all.jsonl"), filtered_items)
    _write_jsonl(os.path.join(args.output_dir, "latent_manifest_train.jsonl"), train_lat)
    _write_jsonl(os.path.join(args.output_dir, "latent_manifest_val.jsonl"), val_lat)
 
 
if __name__ == "__main__":
    main()
