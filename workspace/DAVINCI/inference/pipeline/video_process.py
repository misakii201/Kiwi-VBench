# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import subprocess
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from scipy.signal import resample
from torch.nn import functional as F

from inference.utils import print_rank_0


def write_video_ffmpeg(video_np: np.ndarray, *, fps: int, save_path: str, crf: int = 18, preset: str = "veryfast") -> None:
    if video_np.ndim != 4 or video_np.shape[-1] != 3:
        raise ValueError(f"video_np must be (T,H,W,3). Got: shape={video_np.shape}")
    if video_np.dtype != np.uint8:
        raise ValueError(f"video_np must be uint8. Got: {video_np.dtype}")
    video_np = np.ascontiguousarray(video_np)
    h = int(video_np.shape[1])
    w = int(video_np.shape[2])

    try:
        import av  # type: ignore

        container = av.open(save_path, mode="w")
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(int(crf)), "preset": str(preset)}

        for frame_np in video_np:
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return
    except Exception:
        pass

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(int(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(int(crf)),
        "-preset",
        str(preset),
        "-loglevel",
        "error",
        save_path,
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        proc.stdin.write(video_np.tobytes())
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg encode failed: {e}") from e


def merge_video_and_audio(video_path: str, audio_path: str, save_path: str):
    # Merge video with audio and keep the shortest stream.
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-y",
        save_path,
        "-loglevel",
        "error",
    ]
    try:
        subprocess.run(cmd, check=True)
        os.remove(video_path)
        os.remove(audio_path)
    except subprocess.CalledProcessError as e:
        print_rank_0(f"ffmpeg failed: {e}")


def upsample_video(video_np: np.ndarray, width: int, height: int, upsample_mode: Optional[str] = "bicubic") -> np.ndarray:
    """
    Upsample video NumPy array to specified resolution.

    This function assumes the input NumPy array is the result of VAE decoding,
    with data type uint8 and dimension order (T, H, W, C).

    Args:
        video_np (np.ndarray): Input video array with shape (T, H, W, C),
                               data type uint8.
        width (int): Target width.
        height (int): Target height.
        upsample_mode (str): Upsampling mode. Supports "bilinear", "nearest", "bicubic".
                            Defaults to "bilinear".

    Returns:
        np.ndarray: Upsampled video array with shape (T, height, width, C),
                    data type uint8.
    """
    def _maybe_fix_horizontal_canvas(video_u8: np.ndarray) -> np.ndarray:
        if os.getenv("MAGI_CANVAS_FIX", "").strip() not in ("1", "true", "True", "yes", "YES"):
            return video_u8
        if not isinstance(video_u8, np.ndarray) or video_u8.ndim != 4 or video_u8.shape[-1] != 3:
            return video_u8
        t, h0, w0, _ = video_u8.shape
        if t < 1 or w0 < 256 or h0 < 256:
            return video_u8

        frame0 = video_u8[0]
        gray = frame0.astype(np.float32).mean(axis=2)
        col_std = gray.std(axis=0)
        row_std = gray.std(axis=1)
        if col_std.size < 256 or row_std.size < 256:
            return video_u8

        pctl_str = os.getenv("MAGI_CANVAS_STD_PCTL", "").strip()
        mult_str = os.getenv("MAGI_CANVAS_STD_MULT", "").strip()
        min_std_str = os.getenv("MAGI_CANVAS_STD_MIN", "").strip()
        try:
            pctl = float(pctl_str) if pctl_str else 30.0
        except ValueError:
            pctl = 30.0
        try:
            mult = float(mult_str) if mult_str else 0.6
        except ValueError:
            mult = 0.6
        try:
            min_std = float(min_std_str) if min_std_str else 2.0
        except ValueError:
            min_std = 2.0
        pctl = max(1.0, min(pctl, 60.0))
        mult = max(0.1, min(mult, 1.5))
        min_std = max(0.5, min(min_std, 20.0))

        k = 9
        kernel = np.ones(k, dtype=np.int32)

        min_len = int(os.getenv("MAGI_CANVAS_SEG_MIN", "128").strip() or "128")
        min_len = max(64, min(min_len, max(w0, h0)))

        def _segments(std_1d: np.ndarray, *, n: int, min_len_px: int) -> list[tuple[int, int]]:
            p = float(np.percentile(std_1d, pctl))
            thr = max(float(min_std), float(mult) * p)
            active = std_1d > thr
            if not np.any(active):
                return []
            active_i = active.astype(np.int32)
            conv = np.convolve(active_i, kernel, mode="same")
            active2 = conv >= (k // 2 + 1)
            segs_out: list[tuple[int, int]] = []
            i2 = 0
            while i2 < n:
                if not active2[i2]:
                    i2 += 1
                    continue
                j2 = i2 + 1
                while j2 < n and active2[j2]:
                    j2 += 1
                if j2 - i2 >= int(min_len_px):
                    segs_out.append((int(i2), int(j2)))
                i2 = j2 + 1
            return segs_out

        segs_x = _segments(col_std, n=w0, min_len_px=min(min_len, w0))
        segs_y = _segments(row_std, n=h0, min_len_px=min(min_len, h0))

        if len(segs_x) < 2 and len(segs_y) < 2:
            return video_u8

        def _energy(y0: int, y1: int, x0: int, x1: int) -> float:
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0 or roi.shape[0] < 64 or roi.shape[1] < 64:
                return -1.0
            gx = float(np.abs(roi[:, 1:] - roi[:, :-1]).mean())
            gy = float(np.abs(roi[1:, :] - roi[:-1, :]).mean())
            return gx + gy

        if len(segs_x) >= 2:
            best_x = max(segs_x, key=lambda se: _energy(0, h0, int(se[0]), int(se[1])))
            x0, x1 = int(best_x[0]), int(best_x[1])
        else:
            x0, x1 = 0, int(w0)

        if len(segs_y) >= 2:
            best_y = max(segs_y, key=lambda se: _energy(int(se[0]), int(se[1]), 0, w0))
            y0, y1 = int(best_y[0]), int(best_y[1])
        else:
            y0, y1 = 0, int(h0)

        if (x1 - x0) <= 0 or (y1 - y0) <= 0:
            return video_u8

        seg = np.ascontiguousarray(video_u8[:, y0:y1, x0:x1, :])
        mode = os.getenv("MAGI_CANVAS_FIX_MODE", "").strip().lower()
        if mode not in ("", "pad", "pad_center", "crop"):
            mode = ""
        if not mode:
            mode = "pad_center"
        if mode == "crop":
            return seg
        seg_h = int(seg.shape[1])
        seg_w = int(seg.shape[2])
        if seg_w <= 0 or seg_h <= 0 or (seg_w >= int(w0 * 0.98) and seg_h >= int(h0 * 0.98)):
            return video_u8

        out = np.empty_like(video_u8)
        top_pad = (h0 - seg_h) // 2
        left_pad = (w0 - seg_w) // 2
        out[:, top_pad : top_pad + seg_h, left_pad : left_pad + seg_w, :] = seg

        if top_pad > 0:
            out[:, :top_pad, :, :] = out[:, top_pad : top_pad + 1, :, :]
        bottom_start = top_pad + seg_h
        if bottom_start < h0:
            out[:, bottom_start:, :, :] = out[:, bottom_start - 1 : bottom_start, :, :]
        if left_pad > 0:
            out[:, :, :left_pad, :] = out[:, :, left_pad : left_pad + 1, :]
        right_start = left_pad + seg_w
        if right_start < w0:
            out[:, :, right_start:, :] = out[:, :, right_start - 1 : right_start, :]
        return out


    if upsample_mode is None:
        upsample_mode = "bicubic"
    assert upsample_mode in ["bilinear", "nearest", "bicubic"], "Supported upsample modes: bilinear, nearest, bicubic"
    
    if isinstance(video_np, np.ndarray) and video_np.dtype == np.uint8:
        video_np = _maybe_fix_horizontal_canvas(video_np)

    # 1. Convert NumPy array to PyTorch tensor
    video_tensor = torch.from_numpy(video_np)

    # 2. Convert from uint8 to float32 and normalize to [0, 1]
    #    F.interpolate works better on floating point numbers
    if video_tensor.dtype == torch.uint8:
        video_tensor = video_tensor.float() / 255.0

    # 3. Adjust dimension order to match F.interpolate requirements (T, H, W, C) -> (T, C, H, W)
    video_tensor = rearrange(video_tensor, "t h w c -> t c h w").contiguous()

    in_h = int(video_tensor.shape[-2])
    in_w = int(video_tensor.shape[-1])
    strategy = os.getenv("MAGI_RESIZE_STRATEGY", "").strip().lower()
    if strategy in ("resize", "scale", "interpolate"):
        strategy = "resize"
    auto_crop = False
    auto_pad = False
    if not strategy:
        dh = int(in_h) - int(height)
        dw = int(in_w) - int(width)
        if height <= in_h and width <= in_w and 0 <= dh <= 64 and 0 <= dw <= 64:
            auto_crop = True
        dh2 = int(height) - int(in_h)
        dw2 = int(width) - int(in_w)
        if height >= in_h and width >= in_w and 0 <= dh2 <= 64 and 0 <= dw2 <= 64:
            auto_pad = True

    if (strategy in ("pad", "edge_pad", "replicate_pad", "reflect_pad")) or auto_pad:
        if height >= in_h and width >= in_w:
            pad_h = int(height) - int(in_h)
            pad_w = int(width) - int(in_w)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            if pad_top or pad_bottom or pad_left or pad_right:
                pad_mode = os.getenv("MAGI_PAD_MODE", "").strip().lower()
                if pad_mode not in ("replicate", "reflect"):
                    pad_mode = "replicate"
                video_tensor = F.pad(video_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode=pad_mode)
                in_h = int(video_tensor.shape[-2])
                in_w = int(video_tensor.shape[-1])

    if ((strategy in ("crop", "center_crop")) or auto_crop) and height <= in_h and width <= in_w:
        top = (in_h - int(height)) // 2
        left = (in_w - int(width)) // 2
        video_tensor = video_tensor[:, :, top : top + int(height), left : left + int(width)].contiguous()
        in_h = int(video_tensor.shape[-2])
        in_w = int(video_tensor.shape[-1])
    
    # 4. Use F.interpolate for upsampling
    #    Note: interpolate operates on spatial dimensions (H, W), so size=(height, width)
    if in_h == int(height) and in_w == int(width):
        upsampled_tensor = video_tensor
    else:
        upsampled_tensor = F.interpolate(
            video_tensor,
            size=(height, width),
            mode=upsample_mode,
            align_corners=False if upsample_mode in ["bilinear", "bicubic"] else None,
        )

    # 5. Adjust dimension order back (T, C, H, W) -> (T, H, W, C)
    upsampled_tensor = rearrange(upsampled_tensor, "t c h w -> t h w c").contiguous()
    
    # 6. Convert data from [0, 1] range back to [0, 255] and convert to uint8
    upsampled_tensor = (upsampled_tensor.clamp(0, 1) * 255).byte()
    
    # 7. Convert PyTorch tensor back to NumPy array
    return upsampled_tensor.numpy()


def resizecrop(image: Image.Image, th: int, tw: int) -> Image.Image:
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


def resample_audio_sinc(audio: torch.Tensor, time_stretching: float):
    print_rank_0(f"before resample audio: {audio.shape}")
    new_length = int(audio.shape[0] * time_stretching)
    audio = resample(audio, new_length)
    print_rank_0(f"after resample audio: {audio.shape}")
    return audio


def merge_overlapping_vae_features(audio_feats, overlap_ratio=0.5):
    if not audio_feats:
        return None
    if len(audio_feats) == 1:
        return audio_feats[0]

    batch_size, total_frames, feature_dim = audio_feats[0].shape
    overlap_frames = int(total_frames * overlap_ratio)
    step_frames = total_frames - overlap_frames
    final_length = (len(audio_feats) - 1) * step_frames + total_frames
    output_feat = torch.zeros(batch_size, final_length, feature_dim, device=audio_feats[0].device, dtype=audio_feats[0].dtype)

    for block_idx, current_feat in enumerate(audio_feats):
        output_start = block_idx * step_frames
        if block_idx == 0:
            output_feat[:, output_start : output_start + total_frames, :] = current_feat
            continue

        non_overlap_start = output_start + overlap_frames
        non_overlap_end = output_start + total_frames
        output_feat[:, non_overlap_start:non_overlap_end, :] = current_feat[:, overlap_frames:, :]

        for frame_idx in range(overlap_frames):
            output_pos = output_start + frame_idx
            prev_weight = (overlap_frames - frame_idx) / overlap_frames
            curr_weight = frame_idx / overlap_frames
            output_feat[:, output_pos, :] = (
                prev_weight * output_feat[:, output_pos, :] + curr_weight * current_feat[:, frame_idx, :]
            )
    return output_feat


def load_audio_and_encode(audio_vae: any, audio_path: str, seconds: Optional[float] = None) -> torch.Tensor:
    """Load and encode audio using the provided audio VAE."""
    try:
        import whisper  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'whisper'. Install openai-whisper to use --audio_path / audio conditioning."
        ) from e

    sample_rate = 51200
    audio_chunk_duration = 29
    overlap_ratio = 0.5

    audio_full = whisper.load_audio(audio_path, sr=sample_rate)
    if seconds is not None:
        audio_full = audio_full[: min(int(float(seconds) * sample_rate), audio_full.shape[0])]
    total_samples = audio_full.shape[0]

    window_size = int(audio_chunk_duration * sample_rate)
    step_size = int(window_size * (1 - overlap_ratio))
    if total_samples <= window_size:
        audio = torch.from_numpy(audio_full).cuda()
        audio = audio.unsqueeze(0).expand(2, -1)
        return audio_vae.vae_model.encode(audio)

    encoded_chunks = []
    latent_to_audio_ratio = None
    for offset_start in range(0, total_samples, step_size):
        offset_end = min(offset_start + window_size, total_samples)
        chunk = whisper.pad_or_trim(audio_full[offset_start:offset_end], length=window_size)
        chunk_tensor = torch.from_numpy(chunk).cuda().unsqueeze(0).expand(2, -1)
        encoded_chunk = audio_vae.vae_model.encode(chunk_tensor)

        if latent_to_audio_ratio is None:
            latent_to_audio_ratio = encoded_chunk.shape[-1] / window_size

        encoded_chunks.append(encoded_chunk.permute(0, 2, 1))
        if offset_end >= total_samples:
            break

    final_feat = merge_overlapping_vae_features(encoded_chunks, overlap_ratio=overlap_ratio).permute(0, 2, 1)
    final_target_len = math.ceil(total_samples * latent_to_audio_ratio)
    return final_feat[:, :, :final_target_len]
