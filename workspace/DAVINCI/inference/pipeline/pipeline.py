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

import os
import random
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from inference.common import EvaluationConfig, parse_config
from inference.model.dit import get_dit
from inference.model.dit import DiTModel
from .video_generate import MagiEvaluator
from .video_process import merge_video_and_audio, upsample_video, write_video_ffmpeg


class MagiPipeline:
    """Pipeline facade for inference."""

    def __init__(self, model: DiTModel, evaluation_config: EvaluationConfig, device: str = "cuda"):
        self.model = model
        self.evaluation_config = evaluation_config
        config = parse_config()
        if evaluation_config.use_sr_model:
            config.engine_config.load = evaluation_config.sr_model_path
            sr_model = get_dit(config.sr_arch_config, config.engine_config)
        else:
            sr_model = None
        self.evaluator = MagiEvaluator(model, sr_model, evaluation_config, device)

    def _validate_offline_request(
        self,
        prompt: str,
        save_path_prefix: str,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("`prompt` must be a non-empty string.")
        if not save_path_prefix or not save_path_prefix.strip():
            raise ValueError("`save_path_prefix` must be a non-empty string.")

    def run_offline(
        self,
        prompt: str,
        image: Union[str, Image.Image, None],
        audio: Optional[str],
        save_path_prefix: str,
        seed: int = 42,
        seconds: float = 4.0,
        br_width: int = 480,
        br_height: int = 272,
        sr_width: Optional[int] = None,
        sr_height: Optional[int] = None,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
        upsample_mode: Optional[str] = None,
    ):
        self._validate_offline_request(prompt=prompt, save_path_prefix=save_path_prefix)
        if output_width is not None and output_height is not None:
            if not os.getenv("MAGI_ALIGN_RESOLUTION", "").strip():
                os.environ["MAGI_ALIGN_RESOLUTION"] = "ceil"
            if not os.getenv("MAGI_RESIZE_STRATEGY", "").strip():
                os.environ["MAGI_RESIZE_STRATEGY"] = "resize"
            if not os.getenv("MAGI_CANVAS_FIX", "").strip():
                os.environ["MAGI_CANVAS_FIX"] = "1"

        if self.evaluator.sr_model is not None:
            save_path = f"{save_path_prefix}_{seconds}s_{br_width}x{br_height}_{sr_width}x{sr_height}.mp4"
        else:
            save_path = f"{save_path_prefix}_{seconds}s_{br_width}x{br_height}.mp4"

        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.random.manual_seed(seed)
            video_np, audio_np = self.evaluator.evaluate(
                prompt,
                image,
                audio,
                seconds=seconds,
                br_width=br_width,
                br_height=br_height,
                sr_width=sr_width,
                sr_height=sr_height,
                br_num_inference_steps=self.evaluation_config.num_inference_steps,
                sr_num_inference_steps=self.evaluation_config.sr_num_inference_steps,
            )

        is_last_rank = (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or (
            torch.distributed.get_rank() == torch.distributed.get_world_size() - 1
        )
        if is_last_rank:
            if video_np is None or audio_np is None:
                raise RuntimeError(
                    "Evaluator returned empty outputs on the saving rank. "
                    "Check VAE/video decode and distributed rank setup."
                )
            if output_width is not None and output_height is not None:
                video_np = upsample_video(video_np, output_width, output_height, upsample_mode)
            alpha_str = os.getenv("MAGI_TEMPORAL_SMOOTH_ALPHA", "").strip()
            if alpha_str:
                try:
                    alpha = float(alpha_str)
                except ValueError:
                    alpha = -1.0
                if 0.0 < alpha < 1.0 and isinstance(video_np, np.ndarray) and video_np.ndim == 4 and video_np.shape[0] >= 2:
                    x = video_np.astype(np.float32, copy=False)
                    y = np.empty_like(x, dtype=np.float32)
                    y[0] = x[0]
                    a = float(alpha)
                    b = 1.0 - a
                    for i in range(1, int(x.shape[0])):
                        y[i] = a * x[i] + b * y[i - 1]
                    video_np = np.clip(y, 0, 255).astype(np.uint8)

        if is_last_rank:
            out_dir = os.path.dirname(os.path.abspath(save_path)) or "."
            os.makedirs(out_dir, exist_ok=True)
            saving_name = f"{prompt.replace(' ', '_')[:10]}"
            token = f"{os.getpid()}_{random.randint(0, 1000000)}"
            audio_path = os.path.join(out_dir, f"{saving_name}_{token}.wav")
            video_path = os.path.join(out_dir, f"{saving_name}_{token}.mp4")
            sf.write(audio_path, audio_np, self.evaluator.audio_vae.sample_rate)
            video_np = np.ascontiguousarray(video_np)
            write_video_ffmpeg(video_np, fps=int(self.evaluation_config.fps), save_path=video_path)
            assert os.path.exists(video_path)
            merge_video_and_audio(video_path, audio_path, save_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return save_path
