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

import copy
import contextlib
import math
import os
import shutil
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor
from PIL import Image
from tqdm import tqdm

from inference.common import CPUOffloadWrapper, EvaluationConfig, get_arch_memory
from inference.infra.distributed import get_cp_group
from inference.model.dit import DiTModel
from inference.model.sa_audio import SAAudioFeatureExtractor
from inference.model.turbo_vaed import TurboVAED, get_turbo_vaed
from inference.model.vae2_2 import Wan2_2_VAE, get_vae2_2
from inference.utils import env_is_true, event_path_timer, print_mem_info_rank_0, print_rank_0, print_rank_last
from .prompt_process import get_padded_t5_gemma_embedding, pad_or_trim
from .scheduler_unipc import FlowUniPCMultistepScheduler
from .data_proxy import MagiDataProxy
from .video_process import load_audio_and_encode, resample_audio_sinc, resizecrop


def _cfg_rescale(
    cfg: torch.Tensor,
    uncond: torch.Tensor,
    *,
    rescale: float,
    dims: tuple[int, ...],
    eps: float = 1e-6,
) -> torch.Tensor:
    if rescale <= 0:
        return cfg
    if rescale > 1:
        rescale = 1.0
    cfg_f = cfg.float()
    uncond_f = uncond.float()
    std_cfg = cfg_f.std(dim=dims, keepdim=True)
    std_uncond = uncond_f.std(dim=dims, keepdim=True)
    cfg_rescaled = cfg_f * (std_uncond / (std_cfg + eps))
    out = cfg_f + rescale * (cfg_rescaled - cfg_f)
    return out.to(dtype=cfg.dtype)


def schedule_latent_step(
    *,
    video_scheduler,
    audio_scheduler,
    latent_video: torch.Tensor,
    latent_audio: torch.Tensor,
    t,
    idx: int,
    steps: int,
    v_cfg_video: torch.Tensor,
    v_cfg_audio: torch.Tensor,
    is_a2v: bool,
    cfg_number: int,
    use_sr_model: bool,
    using_sde_flag: bool,
):
    if cfg_number == 1 and (not use_sr_model):
        latent_video = video_scheduler.step_ddim(v_cfg_video, idx, latent_video)
        latent_audio = audio_scheduler.step_ddim(v_cfg_audio, idx, latent_audio)
        return latent_video, latent_audio

    if using_sde_flag:
        print_rank_0("Using sde scheduler")
        if use_sr_model:
            latent_video = video_scheduler.step(v_cfg_video, t, latent_video, return_dict=False)[0]
            return latent_video, latent_audio

        if idx < int(steps * (3 / 4)):
            noise_theta = 1.0 if (idx + 1) % 2 == 0 else 0.0
        else:
            noise_theta = 1.0 if idx % 3 == 0 else 0.0

        latent_video = video_scheduler.step_sde(v_cfg_video, idx, latent_video, noise_theta=noise_theta)
        if not is_a2v:
            latent_audio = audio_scheduler.step_sde(v_cfg_audio, idx, latent_audio, noise_theta=noise_theta)
        return latent_video, latent_audio

    latent_video = video_scheduler.step(v_cfg_video, t, latent_video, return_dict=False)[0]
    if not is_a2v and not use_sr_model:
        latent_audio = audio_scheduler.step(v_cfg_audio, t, latent_audio, return_dict=False)[0]
    return latent_video, latent_audio


@dataclass
class EvalInput:
    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: torch.Tensor | list[int]
    txt_feat: torch.Tensor
    txt_feat_len: torch.Tensor | list[int]


class ZeroSNRDDPMDiscretization:
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        shift_scale=1.0,  # noise schedule t_n -> t_m: logSNR(t_m) = logSNR(t_n) - log(shift_scale)
        keep_start=False,
        post_shift=False,
    ):
        if keep_start and not post_shift:
            linear_start = linear_start / (shift_scale + (1 - shift_scale) * linear_start)
        self.num_timesteps = num_timesteps
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64) ** 2
        betas = betas.numpy()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.to_torch = partial(torch.tensor, dtype=torch.float32)

        # SNR shift
        if not post_shift:
            self.alphas_cumprod = self.alphas_cumprod / (shift_scale + (1 - shift_scale) * self.alphas_cumprod)

        self.post_shift = post_shift
        self.shift_scale = shift_scale

    def __call__(self, n, do_append_zero=True, device="cpu", flip=False, return_idx=False):
        if return_idx:
            sigmas, idx = self.get_sigmas(n, device=device, return_idx=return_idx)
        else:
            sigmas = self.get_sigmas(n, device=device, return_idx=return_idx)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])]) if do_append_zero else sigmas
        if return_idx:
            return sigmas if not flip else torch.flip(sigmas, (0,)), idx
        else:
            return sigmas if not flip else torch.flip(sigmas, (0,))

    def get_sigmas(self, n, device="cpu", return_idx=False):
        if n < self.num_timesteps:
            timesteps = np.linspace(self.num_timesteps - 1, 0, n, endpoint=False).astype(int)[::-1]
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        alphas_cumprod = to_torch(alphas_cumprod)
        alphas_cumprod_sqrt = alphas_cumprod.sqrt()
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()

        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)

        if self.post_shift:
            alphas_cumprod_sqrt = (
                alphas_cumprod_sqrt**2 / (self.shift_scale + (1 - self.shift_scale) * alphas_cumprod_sqrt**2)
            ) ** 0.5

        if return_idx:
            return torch.flip(alphas_cumprod_sqrt, (0,)), timesteps
        else:
            return torch.flip(alphas_cumprod_sqrt, (0,))


class MagiEvaluator:
    def __init__(
        self,
        model: DiTModel,
        sr_model: Optional[DiTModel],
        config: EvaluationConfig,
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16,
    ):
        device = f"cuda:{torch.cuda.current_device()}"

        self.model = model
        self.model.eval()
        self.sr_model = sr_model
        if self.sr_model is not None:
            self.sr_model.eval()
            if env_is_true("CPU_OFFLOAD") and env_is_true("SR2_1080"):
                self.model = self.model.to(torch.device("cpu"))
                self.sr_model = self.sr_model.to(torch.device("cpu"))
        self.device = device
        self.config = config
        self.dtype = weight_dtype
        self.data_proxy = MagiDataProxy(config.data_proxy_config)
        sr_data_proxy_config = copy.deepcopy(config.data_proxy_config)
        sr_data_proxy_config.coords_style = "v1"
        self.sr_data_proxy = MagiDataProxy(sr_data_proxy_config)
        self.vae_stride = config.vae_stride
        self.z_dim = config.z_dim
        self.patch_size = config.patch_size

        self.sr_video_txt_guidance_scale = config.sr_video_txt_guidance_scale
        self.video_txt_guidance_scale = config.video_txt_guidance_scale
        self.audio_txt_guidance_scale = config.audio_txt_guidance_scale
        self.noise_value = config.noise_value
        self.shift = config.shift
        self.fps = config.fps
        self.use_cfg_trick = config.use_cfg_trick
        self.cfg_trick_start_frame = config.cfg_trick_start_frame
        self.cfg_trick_value = config.cfg_trick_value
        self.using_sde_flag = config.using_sde_flag

        print_mem_info_rank_0("Begin init MagiEvaluator")

        vae_model_path = os.path.join(config.vae_model_path, "Wan2.2_VAE.pth")
        self.vae: Wan2_2_VAE = CPUOffloadWrapper(
            get_vae2_2(vae_model_path, self.device, weight_dtype=weight_dtype), is_cpu_offload=get_arch_memory() <= 48
        )
        if config.use_turbo_vae:
            self.turbo_vae: TurboVAED = CPUOffloadWrapper(
                get_turbo_vaed(config.student_config_path, config.student_ckpt_path, self.device, weight_dtype=weight_dtype),
                is_cpu_offload=get_arch_memory() <= 48,
            )

        print_mem_info_rank_0("After init video vae")
        print_rank_0(f"vae loaded from {vae_model_path}")
        self.video_processor = VideoProcessor(vae_scale_factor=16)
        self.audio_vae = SAAudioFeatureExtractor(device=self.device, model_path=config.audio_model_path)
        self.sigmas = ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)
        print_mem_info_rank_0("After init audio vae")

        default_negative_prompt = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        )
        default_negative_prompt += ", low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, crackle, static, compression artifacts, MP3 artifacts, digital clipping, distortion, muffled, muddy, unclear, echo, reverb, room echo, over-reverberated, hollow sound, distant, washed out, harsh, shrill, piercing, grating, tinny, thin sound, boomy, bass-heavy, flat EQ, over-compressed, abrupt cut, jarring transition, sudden silence, looping artifact, music, instrumental, sirens, alarms, crowd noise, unrelated sound effects, chaotic, disorganized, messy, cheap sound"
        default_negative_prompt += ", emotionless, flat delivery, deadpan, lifeless, apathetic, robotic, mechanical, monotone, flat intonation, undynamic, boring, reading from a script, AI voice, synthetic, text-to-speech, TTS, insincere, fake emotion, exaggerated, overly dramatic, melodramatic, cheesy, cringey, hesitant, unconfident, tired, weak voice, stuttering, stammering, mumbling, slurred speech, mispronounced, bad articulation, lisp, vocal fry, creaky voice, mouth clicks, lip smacks, wet mouth sounds, heavy breathing, audible inhales, plosives, p-pops, coughing, clearing throat, sneezing, speaking too fast, rushed, speaking too slow, dragged out, unnatural pauses, awkward silence, choppy, disjointed, multiple speakers, two voices, background talking, out of tune, off-key, autotune artifacts"
        negative_prompt = os.getenv("MAGI_NEGATIVE_PROMPT", "").strip() or default_negative_prompt
        if negative_prompt != default_negative_prompt:
            print_rank_0(f"Using MAGI_NEGATIVE_PROMPT override ({len(negative_prompt)} chars)")

        txt_encoder_device = "cpu" if get_arch_memory() <= 48 else self.device
        self.txt_model_path = config.txt_model_path

        self.context_null, self.original_context_null_len = get_padded_t5_gemma_embedding(
            negative_prompt,
            self.txt_model_path,
            txt_encoder_device,
            self.dtype,
            config.t5_gemma_target_length,
        )
        print_mem_info_rank_0("After init t5 gamma")

    def forward(self, eval_input: EvalInput, use_sr_model: bool = False):
        amp_enabled = self.dtype != torch.float32
        if use_sr_model:
            eval_input = self.sr_data_proxy.process_input(eval_input)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=self.dtype, enabled=amp_enabled)
                if isinstance(eval_input, (list, tuple)) and len(eval_input) > 0 and isinstance(eval_input[0], torch.Tensor) and eval_input[0].is_cuda
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                noise_pred = self.sr_model(*eval_input)
            noise_pred = self.sr_data_proxy.process_output(noise_pred)
        else:
            eval_input = self.data_proxy.process_input(eval_input)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=self.dtype, enabled=amp_enabled)
                if isinstance(eval_input, (list, tuple)) and len(eval_input) > 0 and isinstance(eval_input[0], torch.Tensor) and eval_input[0].is_cuda
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                noise_pred = self.model(*eval_input)
            noise_pred = self.data_proxy.process_output(noise_pred)
        return noise_pred

    @torch.inference_mode()
    def evaluate(
        self,
        prompt: str,
        image: Optional[Image.Image],
        audio_path: Optional[str],
        seconds: float,
        br_width: int,
        br_height: int,
        sr_width: Optional[int],
        sr_height: Optional[int],
        br_num_inference_steps: int,
        sr_num_inference_steps: int,
    ):
        event_path_timer().reset()
        event_path_timer().synced_record("Step1: Prepare Latent Features")
        align_mode = os.getenv("MAGI_ALIGN_RESOLUTION", "").strip().lower()
        align_up = align_mode in ("ceil", "up", "round_up")

        def _align_latent(hw: int, *, stride: int, patch: int) -> int:
            base = float(hw) / float(stride) / float(patch)
            k = int(math.ceil(base) if align_up else math.floor(base))
            k = max(k, 1)
            return k * int(patch)

        br_latent_height = _align_latent(br_height, stride=self.vae_stride[1], patch=self.patch_size[1])
        br_latent_width = _align_latent(br_width, stride=self.vae_stride[2], patch=self.patch_size[2])
        br_height = br_latent_height * self.vae_stride[1]
        br_width = br_latent_width * self.vae_stride[2]

        # init latent
        if audio_path is not None:
            latent_audio = load_audio_and_encode(self.audio_vae, audio_path, seconds)
            latent_audio = latent_audio.permute(0, 2, 1)
            num_frames = latent_audio.shape[1]
            is_a2v = True
            print_rank_0(f"Using provided audio, latent_audio: {latent_audio.shape}")
        else:
            num_frames = int(round(float(seconds) * float(self.fps))) + 1
            if env_is_true("MAGI_T2V_FREEZE_AUDIO"):
                latent_audio = torch.zeros(1, num_frames, 64, dtype=torch.float32, device=self.device)
                is_a2v = True
                print_rank_0(f"Using zero audio (freeze), latent_audio: {latent_audio.shape}")
            else:
                latent_audio = torch.randn(1, num_frames, 64, dtype=torch.float32, device=self.device)
                is_a2v = False
                print_rank_0(f"Using random audio, latent_audio: {latent_audio.shape}")
        latent_length = (num_frames - 1) // 4 + 1
        latent_video = torch.randn(
            1, self.z_dim, latent_length, br_latent_height, br_latent_width, dtype=torch.float32, device=self.device
        )

        context, original_context_len = get_padded_t5_gemma_embedding(
            prompt, self.txt_model_path, self.device, self.dtype, self.config.t5_gemma_target_length
        )

        event_path_timer().synced_record("Step2: Encode Image for Basic Resolution")
        if image is not None:
            br_image = self.encode_image(image, br_height, br_width)
        else:
            br_image = None
        event_path_timer().synced_record("Step3: Basic Resolution Evaluation")

        if env_is_true("CPU_OFFLOAD") and env_is_true("SR2_1080"):
            self.model = self.model.to(self.device)
        br_latent_video, br_latent_audio = self.evaluate_with_latent(
            context,
            original_context_len,
            br_image,
            latent_video.clone(),
            latent_audio.clone(),
            br_num_inference_steps,
            is_a2v,
            use_sr_model=False,
        )
        if env_is_true("CPU_OFFLOAD") and env_is_true("SR2_1080"):
            self.model = self.model.to(torch.device("cpu"))

        if sr_width is not None and sr_height is not None and self.sr_model is not None:
            event_path_timer().synced_record("Step4: Encode Image for Super Resolution")
            sr_latent_height = _align_latent(sr_height, stride=self.vae_stride[1], patch=self.patch_size[1])
            sr_latent_width = _align_latent(sr_width, stride=self.vae_stride[2], patch=self.patch_size[2])
            sr_height = sr_latent_height * self.vae_stride[1]
            sr_width = sr_latent_width * self.vae_stride[2]
            if image is not None:
                sr_image = self.encode_image(image, sr_height, sr_width)
            else:
                sr_image = None
            latent_video = torch.nn.functional.interpolate(
                br_latent_video, size=(latent_length, sr_latent_height, sr_latent_width), mode="trilinear", align_corners=True
            )
            if self.noise_value != 0:
                noise = torch.randn_like(latent_video, device=latent_video.device)
                sigmas = self.sigmas.to(latent_video.device)
                sigma = sigmas[self.noise_value]
                latent_video = latent_video * sigma + noise * (1 - sigma**2) ** 0.5
            event_path_timer().synced_record("Step5: Super Resolution Evaluation")
            print_mem_info_rank_0("Before super resolution evaluation")
            latent_audio = br_latent_audio.clone()
            br_latent_audio = torch.randn_like(
                br_latent_audio, device=br_latent_audio.device
            ) * self.config.sr_audio_noise_scale + br_latent_audio * (1 - self.config.sr_audio_noise_scale)

            if env_is_true("CPU_OFFLOAD") and env_is_true("SR2_1080"):
                self.sr_model = self.sr_model.to(self.device)
            latent_video, _ = self.evaluate_with_latent(
                context,
                original_context_len,
                sr_image,
                latent_video.clone(),
                br_latent_audio.clone(),
                sr_num_inference_steps,
                is_a2v,
                use_sr_model=True,
            )
            if env_is_true("CPU_OFFLOAD") and env_is_true("SR2_1080"):
                self.sr_model = self.sr_model.to(torch.device("cpu"))
        else:
            latent_video = br_latent_video
            latent_audio = br_latent_audio

        event_path_timer().synced_record("Step6: Decode Video", print_fn=print_rank_last)
        result = self.post_process(latent_video, latent_audio)
        event_path_timer().synced_record("Step8: Post Process", print_fn=print_rank_last)
        return result

    def schedule(
        self,
        video_scheduler,
        audio_scheduler,
        latent_video,
        latent_audio,
        t,
        idx,
        steps,
        v_cfg_video,
        v_cfg_audio,
        is_a2v,
        cfg_number,
        use_sr_model=False,
    ):
        return schedule_latent_step(
            video_scheduler=video_scheduler,
            audio_scheduler=audio_scheduler,
            latent_video=latent_video,
            latent_audio=latent_audio,
            t=t,
            idx=idx,
            steps=steps,
            v_cfg_video=v_cfg_video,
            v_cfg_audio=v_cfg_audio,
            is_a2v=is_a2v,
            cfg_number=cfg_number,
            use_sr_model=use_sr_model,
            using_sde_flag=self.config.using_sde_flag,
        )

    @torch.inference_mode()
    def evaluate_with_latent(
        self,
        context: torch.Tensor,
        original_context_len: int,
        latent_image: Optional[torch.Tensor],
        latent_video: torch.Tensor,
        latent_audio: torch.Tensor,
        num_inference_steps: int,
        is_a2v: bool = False,
        use_sr_model: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_scheduler = FlowUniPCMultistepScheduler()
        audio_scheduler = FlowUniPCMultistepScheduler()
        video_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=self.shift)
        audio_scheduler.set_timesteps(num_inference_steps, device=self.device, shift=self.shift)
        timesteps = video_scheduler.timesteps

        # a inference trick to aviod over exposure in the I2V evaluation
        latent_length = latent_video.shape[2]
        sr_video_txt_guidance_scale = (
            torch.tensor(self.sr_video_txt_guidance_scale, device=self.device).expand(1, 1, latent_length, 1, 1).clone()
        )
        if self.use_cfg_trick:
            sr_video_txt_guidance_scale[:, :, : self.cfg_trick_start_frame] = min(
                self.cfg_trick_value, self.sr_video_txt_guidance_scale
            )

        # forward
        for idx, t in enumerate(
            tqdm(timesteps, disable=torch.distributed.get_rank() != torch.distributed.get_world_size() - 1)
        ):
            if latent_image is not None:
                latent_video[:, :, :1] = latent_image[:, :, :1]
            if env_is_true("MAGI_DISABLE_GUIDANCE_SCHEDULE"):
                video_txt_guidance_scale = self.video_txt_guidance_scale
            else:
                try:
                    low_scale = float(os.getenv("MAGI_GUIDANCE_LOW_SCALE", "2.0"))
                except ValueError:
                    low_scale = 2.0
                flip_schedule = env_is_true("MAGI_GUIDANCE_SCHEDULE_FLIP")
                if flip_schedule:
                    video_txt_guidance_scale = low_scale if t > 500 else self.video_txt_guidance_scale
                else:
                    video_txt_guidance_scale = self.video_txt_guidance_scale if t > 500 else low_scale
            try:
                max_scale = float(os.getenv("MAGI_GUIDANCE_MAX", "0"))
            except ValueError:
                max_scale = 0.0
            if max_scale > 0:
                video_txt_guidance_scale = min(float(video_txt_guidance_scale), max_scale)
            eval_input_cond = EvalInput(
                x_t=latent_video,
                audio_x_t=latent_audio,
                audio_feat_len=[latent_audio.shape[1]],
                txt_feat=context,
                txt_feat_len=[original_context_len],
            )   # txt + audio
            v_output = self.forward(eval_input_cond, use_sr_model=use_sr_model)
            v_cond_video = v_output[0]
            v_cond_audio = v_output[1]

            cfg_number = self.config.sr_cfg_number if use_sr_model else self.config.cfg_number
            if cfg_number == 1:
                v_cfg_video = v_cond_video
                v_cfg_audio = v_cond_audio
            elif cfg_number == 2:
                eval_input_uncond = EvalInput(
                    x_t=latent_video,
                    audio_x_t=latent_audio,
                    audio_feat_len=[latent_audio.shape[1]],
                    txt_feat=self.context_null,
                    txt_feat_len=[self.original_context_null_len],
                )
                v_output_uncond = self.forward(eval_input_uncond, use_sr_model=use_sr_model)
                v_uncond_video = v_output_uncond[0]
                v_uncond_audio = v_output_uncond[1]
                if use_sr_model:
                    v_cfg_video = v_uncond_video + sr_video_txt_guidance_scale * (v_cond_video - v_uncond_video)
                else:
                    v_cfg_video = v_uncond_video + video_txt_guidance_scale * (v_cond_video - v_uncond_video)
                v_cfg_audio = v_uncond_audio + self.audio_txt_guidance_scale * (v_cond_audio - v_uncond_audio)

                try:
                    cfg_rescale = float(os.getenv("MAGI_CFG_RESCALE", "0"))
                except ValueError:
                    cfg_rescale = 0.0
                if cfg_rescale > 0:
                    if v_cfg_video.dim() == 5:
                        v_cfg_video = _cfg_rescale(v_cfg_video, v_uncond_video, rescale=cfg_rescale, dims=(1, 2, 3, 4))
                    if v_cfg_audio.dim() == 3:
                        v_cfg_audio = _cfg_rescale(v_cfg_audio, v_uncond_audio, rescale=cfg_rescale, dims=(1, 2))
            else:
                raise ValueError(f"Invalid cfg_number: {cfg_number}")

            latent_video, latent_audio = self.schedule(
                video_scheduler,
                audio_scheduler,
                latent_video,
                latent_audio,
                t,
                idx,
                timesteps,
                v_cfg_video,
                v_cfg_audio,
                is_a2v,
                cfg_number,
                use_sr_model,
            )

        print_rank_0(f"latent_video: {latent_video.shape}, latent_audio: {latent_audio.shape}")
        if latent_image is not None:
            latent_video[:, :, :1] = latent_image[:, :, :1]
        return latent_video, latent_audio

    def encode_image(self, image: Image.Image, height: int, width: int):
        image = load_image(image)
        image = resizecrop(image, height, width)
        image = self.video_processor.preprocess(image, height=height, width=width)
        image = image.to(device=self.device, dtype=self.dtype).unsqueeze(2)
        image = self.vae.encode(image).to(torch.float32)
        return image

    def decode_video(self, latent: torch.Tensor, group: torch.distributed.ProcessGroup = None):
        def _to_thwc_u8(x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 4:
                raise ValueError(f"Unexpected decoded video tensor rank: {x.dim()} shape={tuple(x.shape)}")
            if x.shape[-1] in (1, 3):
                y = x
            elif x.shape[0] in (1, 3):
                y = x.permute(1, 2, 3, 0)
            elif x.shape[1] in (1, 3):
                y = x.permute(0, 2, 3, 1)
            else:
                y = x.permute(1, 2, 3, 0)
            y = y.contiguous()
            if y.dtype != torch.uint8:
                if torch.is_floating_point(y):
                    y = (y * 255).clamp(0, 255).to(torch.uint8)
                else:
                    y = y.to(torch.uint8)
            return y

        def _maybe_fix_vertical_stack(video_np: np.ndarray) -> np.ndarray:
            if env_is_true("MAGI_DISABLE_SPATIAL_FIX") or env_is_true("MAGI_DISABLE_VERTICAL_FIX"):
                return video_np
            if not isinstance(video_np, np.ndarray) or video_np.ndim != 4:
                return video_np
            if video_np.shape[-1] != 3:
                return video_np
            t, h, w, _ = video_np.shape
            if t < 2 or h < 256:
                return video_np

            frame0 = video_np[0]
            gray = frame0.astype(np.int16).mean(axis=2)
            d = np.mean(np.abs(gray[1:] - gray[:-1]), axis=1)
            if d.size < 32:
                return video_np

            k = int(min(12, d.size))
            top_idx = np.argsort(-d)[:k]
            top_idx = np.sort(top_idx + 1)
            if top_idx.size < 2:
                return video_np

            cuts: list[int] = []
            for r in top_idx.tolist():
                if not cuts or abs(r - cuts[-1]) > 8:
                    cuts.append(int(r))
            if len(cuts) < 2:
                return video_np

            cuts = [c for c in cuts if 8 <= c <= h - 8]
            if len(cuts) < 2:
                return video_np

            boundaries = [0] + cuts + [h]
            segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
            segments = [(s, e) for (s, e) in segments if e - s >= 64]
            if len(segments) < 2:
                return video_np

            heights = np.array([e - s for (s, e) in segments], dtype=np.int32)
            max_h = int(heights.max())
            if max_h >= int(h * 0.85):
                return video_np

            centers = np.array([(s + e) * 0.5 for (s, e) in segments], dtype=np.float32)
            score = heights.astype(np.float32) - 0.1 * np.abs(centers - (h * 0.5))
            best = int(np.argmax(score))
            s, e = segments[best]

            crop = video_np[:, s:e]
            if crop.shape[1] <= 0:
                return video_np
            crop = np.ascontiguousarray(crop)

            x = torch.from_numpy(crop).permute(0, 3, 1, 2).contiguous()
            x = x.to(dtype=torch.float32).div(255.0)
            x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            x = x.clamp(0, 1).mul(255.0).to(torch.uint8)
            out = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            return out

        def _maybe_fix_horizontal_repeat(video_np: np.ndarray) -> np.ndarray:
            if env_is_true("MAGI_DISABLE_SPATIAL_FIX") or env_is_true("MAGI_DISABLE_HORIZONTAL_FIX"):
                return video_np
            if not isinstance(video_np, np.ndarray) or video_np.ndim != 4:
                return video_np
            if video_np.shape[-1] != 3:
                return video_np
            t, h, w, _ = video_np.shape
            if t < 2 or w < 256:
                return video_np

            frame0 = video_np[0]
            gray = frame0.astype(np.float32).mean(axis=2)

            fix_mode = os.getenv("MAGI_HORIZONTAL_FIX_MODE", "").strip().lower()
            if fix_mode not in ("", "auto", "segment", "window", "half"):
                fix_mode = "auto"
            if fix_mode == "":
                fix_mode = "auto"

            force_side = os.getenv("MAGI_HORIZONTAL_CROP_SIDE", "").strip().lower()
            if force_side in ("left", "right"):
                mid = int(w // 2)
                s = 0 if force_side == "left" else mid
                e = s + mid
                if mid >= 64 and e <= w:
                    crop = video_np[:, :, s:e]
                    if crop.shape[2] >= 64:
                        crop = np.ascontiguousarray(crop)
                        x = torch.from_numpy(crop).permute(0, 3, 1, 2).contiguous()
                        x = x.to(dtype=torch.float32).div(255.0)
                        x = torch.nn.functional.interpolate(x, size=(h, w), mode="bicubic", align_corners=False)
                        x = x.clamp(0, 1).mul(255.0).to(torch.uint8)
                        out = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                        return out

            frac_str = os.getenv("MAGI_HORIZONTAL_CROP_FRACTION", "").strip()
            if frac_str:
                try:
                    frac = float(frac_str)
                except ValueError:
                    frac = 0.0
                if 0.1 <= frac < 1.0:
                    offset_str = os.getenv("MAGI_HORIZONTAL_CROP_OFFSET", "").strip()
                    try:
                        offset = float(offset_str) if offset_str else 0.0
                    except ValueError:
                        offset = 0.0
                    offset = max(-1.0, min(1.0, offset))

                    win = int(round(float(w) * float(frac)))
                    win = max(64, min(win, w))
                    center = (float(w) * 0.5) + offset * (float(w) * 0.25)
                    start = int(round(center - float(win) * 0.5))
                    start = max(0, min(start, w - win))
                    end = start + win

                    crop = video_np[:, :, start:end]
                    if crop.shape[2] >= 64:
                        crop = np.ascontiguousarray(crop)
                        x = torch.from_numpy(crop).permute(0, 3, 1, 2).contiguous()
                        x = x.to(dtype=torch.float32).div(255.0)
                        x = torch.nn.functional.interpolate(x, size=(h, w), mode="bicubic", align_corners=False)
                        x = x.clamp(0, 1).mul(255.0).to(torch.uint8)
                        out = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                        return out

            if fix_mode in ("auto", "half"):
                mid = int(w // 2)
                if mid >= 64 and (w - mid) >= 64:
                    left = gray[:, :mid]
                    right = gray[:, mid : mid + mid]
                    if right.shape[1] == left.shape[1]:
                        var = float(np.var(gray)) + 1e-6
                        mse_lr = float(np.mean((left - right) ** 2)) / var
                        left_energy = float(np.mean(np.abs(left[:, 1:] - left[:, :-1])))
                        right_energy = float(np.mean(np.abs(right[:, 1:] - right[:, :-1])))
                        thr_str = os.getenv("MAGI_HORIZONTAL_REPEAT_MSE_THR", "").strip()
                        try:
                            mse_thr = float(thr_str) if thr_str else 0.15
                        except ValueError:
                            mse_thr = 0.15
                        mse_thr = max(0.0, min(mse_thr, 2.0))
                        if fix_mode == "half" or mse_lr < mse_thr:
                            pick_left = left_energy >= right_energy
                            s = 0 if pick_left else mid
                            e = s + mid
                            crop = video_np[:, :, s:e]
                            if crop.shape[2] >= 64:
                                crop = np.ascontiguousarray(crop)
                                x = torch.from_numpy(crop).permute(0, 3, 1, 2).contiguous()
                                x = x.to(dtype=torch.float32).div(255.0)
                                x = torch.nn.functional.interpolate(x, size=(h, w), mode="bicubic", align_corners=False)
                                x = x.clamp(0, 1).mul(255.0).to(torch.uint8)
                                out = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                                return out

            d = np.mean(np.abs(gray[:, 1:] - gray[:, :-1]), axis=0)
            if fix_mode in ("auto", "segment") and d.size >= 32:
                k = int(min(24, d.size))
                top_idx = np.argsort(-d)[:k]
                top_idx = np.sort(top_idx + 1)
                cuts: list[int] = []
                for x in top_idx.tolist():
                    if not cuts or abs(x - cuts[-1]) > 8:
                        cuts.append(int(x))
                cuts = [c for c in cuts if 8 <= c <= w - 8]
                if len(cuts) >= 2:
                    boundaries = [0] + cuts + [w]
                    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
                    segments = [(s, e) for (s, e) in segments if e - s >= 64]
                    if len(segments) >= 2:
                        widths = np.array([e - s for (s, e) in segments], dtype=np.int32)
                        max_w = int(widths.max())
                        if max_w < int(w * 0.85):
                            col_energy = np.var(gray, axis=0)
                            col_energy = col_energy / (float(col_energy.mean()) + 1e-6)

                            min_seg_w = int(max(128, w * 0.30))
                            valid_segments = [(s, e) for (s, e) in segments if (e - s) >= min_seg_w]
                            if len(valid_segments) >= 2:
                                seg_energy = np.array(
                                    [float(np.sum(col_energy[s:e])) for (s, e) in valid_segments], dtype=np.float32
                                )
                                seg_w = np.array([float(e - s) for (s, e) in valid_segments], dtype=np.float32)
                                seg_score = seg_energy / (seg_w + 1e-6)
                                best_seg = int(np.argmax(seg_score))
                                seg_s, seg_e = valid_segments[best_seg]

                                win_base = int(w / max(1, len(valid_segments)))
                                win = int(max(w // 3, win_base))
                                seg_wi = int(seg_e - seg_s)
                                if seg_wi < win:
                                    win = seg_wi
                                if win < 64:
                                    return video_np

                                xs = np.arange(seg_s, seg_e, dtype=np.float32)
                                weights = col_energy[seg_s:seg_e].astype(np.float32)
                                wsum = float(np.sum(weights))
                                center = (
                                    float((np.sum(xs * weights) / (wsum + 1e-6)))
                                    if wsum > 0
                                    else float((seg_s + seg_e) * 0.5)
                                )

                                start = int(round(center - win * 0.5))
                                start = max(seg_s, min(start, seg_e - win))
                                end = start + win

                                crop = video_np[:, :, start:end]
                                if crop.shape[2] >= 64:
                                    crop = np.ascontiguousarray(crop)
                                    x = torch.from_numpy(crop).permute(0, 3, 1, 2).contiguous()
                                    x = x.to(dtype=torch.float32).div(255.0)
                                    x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
                                    x = x.clamp(0, 1).mul(255.0).to(torch.uint8)
                                    out = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                                    return out

            col_energy = np.var(gray, axis=0)
            col_energy = col_energy / (float(col_energy.mean()) + 1e-6)

            if fix_mode not in ("auto", "window"):
                return video_np

            best = None
            best_score = 0.0
            best_ratio = 0.0
            for frac in (2, 3, 4):
                win = int(w // frac)
                if win < 128 or win >= int(w * 0.90):
                    continue
                cs = np.concatenate([[0.0], np.cumsum(col_energy, dtype=np.float64)])
                sums = cs[win:] - cs[:-win]
                if sums.size <= 0:
                    continue
                start = int(np.argmax(sums))
                ratio = float(sums[start] / (float(sums.mean()) + 1e-6))
                score = float(ratio * ((float(win) / float(w)) ** 0.2))
                if score > best_score:
                    best_score = score
                    best_ratio = ratio
                    best = (start, start + win)

            if best is None or best_ratio < 1.20:
                return video_np

            s, e = best
            win = int(e - s)
            min_win = int(max(128, w // 2))
            if win < min_win:
                center = float(s + e) * 0.5
                win = min_win
                start = int(round(center - win * 0.5))
                start = max(0, min(start, w - win))
                s = start
                e = start + win
            s = max(0, min(int(s), w - 1))
            e = max(s + 1, min(int(e), w))
            crop = video_np[:, :, s:e]
            if crop.shape[2] < 64:
                return video_np
            crop = np.ascontiguousarray(crop)

            x = torch.from_numpy(crop).permute(0, 3, 1, 2).contiguous()
            x = x.to(dtype=torch.float32).div(255.0)
            x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            x = x.clamp(0, 1).mul(255.0).to(torch.uint8)
            out = x.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            return out

        def _looks_like_vertical_tiling(sample_video: torch.Tensor) -> bool:
            if not isinstance(sample_video, torch.Tensor) or sample_video.dim() != 5:
                return False
            if sample_video.shape[0] < 1:
                return False
            video_4d = sample_video[0].detach().cpu()
            if video_4d.dim() != 4:
                return False
            frame_thwc = _to_thwc_u8(video_4d)[0].numpy()
            h = int(frame_thwc.shape[0])
            if h < 512:
                return False
            a = frame_thwc.astype(np.float32)
            m = min(256, h // 2)
            errs = []
            for p in range(h // 4, (3 * h) // 4, 8):
                x = a[:m]
                y = a[p : p + m]
                if y.shape[0] != m:
                    continue
                errs.append(float(np.mean((x - y) ** 2)))
            if len(errs) < 10:
                return False
            errs_np = np.asarray(errs, dtype=np.float32)
            med = float(np.median(errs_np))
            if med <= 0:
                return False
            ratio = float(errs_np.min() / med)
            return ratio < 0.25

        def _decode_with_wan_vae() -> torch.Tensor:
            return self.vae.decode(latent.to(self.dtype), group=group)

        disable_turbo = env_is_true("MAGI_DISABLE_TURBO_VAE")

        videos = None
        if self.config.use_turbo_vae and (not disable_turbo):
            is_memory_limited = env_is_true("CPU_OFFLOAD") and env_is_true("SR2_1080")
            videos = self.turbo_vae.decode(latent.to(self.dtype), output_offload=is_memory_limited).float()
            tiled = False
            try:
                tiled = _looks_like_vertical_tiling(videos)
            except Exception:
                tiled = False
            if tiled:
                print_rank_0("TurboVAE decode looks tiled; falling back to Wan2.2 VAE decode. Set MAGI_DISABLE_TURBO_VAE=1 to disable TurboVAE.")
                videos = _decode_with_wan_vae()
        else:
            videos = _decode_with_wan_vae()

        if videos is None:
            return None
        videos.mul_(0.5).add_(0.5).clamp_(0, 1)

        fixed = []
        for video in videos:
            v = _to_thwc_u8(video.cpu()).numpy()
            v = _maybe_fix_vertical_stack(v)
            v = _maybe_fix_horizontal_repeat(v)
            fixed.append(v)
        videos = fixed
        return videos

    def post_process(self, latent_video: torch.Tensor, latent_audio: torch.Tensor):
        torch.cuda.empty_cache()
        # CTHW -> THWC
        videos_np = self.decode_video(latent_video, group=get_cp_group())
        torch.cuda.empty_cache()
        event_path_timer().synced_record("Step7: Decode Audio", print_fn=print_rank_last)

        if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
            video_np = videos_np[0]

            latent_audio = latent_audio.squeeze(0)
            if env_is_true("MAGI_T2V_FREEZE_AUDIO") and torch.all(latent_audio == 0):
                duration = float(video_np.shape[0]) / float(self.fps)
                num_samples = int(round(duration * float(self.audio_vae.sample_rate)))
                audio_output_np = np.zeros((num_samples, 2), dtype=np.float32)
            else:
                audio_output = self.audio_vae.decode(latent_audio.T)
                audio_output_np = audio_output.squeeze(0).T.cpu().numpy()
                audio_output_np = resample_audio_sinc(audio_output_np, 441 / 512)

            return video_np, audio_output_np
        else:
            return None, None
