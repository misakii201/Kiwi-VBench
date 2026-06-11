import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

os.environ.setdefault("DAVINCI_DISABLE_MAGI_COMPILE", "1")
os.environ.setdefault("MAGI_DISABLE_COMPILE", "1")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inference.common import parse_config
from inference.infra.checkpoint import load_model_checkpoint
from inference.model.dit.dit_module import DiTModel
from inference.pipeline.data_proxy import MagiDataProxy


@dataclass
class EvalInput:
    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: torch.Tensor | list[int]
    txt_feat: torch.Tensor
    txt_feat_len: torch.Tensor | list[int]


@dataclass
class ManifestItem:
    video_latent_path: str
    audio_latent_path: str
    prompt: str
    audio_len: int
    video_path: Optional[str] = None


def _load_manifest_item(manifest_path: str, index: int) -> ManifestItem:
    items: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if not items:
        raise ValueError(f"Empty manifest: {manifest_path}")
    if index < 0 or index >= len(items):
        raise IndexError(f"Index out of range: index={index}, len={len(items)}")
    obj = items[index]
    return ManifestItem(
        video_latent_path=str(obj["video_latent_path"]),
        audio_latent_path=str(obj["audio_latent_path"]),
        prompt=str(obj.get("prompt", "")),
        audio_len=int(obj.get("audio_len", 0)),
        video_path=obj.get("video_path"),
    )


def _make_dummy_txt_feat(
    *,
    device: torch.device,
    length: int,
    dim: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, List[int]]:
    txt = torch.zeros(1, length, dim, device=device, dtype=dtype)
    return txt, [length]


def _resolve_device(name: str) -> torch.device:
    name = name.strip().lower()
    if name == "cpu":
        return torch.device("cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        if name == "cuda":
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            return torch.device(f"cuda:{local_rank}")
        return torch.device(name)
    raise ValueError(f"Unknown device: {name}")


def _parse_amp_dtype(name: str) -> torch.dtype:
    name = name.strip().lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown amp dtype: {name}")


def _validate_latents(video_lat: torch.Tensor, audio_lat: torch.Tensor, *, expected_z_dim: int) -> None:
    if video_lat.dim() != 4:
        raise ValueError(f"video_latent must be 4D (C,T,H,W). Got: {tuple(video_lat.shape)}")
    if audio_lat.dim() != 2:
        raise ValueError(f"audio_latent must be 2D (T,64). Got: {tuple(audio_lat.shape)}")
    if video_lat.shape[0] != expected_z_dim:
        raise ValueError(f"video_latent C mismatch: got {video_lat.shape[0]}, expected {expected_z_dim}")
    if audio_lat.shape[-1] != 64:
        raise ValueError(f"audio_latent last dim mismatch: got {audio_lat.shape[-1]}, expected 64")


def _print_tensor_stats(name: str, x: torch.Tensor) -> None:
    xf = x.detach().float().reshape(-1)
    if xf.numel() == 0:
        print(f"{name}: empty tensor {tuple(x.shape)} {x.dtype} {x.device}")
        return
    print(
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
        f"min={xf.min().item():.6g} max={xf.max().item():.6g} mean={xf.mean().item():.6g} std={xf.std().item():.6g}"
    )


def _load_pt_checkpoint_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported ckpt type: {type(ckpt)}")

    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    else:
        raise KeyError(f"Cannot find model weights in ckpt keys: {list(ckpt.keys())}")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def build_dit_for_infer(
    model_config,
    engine_config,
    device: torch.device,
    *,
    base_ckpt_dir: str | None = None,
    ckpt_blend_alpha: float = 1.0,
    forced_load_path: str | None = None,
) -> DiTModel:
    model = DiTModel(model_config=model_config)
    
    # 取出真正的加载路径（可能是字符串或含有路径属性的对象）
    load_path = str(forced_load_path) if forced_load_path else (str(engine_config.load) if engine_config.load is not None else "")
    
    # 强制清理可能带有的路径两端的空白字符
    load_path = load_path.strip()
    
    print(f"Debug: build_dit_for_infer load_path='{load_path}' (forced_load_path={forced_load_path})")
    print(f"Debug: is_file={os.path.isfile(load_path)}")
    
    if load_path.endswith(".pt") and os.path.isfile(load_path):
        alpha = float(ckpt_blend_alpha)
        if base_ckpt_dir and 0.0 <= alpha < 1.0:
            base_engine_config = engine_config.model_copy(update={"load": str(base_ckpt_dir)})
            model = load_model_checkpoint(model, base_engine_config)
            base_state = model.state_dict()
            finetune_state = _load_pt_checkpoint_state_dict(load_path)
            blended_state: Dict[str, torch.Tensor] = {}
            for k, v_base in base_state.items():
                v_ft = finetune_state.get(k, None)
                if v_ft is None:
                    blended_state[k] = v_base
                    continue
                if torch.is_floating_point(v_base) and torch.is_floating_point(v_ft):
                    blended = v_base.float().mul(1.0 - alpha).add(v_ft.float().mul(alpha)).to(v_base.dtype)
                    blended_state[k] = blended
                else:
                    blended_state[k] = v_ft
            missing_keys, unexpected_keys = model.load_state_dict(blended_state, strict=False)
        else:
            # 单独加载一个 pt 文件时，如果我们不先加载 base 模型，有可能会缺失部分参数。
            # 为了防止这种缺失，只要提供了 base_ckpt_dir，就算 alpha=1.0（全量用 pt），也先 load 一遍 base 兜底，
            # 之后再用 pt 的参数去全量覆盖它。这样既能读到 pt 里的最新权重，又能补齐未训练的层。
            if base_ckpt_dir:
                base_engine_config = engine_config.model_copy(update={"load": str(base_ckpt_dir)})
                model = load_model_checkpoint(model, base_engine_config)
                
            state_dict = _load_pt_checkpoint_state_dict(load_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
        if missing_keys:
            print("Load Weight Missing Keys:", missing_keys)
        if unexpected_keys:
            print("Load Weight Unexpected Keys:", unexpected_keys)
        
        # 为了防止后续流程再次执行 load_model_checkpoint，我们把 config 里的 load 设空
        engine_config.load = ""
    elif load_path:
        model = load_model_checkpoint(model, engine_config)
    
    model = model.to(device=device, dtype=model_config.params_dtype)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--prompt_files", action="append", default=[])
    parser.add_argument("--prompts", action="append", default=[])
    parser.add_argument("--prompts_csv", type=str, default=None)
    parser.add_argument("--prompts_csv_start", type=int, default=0)
    parser.add_argument("--prompts_csv_num", type=int, default=4)
    parser.add_argument("--prompts_csv_caption_col", type=str, default="caption")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--save_path_prefix", type=str, default="output_test_infer/magi_test")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--br_width", type=int, default=None)
    parser.add_argument("--br_height", type=int, default=None)
    parser.add_argument("--output_width", type=int, default=None)
    parser.add_argument("--output_height", type=int, default=None)
    parser.add_argument("--upsample_mode", type=str, default=None, choices=["bilinear", "nearest", "bicubic"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--video_cfg_scale", type=float, default=None)
    parser.add_argument("--audio_cfg_scale", type=float, default=None)
    parser.add_argument("--cfg_number", type=int, default=None, choices=[1, 2])
    parser.add_argument("--freeze_audio", action="store_true")
    parser.add_argument("--disable_guidance_schedule", action="store_true")
    parser.add_argument("--guidance_low_scale", type=float, default=None)
    parser.add_argument("--flip_guidance_schedule", action="store_true")
    parser.add_argument("--guidance_max", type=float, default=None)
    parser.add_argument("--cfg_rescale", type=float, default=None)
    parser.add_argument("--frame_receptive_field", type=int, default=None)
    parser.add_argument("--base_ckpt_dir", type=str, default=None)
    parser.add_argument("--ckpt_blend_alpha", type=float, default=1.0)
    parser.add_argument("--latent_t", type=int, default=7)
    parser.add_argument("--latent_h", type=int, default=32)
    parser.add_argument("--latent_w", type=int, default=32)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--config-load-path", type=str, default=None)
    parser.add_argument("--data_only", action="store_true")
    args, _ = parser.parse_known_args()

    try:
        if args.generate and not args.config_load_path:
            default_config_path = os.path.join(_PROJECT_ROOT, "example", "base", "config.json")
            if os.path.exists(default_config_path):
                args.config_load_path = default_config_path

        device = _resolve_device(args.device)
        if device.type != "cuda":
            raise RuntimeError(
                "This test relies on CUDA-only local-attention range tensors (data_proxy.py currently hardcodes .to('cuda')). "
                "Please run with --device cuda."
            )
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        torch.cuda.set_device(device.index if device.index is not None else int(os.environ.get("LOCAL_RANK", "0")))
        from inference.infra import initialize_infra

        initialize_infra()

        old_argv = sys.argv
        if args.config_load_path:
            sys.argv = [old_argv[0], "--config-load-path", args.config_load_path]
        config = parse_config()
        sys.argv = old_argv

        if args.frame_receptive_field is not None:
            config.evaluation_config.data_proxy_config.frame_receptive_field = int(args.frame_receptive_field)

        amp_dtype = _parse_amp_dtype(args.amp_dtype)
        config.arch_config.params_dtype = amp_dtype

        prompt_list: List[str] = []
        if args.prompts_csv:
            import csv

            csv_path = str(args.prompts_csv)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(csv_path)
            start = max(0, int(args.prompts_csv_start))
            num = max(1, int(args.prompts_csv_num))
            caption_col = str(args.prompts_csv_caption_col or "caption")
            rows: List[str] = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i < start:
                        continue
                    if len(rows) >= num:
                        break
                    cap = str(row.get(caption_col, "")).strip()
                    if cap and cap.lower() != "nan":
                        rows.append(cap)
            prompt_list = rows
        elif args.prompt_files:
            for p in args.prompt_files:
                if not p:
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    prompt_list.append(f.read())
        elif args.prompts:
            prompt_list = [str(p) for p in args.prompts if str(p).strip()]
        elif args.prompt_file:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt_list = [f.read()]
        else:
            if args.prompt is not None:
                prompt_list = [str(args.prompt)]

        if not prompt_list:
            for idx in ("01", "02", "03", "04"):
                p = os.path.join(_PROJECT_ROOT, "example", "assets", f"prompt{idx}.txt")
                if not os.path.exists(p):
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    prompt_list.append(f.read())

        prompt_list = [p.strip() for p in prompt_list if p and str(p).strip()]
        if not prompt_list:
            raise ValueError("Prompt is empty. Provide --prompt/--prompts/--prompt_file/--prompt_files.")
        prompt_for_feature = prompt_list[0]
        if config.evaluation_config.txt_model_path:
            from inference.pipeline.prompt_process import get_padded_t5_gemma_embedding

            txt_feat, _ = get_padded_t5_gemma_embedding(
                prompt_for_feature,
                config.evaluation_config.txt_model_path,
                str(device),
                amp_dtype,
                config.evaluation_config.t5_gemma_target_length,
            )
            txt_len = [config.evaluation_config.t5_gemma_target_length]
        else:
            txt_feat, txt_len = _make_dummy_txt_feat(
                device=device, length=config.evaluation_config.t5_gemma_target_length, dim=3584, dtype=torch.float32
            )

        if args.manifest:
            item = _load_manifest_item(args.manifest, args.index)
            if not os.path.exists(item.video_latent_path):
                raise FileNotFoundError(item.video_latent_path)
            if not os.path.exists(item.audio_latent_path):
                raise FileNotFoundError(item.audio_latent_path)

            video_lat = torch.load(item.video_latent_path, map_location="cpu")
            audio_lat = torch.load(item.audio_latent_path, map_location="cpu")
            if not isinstance(video_lat, torch.Tensor) or not isinstance(audio_lat, torch.Tensor):
                raise TypeError(f"Latents must be torch.Tensor. Got: video={type(video_lat)}, audio={type(audio_lat)}")

            _validate_latents(video_lat, audio_lat, expected_z_dim=int(config.evaluation_config.z_dim))

            video_x_t = video_lat.unsqueeze(0).to(device=device, dtype=torch.float32)
            audio_x_t = audio_lat.unsqueeze(0).to(device=device, dtype=torch.float32)
            audio_feat_len = int(audio_lat.shape[0])
            if item.audio_len:
                audio_feat_len = min(audio_feat_len, int(item.audio_len))
        else:
            z_dim = int(config.evaluation_config.z_dim)
            t = int(args.latent_t)
            h = int(args.latent_h)
            w = int(args.latent_w)

            patch_size = int(config.evaluation_config.data_proxy_config.patch_size)
            t_patch_size = int(config.evaluation_config.data_proxy_config.t_patch_size)
            if t % t_patch_size != 0:
                raise ValueError(f"latent_t must be divisible by t_patch_size={t_patch_size}. Got latent_t={t}")
            if h % patch_size != 0 or w % patch_size != 0:
                raise ValueError(f"latent_h/latent_w must be divisible by patch_size={patch_size}. Got h={h}, w={w}")

            audio_feat_len = 4 * (t - 1) + 1
            video_x_t = torch.randn(1, z_dim, t, h, w, dtype=torch.float32, device=device)
            audio_x_t = torch.randn(1, audio_feat_len, 64, dtype=torch.float32, device=device)

        eval_input = EvalInput(
            x_t=video_x_t,
            audio_x_t=audio_x_t,
            audio_feat_len=[audio_feat_len],
            txt_feat=txt_feat,
            txt_feat_len=txt_len,
        )

        data_proxy = MagiDataProxy(config.evaluation_config.data_proxy_config)
        model_inputs = data_proxy.process_input(eval_input)
        print("process_input ok")
        print("model_inputs types:", [type(x) for x in model_inputs])
        for i, x in enumerate(model_inputs):
            if isinstance(x, torch.Tensor):
                _print_tensor_stats(f"model_inputs[{i}]", x)

        if args.data_only:
            return

        if args.ckpt_dir:
            config.engine_config.load = args.ckpt_dir

        # 强制更新 load_path 以覆盖后续调用时可能丢失的信息
        engine_load_path = config.engine_config.load

        if not config.engine_config.load:
            print(
                "No checkpoint configured. Skip forward and exit. Use --ckpt_dir or --config-load-path to enable forward."
            )
            return

        model = build_dit_for_infer(
            config.arch_config,
            config.engine_config,
            device,
            base_ckpt_dir=args.base_ckpt_dir,
            ckpt_blend_alpha=float(args.ckpt_blend_alpha),
            forced_load_path=str(engine_load_path),
        )

        if args.generate:
            if not config.evaluation_config.vae_model_path:
                raise ValueError("evaluation_config.vae_model_path is empty. Provide --config-load-path with a valid inference config.")
            if not config.evaluation_config.audio_model_path:
                raise ValueError(
                    "evaluation_config.audio_model_path is empty. Provide --config-load-path with a valid inference config."
                )
            if args.num_inference_steps is not None:
                config.evaluation_config.num_inference_steps = int(args.num_inference_steps)
            if args.video_cfg_scale is not None:
                config.evaluation_config.video_txt_guidance_scale = float(args.video_cfg_scale)
            if args.audio_cfg_scale is not None:
                config.evaluation_config.audio_txt_guidance_scale = float(args.audio_cfg_scale)
            if args.cfg_number is not None:
                config.evaluation_config.cfg_number = int(args.cfg_number)
            if args.freeze_audio:
                os.environ.setdefault("MAGI_T2V_FREEZE_AUDIO", "1")
            else:
                os.environ.pop("MAGI_T2V_FREEZE_AUDIO", None)
            if args.disable_guidance_schedule:
                os.environ.setdefault("MAGI_DISABLE_GUIDANCE_SCHEDULE", "1")
            else:
                os.environ.pop("MAGI_DISABLE_GUIDANCE_SCHEDULE", None)
            if args.guidance_low_scale is not None:
                os.environ["MAGI_GUIDANCE_LOW_SCALE"] = str(float(args.guidance_low_scale))
            else:
                os.environ.pop("MAGI_GUIDANCE_LOW_SCALE", None)
            if args.flip_guidance_schedule:
                os.environ.setdefault("MAGI_GUIDANCE_SCHEDULE_FLIP", "1")
            else:
                os.environ.pop("MAGI_GUIDANCE_SCHEDULE_FLIP", None)
            if args.guidance_max is not None:
                os.environ["MAGI_GUIDANCE_MAX"] = str(float(args.guidance_max))
            else:
                os.environ.pop("MAGI_GUIDANCE_MAX", None)
            if args.cfg_rescale is not None:
                os.environ["MAGI_CFG_RESCALE"] = str(float(args.cfg_rescale))
            else:
                os.environ.pop("MAGI_CFG_RESCALE", None)
            save_path_prefix = str(args.save_path_prefix)
            if not os.path.isabs(save_path_prefix):
                save_path_prefix = os.path.join(_PROJECT_ROOT, save_path_prefix)
            os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)

            from inference.pipeline.pipeline import MagiPipeline

            pipeline = MagiPipeline(model, config.evaluation_config, device=str(device))
            seconds = float(args.seconds) if args.seconds is not None else 4.0
            br_width = args.br_width if args.br_width is not None else 480
            br_height = args.br_height if args.br_height is not None else 272
            output_width = args.output_width if args.output_width is not None else br_width
            output_height = args.output_height if args.output_height is not None else br_height
            for i, p in enumerate(prompt_list):
                prefix_i = f"{save_path_prefix}_{i:02d}"
                save_path = pipeline.run_offline(
                    prompt=p,
                    image=None,
                    audio=None,
                    save_path_prefix=prefix_i,
                    seed=int(args.seed) + i,
                    seconds=float(seconds),
                    br_width=int(br_width),
                    br_height=int(br_height),
                    output_width=int(output_width),
                    output_height=int(output_height),
                    upsample_mode=args.upsample_mode,
                )
                print("saved_video:", save_path)
            return

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32 and device.type == "cuda")
            ):
                out = model(*model_inputs)
            v_pred_video, v_pred_audio = data_proxy.process_output(out)

        print("forward ok")
        _print_tensor_stats("v_pred_video", v_pred_video)
        _print_tensor_stats("v_pred_audio", v_pred_audio)

        if not torch.isfinite(v_pred_video).all():
            raise RuntimeError("v_pred_video contains NaN/Inf")
        if not torch.isfinite(v_pred_audio).all():
            raise RuntimeError("v_pred_audio contains NaN/Inf")
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
