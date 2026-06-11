# inference/scripts

训练 → 推理 → VBench 脚本目录（复现自旧工作区 `daVinci-MagiHuman2`）。

## 目录结构

```
inference/scripts/
├── README.md
├── logs/
├── train_zimage_480x800_from_base.sh   # → example/train_t2i_1w_480x800.sh
├── vbench/          # VBench 2.0 全量：生成 → 评测 → 监控
├── sample/          # manifest 小规模采样试跑
└── archive/         # 历史入口 / 480×800 专用
```

根目录保留**旧命令名**薄包装（`run_vbench2_zimage_standard_parallel.sh` 等），内部 `exec` 到新路径。

## 训练

```bash
cd /kwkj-k8s/cy123/workspace/DAVINCI

# 新名（推荐）
bash example/train_t2i_1w_480x800.sh start
bash example/train_t2i_1w_480x800.sh status

# 旧名（兼容）
bash inference/scripts/train_zimage_480x800_from_base.sh start
```

数据 manifest：`dataset/t2i_1w_480x800_f1/latent_manifest_train.jsonl`  
输出 ckpt：`output_train_t2i_1w_480x800_v1/ckpt_best.pt`

## 单条推理（legacy 参数）

```bash
GPU=0 bash example/legacy_zimage_inference.sh
GPU=0 bash example/legacy_zimage_inference_lowres.sh   # 448×256
```

底层：`inference/test_infer_zimage_native.py`（禁用 spatial fix，pad 导出）

## VBench 全量生成

| 分辨率 | 命令 |
|--------|------|
| **448×256**（旧默认，快） | `bash inference/scripts/run_vbench2_zimage_standard_parallel.sh` |
| **480×800**（竖屏） | `BR_WIDTH=480 BR_HEIGHT=800 bash inference/scripts/run_vbench2_zimage_standard_parallel.sh` |
| 480×800 便捷入口 | `bash example/run_vbench_zimage_480x800_ckptbest.sh` |
| 当前 ckpt 全量 | `bash example/run_t2i_1w_vbench_legacy_ckptbest.sh` |

长驻 worker：`vbench/worker.py`（每卡加载一次模型，**无 torchrun**）

## 一键后台（生成 → 评测）

```bash
# 448×256
bash inference/scripts/start_davinci_448_gen_monitor_bg.sh

# 480×800
BR_WIDTH=480 BR_HEIGHT=800 \
  OUTDIR=output_test_infer/zimage_ckptbest_vbench_standard_480x800_4s \
  bash inference/scripts/vbench/start_bg.sh

tail -f inference/scripts/logs/monitor_gen_then_eval_zimage.out
```

## 小规模采样

```bash
bash inference/scripts/run_first4_zimage_4s.sh      # manifest 前 4 条
GPU=0 bash inference/scripts/run_random_zimage_sample_20.sh
```

## 仅评测

```bash
EVAL_ONLY=1 bash example/run_t2i_1w_vbench_legacy_ckptbest.sh
# 或
VIDEOS_ROOT=output_test_infer/zimage_ckptbest_vbench_standard_480x800_4s \
  bash inference/scripts/evaluate_zimage_vbench_standard.sh
```

## 关键环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `BR_WIDTH` / `BR_HEIGHT` | 448 / 256 | 生成分辨率 |
| `OUTDIR` | 随分辨率自动 | 视频输出目录 |
| `CKPT` | `output_train_t2i_1w_480x800_v1/ckpt_best.pt` | finetune 权重 |
| `BASE` | `daVinci-MagiHuman-main/.../base` | base DiT |
| `SEED` | 521（VBench）/ 42（采样） | |
| `MAGI_DISABLE_SPATIAL_FIX` | 1 | 480×800 必开 |
| `MAGI_CANVAS_FIX` | 0 | |

## 路径对照（旧 → 新）

| 旧路径 | 新路径 |
|--------|--------|
| `/kwkj-k8s/davinci/LJH/daVinci-MagiHuman2` | `/kwkj-k8s/cy123/workspace/DAVINCI` |
| `output_train_zimage_480x800_from_base_v1` | `output_train_t2i_1w_480x800_v1` |
| `dataset/zimage_480x800_1w` | `dataset/t2i_1w_480x800_f1` |
| `models/daVinci-MagiHuman/base` | `daVinci-MagiHuman-main/.../base` |
