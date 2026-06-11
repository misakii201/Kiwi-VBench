#!/usr/bin/env bash
#
# VBench-2.0 全 18 维测评（legacy ckpt_best 生成视频）
# 17 维并行 evaluate.py + Instance_Preservation 多 worker
#
# 用法：
#   bash example/evaluate_t2i_1w_vbench_legacy_ckptbest.sh
#   VIDEOS_ROOT=... OUTPUT_ROOT=... GPUS=0,1,2,3,4,5,6,7 bash example/evaluate_t2i_1w_vbench_legacy_ckptbest.sh
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VBENCH_ROOT="/kwkj-k8s/cy123/workspace/VBench/VBench-2.0"
PYTHON_VB="/kwkj-k8s/YANG_LTX/Envs/VB/bin/python"

VIDEOS_ROOT="${VIDEOS_ROOT:-${PROJECT_ROOT}/output_vbench_t2i_1w_legacy_ckptbest}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output_vbench_t2i_1w_legacy_ckptbest_eval}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MODEL_NAME="${MODEL_NAME:-t2i_1w_legacy_ckptbest_480x800}"

export VBENCH2_CACHE_DIR="${VBENCH2_CACHE_DIR:-/kwkj-k8s/YANG_LTX/Models/vbench}"
export HF_HOME="${HF_HOME:-/kwkj-k8s/YANG_LTX/Models/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/kwkj-k8s/YANG_LTX/Models/torch}"

dimensions=(
  "Human_Anatomy" "Human_Identity" "Human_Clothes" "Diversity" "Composition"
  "Dynamic_Spatial_Relationship" "Dynamic_Attribute" "Motion_Order_Understanding"
  "Human_Interaction" "Complex_Landscape" "Complex_Plot" "Camera_Motion"
  "Motion_Rationality" "Mechanics" "Thermotics" "Material"
  "Multi-View_Consistency"
)

IFS=',' read -ra GPU_ARR <<< "$GPUS"
num_gpus="${#GPU_ARR[@]}"

mkdir -p "$OUTPUT_ROOT"
cd "$VBENCH_ROOT"

echo "[eval] videos: ${VIDEOS_ROOT}"
echo "[eval] output: ${OUTPUT_ROOT}"
echo "[eval] GPUs:   ${GPUS}"

for i in "${!dimensions[@]}"; do
  dimension="${dimensions[i]}"
  gpu_id="${GPU_ARR[$((i % num_gpus))]}"
  videos_path="${VIDEOS_ROOT}/${dimension}"
  output_path="${OUTPUT_ROOT}/${dimension}"

  if [[ ! -d "$videos_path" ]]; then
    echo "[warn] skip ${dimension}: no dir ${videos_path}"
    continue
  fi

  mkdir -p "$output_path"
  echo "[eval] ${dimension} on GPU ${gpu_id}"

  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_VB" evaluate.py \
    --videos_path "$videos_path" \
    --dimension "$dimension" \
    --output_path "$output_path" \
    --load_ckpt_from_local True \
    > "${output_path}/eval_run.log" 2>&1 &

  sleep 1
done

wait
echo "[eval] 17 dimensions done"

# Instance_Preservation（需多 GPU worker）
ip_dim="Instance_Preservation"
ip_videos="${VIDEOS_ROOT}/${ip_dim}"
ip_output="${OUTPUT_ROOT}/${ip_dim}"
if [[ -d "$ip_videos" ]]; then
  mkdir -p "$ip_output"
  IP_GPUS="${IP_GPUS:-${GPUS}}"
  IP_WORKERS="${IP_WORKERS:-4}"
  echo "[eval] ${ip_dim} workers=${IP_WORKERS} gpus=${IP_GPUS}"
  "$PYTHON_VB" evaluate.py \
    --videos_path "$ip_videos" \
    --dimension "$ip_dim" \
    --output_path "$ip_output" \
    --load_ckpt_from_local True \
    --num_workers "$IP_WORKERS" \
    --gpu_ids "$IP_GPUS" \
    > "${ip_output}/eval_run_parallel.log" 2>&1
else
  echo "[warn] skip ${ip_dim}: no dir ${ip_videos}"
fi

ZIP_FILE="${OUTPUT_ROOT}/evaluation_results_${MODEL_NAME}.zip"
rm -f "$ZIP_FILE"
cd "$OUTPUT_ROOT"
"$PYTHON_VB" -c "import zipfile, glob, os; z=zipfile.ZipFile('${ZIP_FILE}','w'); [z.write(f, os.path.basename(f)) for f in glob.glob('*/*.json')]; z.close()"

cd "$VBENCH_ROOT"
"$PYTHON_VB" scripts/cal_final_score.py --zip_file "$ZIP_FILE" --model_name "$MODEL_NAME"

REPORT="${OUTPUT_ROOT}/cal_local_scores_report.txt"
"$PYTHON_VB" cal_local_scores.py \
  --dir "$OUTPUT_ROOT" \
  --report "$REPORT" \
  --title "VBench-2.0 | t2i_1w legacy ckpt_best 480x800 cfg6 4s"

echo "[done] zip: ${ZIP_FILE}"
echo "[done] report: ${REPORT}"
