#!/usr/bin/env bash
# VBench-2.0：Instance_Preservation（官方 base + SR 720P 视频）
# 与 evaluate_davinci_ip.sh 一致，路径对齐 config_davinci_vbench2_official_base_sr720.yaml

source /kwkj-k8s/YANG_LTX/Envs/VB/bin/activate
cd /kwkj-k8s/YANG_LTX/workspace/VBench/VBench-2.0
export VBENCH2_CACHE_DIR=/kwkj-k8s/YANG_LTX/Models/vbench
export HF_HOME=/kwkj-k8s/YANG_LTX/Models/huggingface
export TORCH_HOME=/kwkj-k8s/YANG_LTX/Models/torch

WORKERS=4
GPU_LIST="4,5,6,7"

VIDEOS_ROOT="/kwkj-k8s/YANG_LTX/workspace/RESULT/Videos/0515_davinci_official_base_sr2x_704x1280"
OUTPUT_ROOT="/kwkj-k8s/YANG_LTX/workspace/RESULT/Evaluation/0515_davinci_official_base_sr2x_704x1280"

dimensions=(
    "Instance_Preservation"
)

for dimension in "${dimensions[@]}"; do
    videos_path="${VIDEOS_ROOT}/${dimension}"
    output_path="${OUTPUT_ROOT}/${dimension}"

    mkdir -p "$output_path"

    echo "============================================================="
    echo "Official base SR720 | Instance_Preservation"
    echo "GPUs: ${GPU_LIST}  Workers: ${WORKERS}"
    echo "Log: ${output_path}/eval_run_parallel.log"
    echo "============================================================="

    python evaluate.py \
        --videos_path "$videos_path" \
        --dimension "$dimension" \
        --output_path "$output_path" \
        --load_ckpt_from_local True \
        --num_workers "$WORKERS" \
        --gpu_ids "$GPU_LIST" > "${output_path}/eval_run_parallel.log" 2>&1

    echo "Done: ${output_path}"
done

ZIP_FILE="${OUTPUT_ROOT}/evaluation_results_davinci_official_base_ip.zip"
rm -f "$ZIP_FILE"

cd "$OUTPUT_ROOT"
python -c "import zipfile, glob, os; z = zipfile.ZipFile('$ZIP_FILE', 'w'); [z.write(f, os.path.basename(f)) for f in glob.glob('*/*.json')]; z.close()"

cd "/kwkj-k8s/YANG_LTX/workspace/VBench/VBench-2.0"
python scripts/cal_final_score.py --zip_file "$ZIP_FILE" --model_name "davinci_official_base_sr720"

echo "All tasks completed."
