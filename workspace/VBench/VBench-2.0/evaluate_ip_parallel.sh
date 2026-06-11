#!/usr/bin/env bash

# Activate UV Environment
source /kwkj-k8s/YANG_LTX/Envs/VB/bin/activate
cd /kwkj-k8s/YANG_LTX/workspace/VBench/VBench-2.0
# Point to Local Checkpoints/Models
export VBENCH2_CACHE_DIR=/kwkj-k8s/YANG_LTX/Models/vbench
export HF_HOME=/kwkj-k8s/YANG_LTX/Models/huggingface
export TORCH_HOME=/kwkj-k8s/YANG_LTX/Models/torch

# Set up concurrency settings (Requested GPUs: 1, 3, 6, 7)
WORKERS=4
GPU_LIST="4,5,6,7"

# Target videos root and output root
VIDEOS_ROOT="/kwkj-k8s/YANG_LTX/workspace/RESULT/Videos/0512_3Wcheck_720P"
OUTPUT_ROOT="/kwkj-k8s/YANG_LTX/workspace/RESULT/Evaluation/0512_3Wcheck_720P"

dimensions=(
    "Instance_Preservation"
)

for dimension in "${dimensions[@]}"; do
    videos_path="${VIDEOS_ROOT}/${dimension}"
    output_path="${OUTPUT_ROOT}/${dimension}"

    mkdir -p "$output_path"

    echo "============================================================="
    echo "Starting Parallel VBench 2.0 Evaluation"
    echo "Dimension:  ${dimension}"
    echo "GPUs:       ${GPU_LIST}"
    echo "Workers:    ${WORKERS}"
    echo "Output Log: ${output_path}/eval_run_parallel.log"
    echo "============================================================="

    python evaluate.py \
        --videos_path "$videos_path" \
        --dimension "$dimension" \
        --output_path "$output_path" \
        --load_ckpt_from_local True \
        --num_workers "$WORKERS" \
        --gpu_ids "$GPU_LIST" > "${output_path}/eval_run_parallel.log" 2>&1

    echo "Evaluation for ${dimension} complete! Check outputs at: ${output_path}"
done

# Zip packing and final scoring automation 
echo "Aggregating results..."
ZIP_FILE="${OUTPUT_ROOT}/evaluation_results_ip_parallel.zip"
rm -f "$ZIP_FILE"

cd "$OUTPUT_ROOT"
python -c "import zipfile, glob, os; z = zipfile.ZipFile('$ZIP_FILE', 'w'); [z.write(f, os.path.basename(f)) for f in glob.glob('*/*.json')]; z.close()"

echo "Aggregating final scores using scripts/cal_final_score.py..."
cd "/kwkj-k8s/YANG_LTX/workspace/VBench/VBench-2.0"
python scripts/cal_final_score.py --zip_file "$ZIP_FILE" --model_name "vbench2_videos_2.3base"

echo "All tasks completed!"
