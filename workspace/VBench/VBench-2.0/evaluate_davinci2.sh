#!/usr/bin/env bash

# Activate UV Environment
source /kwkj-k8s/YANG_LTX/Envs/VB/bin/activate
cd /kwkj-k8s/YANG_LTX/workspace/VBench/VBench-2.0
# Point to Local Checkpoints/Models
export VBENCH2_CACHE_DIR=/kwkj-k8s/YANG_LTX/Models/vbench
export HF_HOME=/kwkj-k8s/YANG_LTX/Models/huggingface
export TORCH_HOME=/kwkj-k8s/YANG_LTX/Models/torch

# Define VBench dimensions


dimensions=(
    "Multi-View_Consistency"
)
out_dimension=(
"Instance_Preservation"
)


VIDEOS_ROOT="/kwkj-k8s/YANG_LTX/workspace/RESULT/Videos/0515_davinci_seedance_v1_step4000_sr2x_704x1280"
OUTPUT_ROOT="/kwkj-k8s/YANG_LTX/workspace/RESULT/Evaluation/0515_davinci_seedance_v1_step4000_sr2x_704x1280"

mkdir -p "$OUTPUT_ROOT"

# Healthy GPUs to use (cards 0 to 6)
gpus=(1)
num_gpus=${#gpus[@]}

echo "Starting VBench 2.0 evaluation on ${#dimensions[@]} dimensions using ${num_gpus} GPUs concurrently..."

for i in "${!dimensions[@]}"; do
    dimension=${dimensions[i]}
    gpu_id=${gpus[i % num_gpus]}

    videos_path="${VIDEOS_ROOT}/${dimension}"
    output_path="${OUTPUT_ROOT}/${dimension}"

    mkdir -p "$output_path"

    echo "Evaluating '${dimension}' on GPU ${gpu_id} with videos from: ${videos_path}"

    CUDA_VISIBLE_DEVICES=$gpu_id python evaluate.py \
        --videos_path "$videos_path" \
        --dimension "$dimension" \
        --output_path "$output_path" \
        --load_ckpt_from_local True > "${output_path}/eval_run.log" 2>&1 &

    sleep 2
done

# Wait for all parallel evaluations to complete
wait

echo "All dimensions evaluated! Creating zip package of JSON results..."
ZIP_FILE="${OUTPUT_ROOT}/evaluation_results_davinci.zip"
rm -f "$ZIP_FILE"

# Find all generated json files and zip them under flat structure
cd "$OUTPUT_ROOT"
python -c "import zipfile, glob, os; z = zipfile.ZipFile('$ZIP_FILE', 'w'); [z.write(f, os.path.basename(f)) for f in glob.glob('*/*.json')]; z.close()"

echo "Aggregating final scores using VBench-2.0 scripts/cal_final_score.py..."
cd "/kwkj-k8s/YANG_LTX/workspace/VBench/VBench-2.0"
python scripts/cal_final_score.py --zip_file "$ZIP_FILE" --model_name "davinci_seedance_v1_step4000"

echo "Evaluation process complete!"
