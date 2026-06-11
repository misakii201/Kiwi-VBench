#!/usr/bin/env bash
# VBench-2.0：官方 daVinci base + SR 720P 生成视频（见 config_davinci_vbench2_official_base_sr720.yaml）
# 与 evaluate_davinci.sh 逻辑一致；Instance_Preservation 见 evaluate_davinci_official_base_ip.sh

source /kwkj-k8s/YANG_LTX/Envs/VB/bin/activate
cd /kwkj-k8s/YANG_LTX/workspace/VBench/VBench-2.0
export VBENCH2_CACHE_DIR=/kwkj-k8s/YANG_LTX/Models/vbench
export HF_HOME=/kwkj-k8s/YANG_LTX/Models/huggingface
export TORCH_HOME=/kwkj-k8s/YANG_LTX/Models/torch

dimensions=(
    "Human_Anatomy" "Human_Identity" "Human_Clothes" "Diversity" "Composition"
    "Dynamic_Spatial_Relationship" "Dynamic_Attribute" "Motion_Order_Understanding"
    "Human_Interaction" "Complex_Landscape" "Complex_Plot" "Camera_Motion"
    "Motion_Rationality" "Mechanics" "Thermotics" "Material"
    "Multi-View_Consistency"
)

VIDEOS_ROOT="/kwkj-k8s/YANG_LTX/workspace/RESULT/Videos/0515_davinci_official_base_sr2x_704x1280"
OUTPUT_ROOT="/kwkj-k8s/YANG_LTX/workspace/RESULT/Evaluation/0515_davinci_official_base_sr2x_704x1280"

mkdir -p "$OUTPUT_ROOT"

gpus=(0 1 2 3 4 5 6)
num_gpus=${#gpus[@]}

echo "VBench 2.0 (official base SR720): ${#dimensions[@]} dimensions, ${num_gpus} GPUs"

for i in "${!dimensions[@]}"; do
    dimension=${dimensions[i]}
    gpu_id=${gpus[i % num_gpus]}

    videos_path="${VIDEOS_ROOT}/${dimension}"
    output_path="${OUTPUT_ROOT}/${dimension}"

    mkdir -p "$output_path"

    echo "Evaluating '${dimension}' on GPU ${gpu_id}: ${videos_path}"

    CUDA_VISIBLE_DEVICES=$gpu_id python evaluate.py \
        --videos_path "$videos_path" \
        --dimension "$dimension" \
        --output_path "$output_path" \
        --load_ckpt_from_local True > "${output_path}/eval_run.log" 2>&1 &

    sleep 2
done

wait

echo "Zipping JSON results..."
ZIP_FILE="${OUTPUT_ROOT}/evaluation_results_davinci_official_base.zip"
rm -f "$ZIP_FILE"

cd "$OUTPUT_ROOT"
python -c "import zipfile, glob, os; z = zipfile.ZipFile('$ZIP_FILE', 'w'); [z.write(f, os.path.basename(f)) for f in glob.glob('*/*.json')]; z.close()"

echo "cal_final_score.py..."
cd "/kwkj-k8s/YANG_LTX/workspace/VBench/VBench-2.0"
python scripts/cal_final_score.py --zip_file "$ZIP_FILE" --model_name "davinci_official_base_sr720"

echo "Done. Videos: ${VIDEOS_ROOT}  Evaluation: ${OUTPUT_ROOT}"
