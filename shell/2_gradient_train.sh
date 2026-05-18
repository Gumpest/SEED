TRAINING_DATA_NAME=cot
TRAINING_DATA_FILE=./data/train/processed/cot/cot_data.jsonl

# TRAINING_DATA_NAME=dolly
# TRAINING_DATA_FILE=./data/train/processed/dolly/dolly_data.jsonl

# TRAINING_DATA_NAME=oasst1
# TRAINING_DATA_FILE=./data/train/processed/oasst1/oasst1_data.jsonl

# TRAINING_DATA_NAME=flan_v2
# TRAINING_DATA_FILE=./data/train/processed/flan_v2/flan_v2_data.jsonl

GRADIENT_TYPE="adam"
DIMS="8192"
MODEL="llama3-8b"
RATIO="0.05"

CKPTS=(105 211 317 420)
GPUS=(0 1 2 3)

LOG_DIR=logs
mkdir -p ${LOG_DIR}

for i in "${!CKPTS[@]}"; do
    CKPT=${CKPTS[$i]}
    GPU=${GPUS[$i]}

    MODEL_PATH=./out/${MODEL}-p${RATIO}-lora-seed3/checkpoint-${CKPT}
    OUTPUT_PATH=./grads/${MODEL}-p${RATIO}-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}

    CUDA_VISIBLE_DEVICES=${GPU} \
    bash ./seed/scripts/get_info/grad/get_train_lora_grads.sh \
        "$TRAINING_DATA_FILE" \
        "$MODEL_PATH" \
        "$OUTPUT_PATH" \
        "$DIMS" \
        "$GRADIENT_TYPE" \
        > ${LOG_DIR}/${TRAINING_DATA_NAME}-ckpt${CKPT}.log 2>&1 &
done

wait