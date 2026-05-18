set -e

DIM=8192
PERCENTAGE=0.05
MODEL="llama3-8b"
GRADIENT_PATH=./grads/${MODEL}-p${PERCENTAGE}-lora-seed3/{}-ckpt{}-adam/dim${DIM}
TRAIN_FILE_NAMES="flan_v2 cot dolly oasst1"
CKPTS="105 211 317 420"

CHECKPOINT_WEIGHTS="1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06" # average lr of the epoch

VALIDATION_GRADIENT_PATH=./grads/${MODEL}-p${PERCENTAGE}-lora-seed3/{}-ckpt{}-sgd/dim${DIM}
TARGET_TASK_NAMES="tydiqa mmlu bbh"
SELECTED_DATA_OUTPUT_PATH=./data/${MODEL}/seed_selected_p${PERCENTAGE}

bash ./seed/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"

python3 -m seed.data_selection.write_selected_data \
    --target_task_names ${TARGET_TASK_NAMES} \
    --train_file_names ${TRAIN_FILE_NAMES} \
    --train_files ./data/train/processed/flan_v2/flan_v2_data.jsonl ./data/train/processed/cot/cot_data.jsonl ./data/train/processed/dolly/dolly_data.jsonl ./data/train/processed/oasst1/oasst1_data.jsonl \
    --output_path $SELECTED_DATA_OUTPUT_PATH \
    --percentage $PERCENTAGE
