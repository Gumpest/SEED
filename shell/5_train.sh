# Array of target task names
set -e
TASK_NAMES=("tydiqa" "mmlu" "bbh")

PERCENTAGE=0.05
MODEL="llama3-8b"
MODEL_PATH=./checkpoints/Meta-Llama-3-8B
    
for TARGET_TASK_NAME in "${TASK_NAMES[@]}"; do
    echo "Processing task: $TARGET_TASK_NAME"
    TRAIN_FILES=./data/${MODEL}/seed_selected_p${PERCENTAGE}/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl
    JOB_NAME=${MODEL}-seed-selected-p${PERCENTAGE}-lora-${TARGET_TASK_NAME}
    
    bash ./seed/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME"
    
    echo "Completed task: $TARGET_TASK_NAME"
done