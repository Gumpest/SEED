CKPTS=(105 211 317 420)
GPUS=(0 1 2 3)
PERCENTAGE=0.05
DIMS="8192"
MODEL="llama3-8b"

LOG_DIR=logs
mkdir -p ${LOG_DIR}

TASKS=("tydiqa" "mmlu" "bbh")

for TASK in "${TASKS[@]}"; do
  echo "===== Start TASK: ${TASK} ====="

  for i in "${!CKPTS[@]}"; do
      CKPT=${CKPTS[$i]}
      GPU=${GPUS[$i]}

      MODEL_PATH=./out/${MODEL}-p${PERCENTAGE}-lora-seed3/checkpoint-${CKPT}
      OUTPUT_PATH=./grads/${MODEL}-p${PERCENTAGE}-lora-seed3/${TASK}-ckpt${CKPT}-sgd
      DATA_DIR=./data

      CUDA_VISIBLE_DEVICES=${GPU} \
      bash ./seed/scripts/get_info/grad/get_eval_lora_grads.sh \
          "$TASK" "$DATA_DIR" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" \
          > ${LOG_DIR}/${TASK}-ckpt${CKPT}.log 2>&1 &
  done

  wait

  echo "===== Finished TASK: ${TASK} ====="
done

echo "All tasks completed."