DATA_DIR=./data

# MODEL_PATH=../checkpoints/Llama-2-7b-hf
# MODEL_PATH=../checkpoints/Llama-2-13b-hf
MODEL_PATH=./checkpoints/Meta-Llama-3-8B

PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
DATA_SEED=3
JOB_NAME=llama3-8b-p${PERCENTAGE}-lora-seed${DATA_SEED}
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash ./seed/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"