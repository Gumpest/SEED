set -e

source evaluation/eval_tydiqa.sh
source evaluation/eval_mmlu.sh
source evaluation/eval_bbh.sh

BASE="./out"
MODEL="llama3-8b"

for percentage in 0.05; do
  for task in tydiqa mmlu bbh; do
    for step in 105 211 317 420; do
      eval_${task} "$BASE/${MODEL}-seed-selected-p${percentage}-lora-${task}/checkpoint-$step"
    done
  done
done