import os
import json
import glob

BASE = "./out"

TASKS = {
    "tydiqa": "average.exact_match",
    "mmlu": "average_acc",
    "bbh": "average_exact_match",
}


def extract_score(task, data):
    if task == "tydiqa":
        return data["average"]["exact_match"]
    elif task == "mmlu":
        return data["average_acc"]
    elif task == "bbh":
        return data["average_exact_match"]
    else:
        raise ValueError(task)


def find_best_for_task(base, task):
    exp_dir = os.path.join(base, f"llama3-8b-seed-selected-p0.05-lora-{task}")

    pattern = os.path.join(exp_dir, "checkpoint-*", "eval", task, "metrics.json")
    files = glob.glob(pattern)
    print(pattern)

    if not files:
        print(f"❌ No files found for {task}")
        return None

    results = []

    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)

            score = extract_score(task, data)
            ckpt = f.split("checkpoint-")[-1].split("/")[0]

            results.append((ckpt, score, f))
        except Exception as e:
            print(f"⚠️ {task} failed on {f}: {e}")

    best = max(results, key=lambda x: x[1])
    return best, results


def main():
    print("\n========== Best checkpoint for each task ==========\n")

    summary = {}

    for task in TASKS:
        ret = find_best_for_task(BASE, task)
        if ret is None:
            continue

        best, results = ret

        summary[task] = best

        print(f"[{task.upper()}]")
        print(f"  Best checkpoint : checkpoint-{best[0]}")
        print(f"  Best score      : {best[1]:.4f}")
        print(f"  Metrics file    : {best[2]}")
        print()

    print("\n================== Summary Table ==================\n")
    print(f"{'Task':<10} | {'Checkpoint':<12} | {'Score':<8}")
    print("-" * 36)

    for task, (ckpt, score, _) in summary.items():
        print(f"{task:<10} | {ckpt:<12} | {score:<8.4f}")

    print("\n==================================================\n")


if __name__ == "__main__":
    main()
