import json
import subprocess as sp
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout

OUTPUT_DIR = Path("outputs/exp-resnet20")
if not OUTPUT_DIR.is_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run():
    scores_by_attack_iters = {
        0: [],
        8: [],
        16: [],
        32: [],
        64: [],
        128: [],
    }
    for epoch in range(0, 32 + 1):
        scores = []

        for attack_iters in [8, 16, 32, 64, 128]:
            command = " ".join(
                [
                    "python3 src/main.py evaluate",
                    "--config configs/experiments/resnet20-epoch/resnet20.yaml",
                    f"--evaluate_config configs/evaluations/evaluate-{attack_iters}.yaml",
                    f"--weight_path logs/exp-resnet20-epoch/epoch_{epoch:03d}.pt"
                    if epoch > 0
                    else "",
                    "--eval_val_only",
                    "--eval_def_only",
                    "--name tmp",
                ]
            )
            ret = sp.run(
                command,
                capture_output=True,
                shell=True,
                text=True,
            )
            score = list(
                map(lambda t: float(t.split(":")[1]), ret.stdout.split("\n")[0].split(" - ")[1:])
            )

            # Append benign accuracy
            scores.extend(score if attack_iters == 8 else score[1:])
            if attack_iters == 8:
                scores_by_attack_iters[0].append(scores[0])

            scores_by_attack_iters[attack_iters].append(score[1])

        print(f'{epoch} -> {" & ".join(map(lambda s: f"{s:.4f}", scores))}')

    with open(OUTPUT_DIR / "log.json", "w") as f:
        json.dump(scores_by_attack_iters, f)


def plot():
    with open(OUTPUT_DIR / "log.json") as f:
        scores_by_attack_iters = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

    colors = ["red", "orange", "yellow", "green", "blue", "purple"]

    for attack_iters, col in zip(scores_by_attack_iters, colors):
        ax.plot(
            scores_by_attack_iters[attack_iters],
            color=col,
            label=("benign" if int(attack_iters) == 0 else f"adv_{attack_iters}"),
        )

    ax.set_title("Accuracies v.s. Training Epochs")
    ax.set_xlabel("Training Epochs")
    ax.set_xticks(list(range(len(scores_by_attack_iters["0"]))))
    ax.set_ylabel("Accuracy")
    ax.legend()

    fig.savefig(OUTPUT_DIR / "output-eval.png")


plot()