import json
import subprocess as sp
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt


def run(output_dir, config_path, log_dir):
    scores_by_attack_iters = {
        0: [],
        8: [],
        16: [],
        32: [],
        64: [],
        128: [],
    }
    for epoch in range(0, 256 + 1, 4):
        scores = []

        for attack_iters in [8, 16, 32, 64, 128]:
            command = " ".join(
                [
                    "python3 src/main.py evaluate",
                    f"--config {config_path}",
                    f"--evaluate_config configs/evaluations/evaluate-{attack_iters}.yaml",
                    f"--weight_path {log_dir}/epoch_{epoch:03d}.pt" if epoch > 0 else "",
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

    with open(output_dir / "log.json", "w") as f:
        json.dump(scores_by_attack_iters, f)


def plot(output_dir):
    with open(output_dir / "log.json") as f:
        scores_by_attack_iters = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)

    colors = ["red", "orange", "yellow", "green", "blue", "purple"]

    for attack_iters, col in zip(scores_by_attack_iters, colors):
        ax.plot(
            list(range(0, 256 + 1, 4)),
            scores_by_attack_iters[attack_iters],
            color=col,
            label=("benign" if int(attack_iters) == 0 else f"adv_{attack_iters}"),
        )

    ax.set_title("Accuracies v.s. Training Epochs")
    ax.set_xlabel("Training Epochs")
    # ax.set_xticks(list(range(0, 256 + 1, 16)))
    ax.set_ylabel("Accuracy")
    ax.legend()

    fig.savefig(output_dir / "output-eval.png")


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("log_dir", type=Path)

    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plot", action="store_true")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not args.output_dir.is_dir():
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.run:
        run(args.output_dir, args.config_path, args.log_dir)

    if args.plot:
        plot(args.output_dir)


main()
