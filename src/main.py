from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from omegaconf.omegaconf import OmegaConf

from modules.recorder import Recorder
from modules.models import CIFAR10_Model

from modules.train import train
from modules.evaluate import evaluate


def main():
    cfg = get_config()

    model = CIFAR10_Model(
        cfg.model, cfg.recorder_root, load_weight=(cfg.task == "evaluate")
    ).cuda()

    if cfg.task == "train":
        recorder = Recorder(cfg.recorder_root, cfg)
        train_time, _ = train(
            model,
            dataset_cfg=cfg.dataset,
            attack_cfg=cfg.attack,
            optimizer_cfg=cfg.optimizer,
            recorder=recorder,
            **cfg.misc,
        )
        print(f"\nAdversarial training finished. Time elapsed: {train_time:.2f}s.")

    elif cfg.task == "evaluate":
        evaluation_time, _ = evaluate(
            model,
            dataset_cfg=cfg.dataset,
            attack_cfg=cfg.attack,
            output_dir=cfg.output_root,
            **cfg.misc,
        )
        print(f"\nEvaluation finished. Time elapsed: {evaluation_time:.2f}s.")


def get_config():
    args = parse_arguments()

    OmegaConf.register_resolver("path", lambda p: Path(p).absolute())
    return OmegaConf.merge(
        OmegaConf.create(
            {
                "recorder_root": f"${{path:logs/{args.name}}}",
                "output_root": f"${{path:results/{args.name}}}",
            }
        ),
        OmegaConf.load(args.base_config),
        OmegaConf.load(args.attack_config if args.task == "attack" else args.evaluate_config),
        OmegaConf.load(args.config),
        OmegaConf.create({"task": args.task}),
    )


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--config", type=lambda p: Path(p).absolute())
    parser.add_argument("--name", default=datetime.now().strftime(r"%m%d-%H%M"))
    parser.add_argument("--base_config", type=lambda p: Path(p), default="configs/base.yaml")
    parser.add_argument("--train_config", type=lambda p: Path(p), default="configs/pgd-at.yaml")
    parser.add_argument(
        "--evaluate_config", type=lambda p: Path(p), default="configs/evaluate.yaml"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
