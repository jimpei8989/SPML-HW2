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

    model = CIFAR10_Model(cfg.model, load_weight=(cfg.task == "evaluation")).cuda()

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
            model, dataset_cfg=cfg.dataset, attack_cfg=cfg.attack, **cfg.misc
        )
        print(f"\nEvaluation finished. Time elapsed: {evaluation_time:.2f}s.")


def get_config():
    args = parse_arguments()

    default_recorder_root = Path("logs") / datetime.now().strftime(r"%m%d-%H%M")

    OmegaConf.register_resolver("path", lambda p: Path(p).absolute())
    return OmegaConf.merge(
        OmegaConf.create({"recorder_root": f"${{path:{default_recorder_root}}}"}),
        OmegaConf.load(args.base_config),
        OmegaConf.load(args.attack_config if args.task == "attack" else args.evaluate_config),
        OmegaConf.load(args.config),
        OmegaConf.create({"task": args.task}),
    )


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--config", type=lambda p: Path(p).absolute())
    parser.add_argument("--base_config", type=lambda p: Path(p), default="configs/base.yaml")
    parser.add_argument("--train_config", type=lambda p: Path(p), default="configs/pgd-at.yaml")
    parser.add_argument(
        "--evaluate_config", type=lambda p: Path(p), default="configs/evaluate.yaml"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
