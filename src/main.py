from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from omegaconf.omegaconf import OmegaConf

from torchvision.datasets import CIFAR10

from modules.recorder import Recorder
from modules.models import build_model
from modules.transforms import build_transform
from modules.train import train
from modules.evaluate import evaluate


def main():
    cfg = get_config()

    recorder = Recorder(cfg.recorder_root, cfg)

    model = build_model(cfg.model_name).cuda()

    train_transform, inference_transform = build_transform(cfg.dataset)

    train_dataset = CIFAR10(
        root=cfg.dataset.dataset_root,
        train=True,
        transform=train_transform,
        download=True,
    )

    validation_dataset = CIFAR10(
        root=cfg.dataset.dataset_root,
        train=False,
        transform=inference_transform,
        download=True,
    )

    if cfg.task == "train":
        train(
            model,
            train_dataset,
            validation_dataset,
            recorder=recorder,
            **cfg.misc,
        )
    elif cfg.task == "evaluate":
        evaluate(model, validation_dataset)


def get_config():
    args = parse_arguments()

    default_recorder_root = Path("logs") / datetime.now().strftime(r"%m%d-%H%M")

    OmegaConf.register_resolver("path", lambda p: Path(p).absolute())
    return OmegaConf.merge(
        OmegaConf.create({"recorder_root": f"${{path:{default_recorder_root}}}"}),
        OmegaConf.load(args.base_config),
        OmegaConf.load(args.config),
        OmegaConf.create({"task": args.task}),
    )


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("config", type=lambda p: Path(p).absolute())
    parser.add_argument(
        "--base_config", type=lambda p: Path(p), default="configs/base.yaml"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
