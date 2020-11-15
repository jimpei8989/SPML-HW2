import random
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path

from omegaconf.omegaconf import OmegaConf

import numpy as np
import torch

from modules.recorder import Recorder

from modules.train import train
from modules.gen_adversarial import generate_adversarial_examples
from modules.evaluate import evaluate

SEED = 0x06902029


def seed_everything(weed):
    random.seed(weed)
    np.random.seed(weed)
    torch.manual_seed(weed)


def sec_to_readable(seconds):
    return str(timedelta(seconds=int(seconds)))


def main():
    seed_everything(SEED)

    args, cfg = get_config()

    if args.task == "train":
        recorder = Recorder(cfg.recorder_root, cfg)
        train_time, _ = train(
            model_cfg=cfg.model,
            dataset_cfg=cfg.dataset,
            attack_cfg=cfg.attack,
            optimizer_cfg=cfg.optimizer,
            recorder=recorder,
            **cfg.misc,
        )
        print(f"\nAdversarial training finished. Time elapsed: {sec_to_readable(train_time)}.")

    elif args.task == "evaluate":
        if args.gen_adv:
            adv_gen_time, _ = generate_adversarial_examples(
                model_cfg=cfg.eval.model,
                attack_cfg=cfg.attack,
                dataset_cfg=cfg.dataset,
                adv_images_dir=cfg.eval.adv_images_dir,
                **cfg.misc,
            )
            print(
                "\nAdversarial examples generation finished."
                + f"Time elapsed: {sec_to_readable(adv_gen_time)}."
            )

        evaluation_time, _ = evaluate(
            model_cfg=cfg.model,
            weight_path=args.weight_path,
            dataset_cfg=cfg.dataset,
            adv_images_dir=cfg.eval.adv_images_dir,
            output_dir=cfg.output_root,
            **cfg.misc,
        )
        print(f"\nEvaluation finished. Time elapsed: {sec_to_readable(evaluation_time)}.")


def get_config():
    args = parse_arguments()

    OmegaConf.register_resolver("path", lambda p: Path(p).absolute())
    return args, OmegaConf.merge(
        OmegaConf.create(
            {
                "recorder_root": f"${{path:logs/{args.name}}}",
                "output_root": f"${{path:results/{args.name}}}",
            }
        ),
        OmegaConf.load(args.base_config),
        OmegaConf.load(args.attack_config if args.task == "train" else args.evaluate_config),
        OmegaConf.load(args.config),
    )


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--name", default=datetime.now().strftime(r"%m%d-%H%M"))
    parser.add_argument("--weight_path", type=Path)
    parser.add_argument("--base_config", type=Path, default="configs/base.yaml")
    parser.add_argument("--attack_config", type=Path, default="configs/pgd-fixed.yaml")
    parser.add_argument("--evaluate_config", type=Path, default="configs/evaluate.yaml")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--gen_adv", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
