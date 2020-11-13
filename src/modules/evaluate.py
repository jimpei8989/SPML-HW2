import json
from functools import partial

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from modules.attacks import Attacker
from modules.datasets import JointDataset, build_dataset, build_adv_dataset
from modules.run_epoch import run_general_epoch
from modules.utils import timer


@timer
def evaluate(
    model,
    dataset_cfg=None,
    attack_cfg=None,
    output_dir=None,
    batch_size=1,
    num_workers=1,
    **kwargs,
):
    to_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=num_workers)

    train_dataset, validation_dataset = build_dataset(dataset_cfg, defense=False)
    def_train_dataset, def_validation_dataset = build_dataset(dataset_cfg, defense=True)

    attacker = Attacker(attack_cfg)
    num_iters = attacker.request_num_iters()

    (attack_train_time, adv_train_dataset), (attack_validation_time, adv_validation_dataset) = map(
        lambda p: attacker.attack(model, to_dataloader(p[0]), num_iters=num_iters, name=p[1]),
        zip(build_dataset(dataset_cfg, defense=False), ("attack_train", "attack_validation")),
    )

    adv_train_dataset.save_to_directory(output_dir / "train")
    adv_validation_dataset.save_to_directory(output_dir / "validation")

    def_adv_train_dataset, def_adv_validation_dataset = build_adv_dataset(
        output_dir, dataset_cfg, defense=True
    )

    print(f"Attacking time: {attack_train_time:.2f}s / {attack_validation_time:.2f}s")

    criterion = CrossEntropyLoss()
    run_epoch = partial(run_general_epoch, model=model, criterion=criterion, train=False)

    output = []

    def run_eval(dataset, split, defense=False):
        elapsed_time, log = run_epoch(
            to_dataloader(dataset), name=f"{'W' if defense else 'WO'} def {split}"
        )
        print(
            f"\nEvaluation on {split} set ({'w/' if defense else 'w/o'} preprocessing defense) "
            + f"[{elapsed_time:.2f}s]"
        )
        print("==> " + " - ".join(f"{k}: {log[k]:.4f}" for k in ["loss", "benign_acc", "adv_acc"]))

        output.append(
            {
                "split": split,
                "defense": defense,
                "time": elapsed_time,
                "log": log,
            }
        )

    run_eval(JointDataset(train_dataset, adv_train_dataset), "train", False)
    run_eval(JointDataset(validation_dataset, adv_validation_dataset), "validation", False)
    run_eval(JointDataset(def_train_dataset, def_adv_train_dataset), "train", True)
    run_eval(JointDataset(def_validation_dataset, def_adv_validation_dataset), "validation", True)

    with open(output_dir / "output.json", "w") as f:
        json.dump(output, f, indent=2)
