import json
from functools import partial

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from .models import CIFAR10_Model
from .datasets import JointDataset, build_dataset, build_adv_dataset
from .run_epoch import run_general_epoch
from .utils import timer


@timer
def evaluate(
    model_cfg=None,
    weight_path=None,
    dataset_cfg=None,
    adv_images_dir=None,
    output_dir=None,
    batch_size=1,
    num_workers=1,
    eval_val_only=False,
    eval_def_only=False,
    **kwargs,
):
    model = CIFAR10_Model(model_cfg, weight_path=weight_path).cuda()
    criterion = CrossEntropyLoss()
    run_epoch = partial(run_general_epoch, model=model, criterion=criterion, train=False)

    to_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=num_workers)

    output = []

    def run_eval(dataset, split, defense=False):
        elapsed_time, log = run_epoch(
            to_dataloader(dataset), name=f"{'W' if defense else 'WO'} def {split}"
        )
        print(
            f"{'Eval ' + split:20s} ({'w/ ' if defense else 'w/o'} defense) [{elapsed_time:6.2f}s]"
            + " - ".join(f"{k}: {log[k]:.4f}" for k in ["loss", "benign_acc", "adv_acc"])
        )
        output.append({"split": split, "defense": defense, "time": elapsed_time, "log": log})

    for defense in [True] if eval_def_only else [False, True]:
        train_dataset, validation_dataset = build_dataset(dataset_cfg, defense=defense)
        adv_train_dataset, adv_validation_dataset = build_adv_dataset(
            adv_images_dir, dataset_cfg, defense=defense
        )
        if not eval_val_only:
            run_eval(JointDataset(train_dataset, adv_train_dataset), "train", defense)
        run_eval(JointDataset(validation_dataset, adv_validation_dataset), "validation", defense)

    if not output_dir.exists():
        output_dir.mkdir()
    with open(output_dir / "output.json", "w") as f:
        json.dump(output, f, indent=2)
