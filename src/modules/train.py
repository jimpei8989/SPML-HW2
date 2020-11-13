from functools import partial
from typing import Optional

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from modules.attacks import attack
from modules.run_epoch import run_general_epoch
from modules.datasets import JointDataset, build_dataset
from modules.models import build_optimizer
from modules.recorder import Recorder
from modules.utils import timer


@timer
def train(
    model,
    dataset_cfg=None,
    attack_cfg=None,
    optimizer_cfg=None,
    recorder: Optional[Recorder] = None,
    num_epochs=1,
    batch_size=1,
    num_workers=1,
    checkpoint_period=10,
    adversarial_examples_resample_period=10,
    **kwargs,
):
    to_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=num_workers)

    train_dataset, validation_dataset = build_dataset(dataset_cfg, defense=False)
    train_dataloader = to_dataloader(train_dataset)
    validation_dataloader = to_dataloader(validation_dataset)

    criterion = CrossEntropyLoss()

    optimizer, (scheduler, scheduler_type) = build_optimizer(optimizer_cfg, model.parameters())

    run_epoch = partial(
        run_general_epoch,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_type=scheduler_type,
    )

    for epoch in range(0, 1 + num_epochs):
        print(f"Epoch: {epoch:3d} / {num_epochs}")

        attack_train_time, attack_validation_time = 0, 0

        # 1. Generate adversarial datasets for training and validation and mix the benign and
        # adversarial examples
        if epoch == 0 or (epoch != 1 and (epoch - 1) % adversarial_examples_resample_period == 0):
            attack_train_time, adv_train_dataset = attack(
                model,
                train_dataloader,
                name="pgd_train",
                cfg=attack_cfg,
            )
            attack_validation_time, adv_validation_dataset = attack(
                model,
                validation_dataloader,
                name="pgd_validation",
                cfg=attack_cfg,
            )

            adv_train_dataset.save_to_directory(
                recorder.root_dir / "adv_images" / f"adv_train_{epoch}"
            )
            adv_validation_dataset.save_to_directory(
                recorder.root_dir / "adv_images" / f"adv_validation_{epoch}"
            )

            joint_train_dataloader = to_dataloader(
                JointDataset(train_dataset, adv_train_dataset), shuffle=True
            )
            joint_validation_dataloader = to_dataloader(
                JointDataset(validation_dataset, adv_validation_dataset)
            )

        # 2.1 Run train on the joint dataset
        train_time, train_log = run_epoch(
            dataloader=joint_train_dataloader,
            train=epoch > 0,
            name="joint_train",
        )

        # 2.2 Run validation on benign dataset and validation dataset
        validation_time, validation_log = run_epoch(
            dataloader=joint_validation_dataloader,
            train=False,
            name="joint_validation",
        )

        recorder.on_epoch_ends(
            train_time=train_time,
            train_log=train_log,
            validation_time=validation_time,
            validation_log=validation_log,
        )

        print(
            f"Train       [{attack_train_time:6.2f}s | {train_time:6.2f}s] ~ "
            + " - ".join(f"{k}: {train_log[k]:.4f}" for k in ["loss", "benign_acc", "adv_acc"])
        )
        print(
            f"Validation  [{attack_validation_time:6.2f}s | {validation_time:6.2f}s] ~ "
            + " - ".join(
                f"{k}: {validation_log[k]:.4f}" for k in ["loss", "benign_acc", "adv_acc"]
            )
        )

        if epoch > 0 and epoch % checkpoint_period == 0:
            recorder.save_checkpoint(epoch, model, optimizer, scheduler, scheduler_type)

    recorder.finish_training(model)
