from functools import partial
from typing import Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.attacks import attack
from modules.datasets import JointDataset
from modules.models import build_optimizer
from modules.recorder import Recorder
from modules.utils import timer


def train(
    model,
    train_dataset,
    validation_dataset,
    attack_cfg=None,
    optimizer_cfg=None,
    num_epochs=1,
    batch_size=1,
    num_workers=1,
    checkpoint_period=5,
    adversarial_examples_resample_period=5,
    recorder: Optional[Recorder] = None,
):
    to_train_dataloader = partial(
        DataLoader, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    to_validation_dataloader = partial(
        DataLoader, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )

    benign_train_dataloader = to_train_dataloader(train_dataset)
    benign_validation_dataloader = to_validation_dataloader(validation_dataset)

    criterion = CrossEntropyLoss()

    if train:
        optimizer, (scheduler, step_type) = build_optimizer(
            optimizer_cfg, model.parameters()
        )

    @timer
    def run_epoch(dataloader, train=True, name="no-name"):
        losses, benign_accuracies, adv_accuracies = [], [], []
        if train:
            model.train()
        else:
            model.eval()

        with torch.enable_grad() if train else torch.no_grad():
            for x, y, is_adv in tqdm(
                dataloader,
                desc=f"{'☼' if train else '☃︎'} [{name[:10].center(10):10s}]",
                ncols=120,
                leave=False,
            ):
                # 1. clear the gradients of the optimizers
                if train:
                    optimizer.zero_grad()

                # 2. Run forward and calculate the loss
                y_hat = model(x.cuda())
                loss = criterion(y_hat, y.cuda())

                # 3. backward prop the loss and step the optimizer
                if train:
                    loss.backward()
                    optimizer.step()

                # 4. calculate the accuracy
                losses.append(loss.detach().cpu().item())
                top1_acc = torch.eq(y_hat.cpu().argmax(dim=1), y)

                benign_accuracies.append(torch.masked_select(top1_acc, is_adv.eq(0)))
                adv_accuracies.append(torch.masked_select(top1_acc, is_adv.eq(1)))

                if train and scheduler and step_type == "batch":
                    scheduler.step()

        if train and scheduler and step_type == "epoch":
            scheduler.step()

        benign_accuracies = torch.cat(benign_accuracies).tolist()
        adv_accuracies = torch.cat(adv_accuracies).tolist()

        return {
            "loss": np.mean(losses),
            "benign_acc": np.mean(benign_accuracies),
            "adv_acc": np.mean(adv_accuracies) if adv_accuracies else 0,
        }

    for epoch in range(0, 1 + num_epochs):
        print(f"Epoch: {epoch:3d} / {num_epochs}")

        attack_train_time, attack_validation_time = 0, 0

        # 1. Generate adversarial datasets for training and validation and mix the benign and
        # adversarial examples
        if epoch == 0 or (
            epoch != 1 and (epoch - 1) % adversarial_examples_resample_period == 0
        ):
            attack_train_time, adv_train_dataset = attack(
                model,
                benign_train_dataloader,
                name="pgd_train",
                cfg=attack_cfg,
            )
            attack_validation_time, adv_validation_dataset = attack(
                model,
                benign_validation_dataloader,
                name="pgd_validation",
                cfg=attack_cfg,
            )

            joint_train_dataset = JointDataset(train_dataset, adv_train_dataset)
            joint_validation_dataset = JointDataset(
                validation_dataset, adv_validation_dataset
            )

        # 2.1 Run train on the joint dataset
        train_time, train_log = run_epoch(
            to_train_dataloader(joint_train_dataset),
            name="joint_train",
            train=epoch > 0,
        )

        # 2.2 Run validation on benign dataset and validation dataset
        validation_time, validation_log = run_epoch(
            to_validation_dataloader(joint_validation_dataset),
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
            + " - ".join(
                f"{k}: {train_log[k]:.4f}" for k in ["loss", "benign_acc", "adv_acc"]
            )
        )
        print(
            f"Validation  [{attack_validation_time:6.2f}s | {validation_time:6.2f}s] ~ "
            + " - ".join(
                f"{k}: {validation_log[k]:.4f}"
                for k in ["loss", "benign_acc", "adv_acc"]
            )
        )

        if epoch > 0 and epoch % checkpoint_period == 0:
            recorder.save_checkpoint(epoch, model, optimizer)

    recorder.finish_training(model)
