from functools import partial

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.attacks import pgd_attack
from modules.utils import timer


def train(
    model,
    train_dataset,
    validation_dataset,
    num_epochs=1,
    batch_size=1,
    num_workers=1,
    recorder=None,
):
    to_train_dataloader = partial(
        DataLoader, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    to_validation_dataloader = partial(
        DataLoader, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )

    train_dataloader = to_train_dataloader(train_dataset)
    validation_dataloader = to_validation_dataloader(validation_dataset)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    @timer
    def run_epoch(dataloader, train=True, name="default"):
        losses, accuracies = [], []
        for x, y in tqdm(
            dataloader,
            desc=f"{'☼' if train else '☃︎'} [{name[:10].center(10):10s}]",
            ncols=120,
            leave=False,
        ):
            y_hat = model(x.cuda())

            if train:
                optimizer.zero_grad()

            loss = criterion(y_hat, y.cuda())

            if train:
                loss.backward()
                optimizer.step()

            losses.append(loss.detach().cpu().item())
            accuracies.append(sum(torch.argmax(y_hat.cpu(), dim=1) == y))

        return {
            "loss": torch.mean(losses).item(),
            "accuracy": torch.mean(accuracies).item(),
        }

    model.train()

    for epoch in range(1, 1 + num_epochs):
        print(f"Epoch: {epoch:3d} / {num_epochs}")

        adv_train_dataloader = to_train_dataloader(
            pgd_attack(model, train_dataloader, name="pgd_train")
        )

        adv_validation_dataloader = to_validation_dataloader(
            pgd_attack(model, validation_dataloader, name="pgd_validation")
        )

        train_time, train_log = run_epoch(train_dataloader, name="train")
        validation_time, validation_log = run_epoch(
            validation_dataloader, train=False, name="validation"
        )

        adv_train_time, adv_train_log = run_epoch(
            adv_train_dataloader, name="adv_train"
        )
        adv_validation_time, adv_validation_log = run_epoch(
            adv_validation_dataloader, train=False, name="adv_validation"
        )

    return
