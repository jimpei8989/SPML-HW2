from tqdm import tqdm

import numpy as np
import torch

from modules.utils import timer


@timer
def run_general_epoch(
    dataloader,
    model=None,
    criterion=None,
    optimizer=None,
    scheduler=None,
    scheduler_type=None,
    train=True,
    name="no-name",
):
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

            if train and scheduler and scheduler_type == "batch":
                scheduler.step()

    if train and scheduler and scheduler_type == "epoch":
        scheduler.step()

    benign_accuracies = torch.cat(benign_accuracies).tolist()
    adv_accuracies = torch.cat(adv_accuracies).tolist()

    return {
        "loss": np.mean(losses),
        "benign_acc": np.mean(benign_accuracies) if benign_accuracies else 0,
        "adv_acc": np.mean(adv_accuracies) if adv_accuracies else 0,
    }
