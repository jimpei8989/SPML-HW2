import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from modules.datasets import AdversarialDataset


def pgd_attack(model, dataloader, name="", epsilon=0.03125, num_iters=1):
    adv_X, adv_Y = [], []
    criterion = CrossEntropyLoss()

    model.eval()
    for x, y in tqdm(
        dataloader,
        desc=f"⚔ [{name[:10].center(10):10s}]",
        ncols=120,
        leave=False,
    ):
        x, y = x.cuda(), y.cuda()
        lower_bound = torch.max(torch.zeros_like(x), x - epsilon)
        upper_bound = torch.min(torch.ones_like(x), x + epsilon)

        for _ in range(num_iters):
            x.requires_grad = True

            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()

            grad = x.grad.detach().sign()
            x.requires_grad = False

            x += epsilon * grad

            x = torch.max(torch.min(x, upper_bound), lower_bound)

        adv_X.append(x)
        adv_Y.append(y)

    return AdversarialDataset(torch.cat(adv_X), torch.cat(adv_Y))
