import random
from functools import partial
from itertools import cycle, repeat

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from modules.datasets import AdversarialDataset
from modules.utils import timer


class RandomChooseIterable:
    def __init__(self, pool):
        self.pool = pool

    def __next__(self):
        return random.choice(self.pool)


def pgd_attack(model, dataloader, name, num_iters=1, epsilon=0.03125, alpha=None, **kwargs):
    if alpha is None:
        alpha = epsilon

    adv_X, adv_Y = [], []
    criterion = CrossEntropyLoss()

    model.eval()
    for x, y in tqdm(
        dataloader,
        desc=f"⚔ [{name[:12].center(12):10s} * {num_iters}]",
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

            x += alpha * grad

            x = torch.max(torch.min(x, upper_bound), lower_bound)

        adv_X.append(x.cpu())
        adv_Y.append(y.cpu())

    return AdversarialDataset(torch.cat(adv_X), torch.cat(adv_Y))


attack_func_mapping = {
    "pgd": pgd_attack,
}


class Attacker:
    def __init__(self, cfg):
        method = cfg.pop("method")

        iter_type = cfg.pop("type")
        if iter_type == "fixed":
            self.num_iters_generator = repeat(cfg.args.num_iters)
        elif iter_type == "cycle":
            self.num_iters_generator = cycle(cfg.args.num_iters_list)
        elif iter_type == "expand":
            self.num_iters_generator = cycle(
                sum(([r] * n for r, n in cfg.args.num_iters_list), start=[])
            )
        elif iter_type == "random":
            self.num_iters_generator = RandomChooseIterable(cfg.args.num_iters_list)

        self.attack_func = partial(attack_func_mapping[method], **cfg)

    def request_num_iters(self):
        return next(self.num_iters_generator)

    @timer
    def attack(self, model, dataloader, num_iters, **kwargs):
        return self.attack_func(model, dataloader, num_iters=num_iters, **kwargs)
