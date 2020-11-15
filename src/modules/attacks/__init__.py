import random
from functools import partial
from itertools import repeat, chain

from modules.utils import timer

from .pgd import pgd_attack

attack_func_mapping = {
    "pgd": pgd_attack,
}


class RandomChoiceIterable:
    def __init__(self, pool):
        self.pool = pool

    def __next__(self):
        return random.choice(self.pool)


class Attacker:
    def __init__(self, cfg):
        if cfg.iters.type == "fixed":
            self.num_iters_generator = repeat(cfg.iters.args)
        elif cfg.iters.type == "sequence":
            self.num_iters_generator = chain(iter(cfg.iters.args), repeat(cfg.iters.args[-1]))
        elif cfg.iters.type == "expand":
            self.num_iters_generator = chain(
                iter(sum(([r] * n for r, n in cfg.iters.args), start=[])),
                repeat(cfg.iters.args[-1][0]),
            )
        elif cfg.iters.type == "random":
            self.num_iters_generator = RandomChoiceIterable(cfg.iters.args)
        else:
            raise NotImplementedError

        self.attack_func = partial(attack_func_mapping[cfg.method], **cfg.args)

    def request_num_iters(self):
        return next(self.num_iters_generator)

    @timer
    def attack(self, model, dataloader, num_iters, **kwargs):
        return self.attack_func(model, dataloader, num_iters=num_iters, **kwargs)
