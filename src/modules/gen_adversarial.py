from functools import partial

from torch.utils.data import DataLoader

from .models import CIFAR10_Model
from .datasets import build_dataset
from .attacks import Attacker
from .utils import timer


@timer
def generate_adversarial_examples(
    model_cfg=None,
    attack_cfg=None,
    dataset_cfg=None,
    adv_images_dir=None,
    batch_size=1,
    num_workers=1,
    **kwargs,
):
    model = CIFAR10_Model(model_cfg).cuda()
    to_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=num_workers)

    attacker = Attacker(attack_cfg)
    num_iters = attacker.request_num_iters()

    (_, adv_train_dataset), (_, adv_validation_dataset) = map(
        lambda p: attacker.attack(model, to_dataloader(p[0]), num_iters=num_iters, name=p[1]),
        zip(build_dataset(dataset_cfg, defense=False), ("attack_train", "attack_validation")),
    )

    adv_train_dataset.save_to_directory(adv_images_dir / "train")
    adv_validation_dataset.save_to_directory(adv_images_dir / "validation")
