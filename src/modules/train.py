from functools import partial
from typing import Optional

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from modules.attacks import Attacker
from modules.run_epoch import run_general_epoch
from modules.datasets import JointDataset, build_dataset
from modules.models import build_optimizer
from modules.recorder import Recorder
from modules.utils import timer


def get_attack_epochs(cfg, max_epoch):
    if cfg.type == "fixed":
        epochs = list(range(1, max_epoch + 1, cfg.args))
    elif cfg.type == "absolute":
        epochs = cfg.args
    elif cfg.type == "relative":
        epochs, cur_epoch = [], 1
        iter_seq = iter(cfg.args)

        while cur_epoch <= max_epoch:
            epochs.append(cur_epoch)
            try:
                cur_epoch += next(iter_seq)
            except StopIteration:
                cur_epoch += cfg.args[-1]
    elif cfg.type == "expand":
        epochs, cur_epoch = [], 1
        iter_seq = iter(sum(([r] * n for r, n in cfg.args), []))

        while cur_epoch <= max_epoch:
            epochs.append(cur_epoch)
            try:
                cur_epoch += next(iter_seq)
            except StopIteration:
                cur_epoch += cfg.args[-1][0]
    else:
        raise NotImplementedError()
    return set(epochs)


def print_verbose(desc, time, log, is_eval=False):
    print(
        f"{'⚔' if is_eval else '⚘'} {desc[:16].center(16):16s} [{time:6.2f}s] ~ "
        + " - ".join(f"{k}: {log[k]:.4f}" for k in ["loss", "benign_acc", "adv_acc"])
    )


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
    **kwargs,
):
    to_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=num_workers)

    train_dataset, validation_dataset = build_dataset(dataset_cfg, defense=False)
    train_dataloader = to_dataloader(train_dataset)
    validation_dataloader = to_dataloader(validation_dataset)

    criterion = CrossEntropyLoss()

    optimizer, (scheduler, scheduler_type) = build_optimizer(optimizer_cfg, model.parameters())

    attacker = Attacker(attack_cfg)
    attack_epochs = get_attack_epochs(attack_cfg.freq, num_epochs)

    run_epoch = partial(
        run_general_epoch,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_type=scheduler_type,
    )

    for epoch in range(1, 1 + num_epochs):
        attack_num_iters = attacker.request_num_iters() if epoch in attack_epochs else None
        print(
            f"Epoch: {epoch:3d} / {num_epochs}"
            + (f" (️⚔ {attack_num_iters})" if attack_num_iters else "")
        )

        # 1. Generate adversarial datasets for training and validation and mix the benign and
        # adversarial examples
        if attack_num_iters:
            attack_train_time, adv_train_dataset = attacker.attack(
                model,
                train_dataloader,
                num_iters=attack_num_iters,
                name="attack train",
            )
            attack_validation_time, adv_validation_dataset = attacker.attack(
                model,
                validation_dataloader,
                num_iters=attack_num_iters,
                name="attack validation",
            )

            # adv_train_dataset.save_to_directory(
            #     recorder.root_dir / "adv_images" / f"adv_train_{epoch}"
            # )
            # adv_validation_dataset.save_to_directory(
            #     recorder.root_dir / "adv_images" / f"adv_validation_{epoch}"
            # )

            joint_train_dataloader = to_dataloader(
                JointDataset(train_dataset, adv_train_dataset), shuffle=True
            )
            joint_validation_dataloader = to_dataloader(
                JointDataset(validation_dataset, adv_validation_dataset)
            )

            # Evaluate the adversarial examples
            _, eval_adv_train_log = run_epoch(
                dataloader=joint_train_dataloader,
                train=False,
                name="eval adv_train",
            )

            _, eval_adv_validation_log = run_epoch(
                dataloader=joint_validation_dataloader,
                train=False,
                name="eval adv_validation",
            )
            print_verbose("Adv. Train", attack_train_time, eval_adv_train_log, is_eval=True)
            print_verbose(
                "Adv. Validation", attack_validation_time, eval_adv_validation_log, is_eval=True
            )

            recorder.on_attack_ends(
                epoch=epoch,
                attack_train_time=attack_train_time,
                attack_validation_time=attack_validation_time,
                attack_train_log=eval_adv_train_log,
                attack_validation_log=eval_adv_validation_log,
            )

        # 2.1 Run train on the joint dataset
        train_time, train_log = run_epoch(
            dataloader=joint_train_dataloader,
            train=True,
            name="joint_train",
        )

        # 2.2 Run validation on benign dataset and validation dataset
        validation_time, validation_log = run_epoch(
            dataloader=joint_validation_dataloader,
            train=False,
            name="joint_validation",
        )

        print_verbose("Train", train_time, train_log)
        print_verbose("Validation", validation_time, validation_log)

        recorder.on_epoch_ends(
            epoch=epoch,
            train_time=train_time,
            train_log=train_log,
            validation_time=validation_time,
            validation_log=validation_log,
        )

        if epoch % checkpoint_period == 0:
            recorder.save_checkpoint(epoch, model, optimizer, scheduler, scheduler_type)

    recorder.finish_training(model)
