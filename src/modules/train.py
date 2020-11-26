from functools import partial
from typing import Optional

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from .models import CIFAR10_Model, build_optimizer

from .attacks import Attacker
from .run_epoch import run_general_epoch
from .datasets import JointDataset, build_dataset
from .recorder import Recorder
from .utils import timer

from .train_utils import get_attack_epochs, print_verbose


@timer
def train(
    model_cfg=None,
    dataset_cfg=None,
    attack_cfg=None,
    optimizer_cfg=None,
    recorder: Optional[Recorder] = None,
    batch_size=1,
    num_workers=1,
    checkpoint_period=16,
    **kwargs,
):
    model = CIFAR10_Model(model_cfg).cuda()

    # Datasets
    to_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=num_workers)
    train_dataset, validation_dataset = build_dataset(dataset_cfg, defense=False)
    train_dataloader, validation_dataloader = map(
        to_dataloader, [train_dataset, validation_dataset]
    )

    # Prepare for run_epoch
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

    attacker = Attacker(attack_cfg)
    attack_epochs = get_attack_epochs(attack_cfg.freq, model.num_epochs)

    for epoch in range(1, 1 + model.num_epochs):
        attack_num_iters = attacker.request_num_iters() if epoch in attack_epochs else None
        attack_name = model.get_single_model() if attack_num_iters else None

        print(
            f"Epoch: {epoch:3d} / {model.num_epochs}"
            + (f" (️⚔ {attack_num_iters} on {attack_name})" if attack_num_iters else "")
        )

        # 1. Generate adversarial datasets for training and validation and mix the benign and
        # adversarial examples
        if attack_num_iters:
            attack_model = getattr(model, attack_name)
            attack_train_time, adv_train_dataset = attacker.attack(
                attack_model,
                train_dataloader,
                num_iters=attack_num_iters,
                name="attack train",
            )
            attack_validation_time, adv_validation_dataset = attacker.attack(
                attack_model,
                validation_dataloader,
                num_iters=attack_num_iters,
                name="attack validation",
            )

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
                target="attack_name",
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

    recorder.dump_logs()
    model.dump_weights(recorder.root_dir / "model_weights.pt")
