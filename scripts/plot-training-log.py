import json

from matplotlib import pyplot as plt


def plot_training_log():
    epochs = []
    train_benign_acc = []
    train_adv_acc = []
    validation_benign_acc = []
    validation_adv_acc = []

    with open("logs/final-256/training_log.json") as f:
        train_log = json.load(f)

        for sth in train_log:
            epochs.append(sth["epoch"])
            train_benign_acc.append(sth["train_log"]["benign_acc"])
            train_adv_acc.append(sth["train_log"]["adv_acc"])
            validation_benign_acc.append(sth["validation_log"]["benign_acc"])
            validation_adv_acc.append(sth["validation_log"]["adv_acc"])

    attack_epochs = []
    attack_target = []
    attack_train_acc = []
    attack_validation_acc = []

    with open("logs/final-256/terminal_output.txt") as f:
        for line in f.readlines()[::14][:-1]:
            attack_target.append(line.strip().split(" ")[-1].replace(")", ""))

    with open("logs/final-256/attacking_log.json") as f:
        attack_log = json.load(f)

        for sth in attack_log:
            attack_epochs.append(sth["epoch"])
            attack_train_acc.append(sth["attack_train_log"]["adv_acc"])
            attack_validation_acc.append(sth["attack_validation_log"]["adv_acc"])

    attack_acc = [(a + b) / 2 for a, b in zip(attack_train_acc, attack_validation_acc)]

    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)

    ax.plot(epochs, train_benign_acc, label="train_benign_acc", color="blue", linestyle=":")
    ax.plot(epochs, train_adv_acc, label="train_adv_acc", color="blue", linestyle="-")
    ax.plot(epochs, validation_benign_acc, label="val_benign_acc", color="green", linestyle=":")
    ax.plot(epochs, validation_adv_acc, label="val_adv_acc", color="green", linestyle="-")

    ax.plot(attack_epochs, attack_acc, label="attack_acc", color="red", linestyle="--")

    for attack_name, m in zip(sorted(list(set(attack_target))), ["o", "^", "s", "*", "h"]):
        its_epochs, its_acc = zip(
            *(
                (e, a)
                for e, a, t in zip(attack_epochs, attack_acc, attack_target)
                if t == attack_name
            )
        )
        ax.scatter(
            its_epochs, its_acc, s=24, color="black", alpha=0.5, marker=m, label=attack_name
        )

    ax.set_title("Training Log")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()

    fig.savefig("outputs/training-log.png", dpi=300)


plot_training_log()
