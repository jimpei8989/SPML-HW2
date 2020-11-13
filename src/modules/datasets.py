from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.transforms import functional as TF

from modules.transforms import build_transform


class AdversarialDataset(Dataset):
    labels = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    def __init__(self, X, Y):
        super().__init__()

        self.X = X
        self.Y = Y.tolist()

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]

    def save_to_directory(self, directory: Path):
        for label in self.labels:
            (directory / label).mkdir(parents=True, exist_ok=True)

        counter = [0] * 10
        for x, y in zip(self.X, self.Y):
            TF.to_pil_image(x).save(directory / self.labels[y] / f"{counter[y]}.png")
            counter[y] += 1


class JointDataset(Dataset):
    def __init__(self, dataset_a, dataset_b):
        super().__init__()
        self.dataset_a = dataset_a if dataset_a else []
        self.dataset_b = dataset_b if dataset_b else []

    def __getitem__(self, index):
        return (
            self.dataset_a[index] + (0,)
            if index < len(self.dataset_a)
            else self.dataset_b[index - len(self.dataset_a)] + (1,)
        )

    def __len__(self):
        return len(self.dataset_a) + len(self.dataset_b)


def build_dataset(cfg, defense=False):
    transform = build_transform(cfg, defense=defense)

    return (
        CIFAR10(
            root=cfg.dataset_root,
            train=True,
            transform=transform,
        ),
        CIFAR10(
            root=cfg.dataset_root,
            train=False,
            transform=transform,
        ),
    )


def build_adv_dataset(dataset_root, cfg, defense=False):
    transform = build_transform(cfg, defense=defense)

    return (
        ImageFolder(
            root=dataset_root / "train",
            transform=transform,
        ),
        ImageFolder(
            root=dataset_root / "validation",
            transform=transform,
        ),
    )
