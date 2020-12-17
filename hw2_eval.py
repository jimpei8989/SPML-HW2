import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pytorchcv.model_provider import get_model

from src.modules.utils import timer

MODEL_WEIGHT_PATH = Path("model_weight/ensemble_weights.pt")
CIFAR10_CLASSES = [
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
]


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean, std = map(torch.as_tensor, [mean, std])
        self.mean = nn.Parameter(mean.view(3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(std.view(3, 1, 1), requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class EnsembleModel(nn.Module):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    model_names = ("nin", "resnet20", "sepreresnet56", "densenet40_k12_bc", "diaresnet110")

    def __init__(self, weight_path=None):
        super().__init__()

        for name in self.model_names:
            self.add_module(
                name,
                nn.Sequential(
                    Normalize(self.mean, self.std),
                    get_model(name + "_cifar10", pretrained=True),
                ),
            )

        if weight_path:
            ckpt = torch.load(str(weight_path))  # pytorch 1.5.1 does not accept Path-like
            if "model_state_dict" in ckpt:
                self.load_state_dict(ckpt["model_state_dict"])
            else:
                self.load_state_dict(ckpt)

    def forward(self, x):
        return torch.mean(torch.stack([m(x) for m in self.children()], dim=1), dim=1)


class TestDataset(Dataset):
    def __init__(self, images, transform=None):
        super().__init__()
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.images[index])
        else:
            return self.images[index]


class JpegCompression:
    def __init__(self, quality=60):
        self.quality = quality

    def __call__(self, img):
        with BytesIO() as f:
            img.save(f, format="JPEG", quality=self.quality)
            img = Image.open(f).copy()
        return img


def build_dataset(dataset_dir, transform=None):
    images, idx = [], 1
    while True:
        try:
            images.append(Image.open(dataset_dir / f"{idx}.png"))
            idx += 1
        except FileNotFoundError:
            break
    return TestDataset(images, transform)


def main():
    test_dir = Path(sys.argv[1] if len(sys.argv) >= 2 else Path("./example_folder"))

    dataset = build_dataset(test_dir, transform=T.Compose([JpegCompression(), T.ToTensor()]))
    # dataset = build_dataset(test_dir, transform=T.Compose([T.ToTensor()]))
    dataloader = DataLoader(dataset)

    # model = EnsembleModel(weight_path=MODEL_WEIGHT_PATH).cuda()
    model = EnsembleModel().cuda()
    model.eval()

    @timer
    def evaluate(model, name):
        outputs = []
        with torch.no_grad():
            for x in dataloader:
                y = model(x.cuda()).argmax(dim=1).cpu()
                outputs.extend(y.tolist())

        outputs = [CIFAR10_CLASSES[k] for k in outputs]

        with open("test/ans.txt") as f:
            ans = [line.strip() for line in f.readlines()]
            return [(i + 1, p, g) for i, (p, g) in enumerate(zip(outputs, ans)) if p != g]

    for name in model.model_names:
        print(f"------ {name} ------")
        t, misclassification = evaluate(getattr(model, name), name)
        print(
            "\n".join(map(lambda p: f"[{p[0]:03d}] - {p[1]:10s} - {p[2]:10s}", misclassification))
        )
        print(f"Accu: {100 - len(misclassification)}%")
        print(f">>> {t:.3f} secs elapsed")

    print("------ ENSEMBLE ------")
    t, misclassification = evaluate(model, "ensemble")
    print("\n".join(map(lambda p: f"[{p[0]:03d}] - {p[1]:10s} - {p[2]:10s}", misclassification)))
    print(f"Accu: {100 - len(misclassification)}%")
    print(f">>> {t:.3f} secs elapsed")


if __name__ == "__main__":
    main()
