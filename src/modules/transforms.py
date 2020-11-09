from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms as T


class JpegCompression:
    def __init__(self, quality=60):
        self._quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        with BytesIO() as f:
            img.save(f, format="JPEG", quality=self._quality)
            img = Image.open(f).convert("RGB")
        return img


def get_transform(transform):
    name, args = transform.get("name"), transform.get("args", {})
    if name == "JPEG":
        return JpegCompression(**args)
    elif hasattr(T, name):
        return getattr(T, name)(**args)


def build_transform(cfg: Optional[Path]):
    if not cfg:
        return None, None
    else:
        return (
            T.Compose(list(map(get_transform, cfg.train_transforms))),
            T.Compose(list(map(get_transform, cfg.inference_transforms))),
        )
