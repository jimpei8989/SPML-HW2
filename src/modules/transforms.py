import random
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image
from torchvision import transforms as T


def jpeg_compress(img: Image.Image, quality) -> Image:
    with BytesIO() as f:
        img.save(f, format="JPEG", quality=quality)
        img = Image.open(f).copy()
    return img


class JpegCompression:
    def __init__(self, quality=60):
        self._quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        return jpeg_compress(img, self._quality)


class Shield:
    def __init__(self, qualities=[20, 40, 60, 80], block_size=8):
        self._qualities = qualities
        self._block_size = block_size

    def __call__(self, img: Image.Image) -> Image.Image:
        block_width, block_height = map(lambda k: k // self._block_size, img.size)
        ret_img = Image.new("RGB", img.size)

        # To fit the coordinate system with PIL, we make i, j to be the w, h of an image
        for l in range(0, img.size[0], block_width):
            for u in range(0, img.size[1], block_height):
                r, d = min(l + block_width, img.size[0]), min(u + block_height, img.size[1])
                ret_img.paste(
                    jpeg_compress(img.crop((l, u, r, d)), quality=random.choice(self._qualities)),
                    box=(l, u, r, d),
                )

        return ret_img


def get_transform(transform):
    name, args = transform.get("name"), transform.get("args", {})
    if name == "JPEG":
        return JpegCompression(**args)
    elif name == "Shield":
        return Shield(**args)
    elif hasattr(T, name):
        return getattr(T, name)(**args)
    else:
        raise NotImplementedError("The transform is not implemented yet")


def build_transform(cfg: Optional[Path], defense=False):
    if not defense:
        return T.Compose(list(map(get_transform, cfg.transforms)))
    else:
        return T.Compose(list(map(get_transform, cfg.defense_transforms)))
