"""Image preprocessing and strict instruction-cache lookup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ovcos.data.manifest import read_rgb
from ovcos.io import read_json


def image_tensor(image, size):
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    mean = pixels.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = pixels.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (pixels - mean) / std


class InstructionCache:
    def __init__(self, directory, split=None):
        directory = Path(directory)
        prefix = f"{split}_" if split else ""
        self.images = np.load(
            directory / f"{prefix}image_embeddings.npy", mmap_mode="r", allow_pickle=False
        )
        self.texts = np.load(directory / f"{prefix}text_embeddings.npy", allow_pickle=False)
        self.index = read_json(directory / f"{prefix}image_index.json")
        if self.images.ndim != 3 or self.texts.shape != self.images.shape[1:]:
            raise ValueError("Instruction cache must have shapes [N,V,D] and [V,D].")
        if sorted(self.index.values()) != list(range(len(self.images))):
            raise ValueError("Instruction index must be a one-to-one mapping onto cache rows.")
        if not np.isfinite(self.texts).all():
            raise ValueError("Non-finite instruction text embeddings.")

    def get(self, identifier):
        if identifier not in self.index:
            raise KeyError(
                f"Missing instruction embeddings for {identifier}. Re-extract this image."
            )
        value = np.array(self.images[self.index[identifier]], dtype=np.float32, copy=True)
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite image embeddings for {identifier}")
        return torch.from_numpy(value), torch.from_numpy(self.texts.astype(np.float32).copy())


class LocalizationDataset(Dataset):
    def __init__(self, rows, cache, size=1024, require_masks=True):
        self.rows, self.cache, self.size, self.require_masks = rows, cache, size, require_masks
        missing = [r["id"] for r in rows if r["id"] not in cache.index]
        if missing:
            raise KeyError(f"Instruction cache missing {len(missing)} images, first: {missing[0]}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = read_rgb(row["image"])
        visual, text = self.cache.get(row["id"])
        result = {
            "id": row["id"],
            "image": image_tensor(image, self.size),
            "instruction_images": visual,
            "instruction_texts": text,
            "height": image.height,
            "width": image.width,
        }
        if self.require_masks:
            with Image.open(row["mask"]) as mask:
                if mask.size != image.size:
                    raise ValueError(f"Image/mask sizes differ for {row['id']}")
                mask = mask.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
                result["mask"] = torch.from_numpy(
                    (np.asarray(mask).copy() > 127).astype(np.float32)
                )[None]
        return result
