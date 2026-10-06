"""Dataset manifests. Inference reads images and IDs without requiring labels."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ovcos.io import read_json, write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_manifest(path, require_masks=False):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    seen = set()
    for row in rows:
        identifier = str(row["id"])
        if (
            not identifier
            or identifier in seen
            or identifier in {".", ".."}
            or "/" in identifier
            or "\\" in identifier
            or Path(identifier).name != identifier
        ):
            raise ValueError(f"Duplicate or unsafe sample ID: {identifier!r}")
        seen.add(identifier)
        row["id"] = identifier
        for field in ("image", "mask", "prediction"):
            if field in row:
                p = Path(row[field]).expanduser()
                row[field] = str(p if p.is_absolute() else path.parent / p)
                if not Path(row[field]).is_file():
                    raise FileNotFoundError(row[field])
        if "image" not in row or (require_masks and "mask" not in row):
            raise ValueError(f"Missing required image/mask for {identifier}")
    return rows


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_rows(directory):
    paths = sorted(p for p in Path(directory).glob("**/*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No images found in {directory}")
    ids = [p.stem for p in paths]
    if len(ids) != len(set(ids)):
        raise ValueError("Image stems must be unique across the input directory.")
    return [{"id": p.stem, "image": str(p.resolve())} for p in paths]


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def prepare_ovcamo(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    classes = read_json(root / "dataset/class_info.json")
    samples = read_json(root / "dataset/sample_info.json")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        subset = [c for c in classes if c["split"] == split]
        names = sorted(c["name"] for c in subset)
        rows = []
        for sample in samples:
            if sample["base_class"] not in names:
                continue
            identifier = str(sample["unique_id"])
            folder = root / "ovcamo_dataset" / split
            candidates = [folder / "image" / (identifier + ext) for ext in sorted(IMAGE_SUFFIXES)]
            image = next((p for p in candidates if p.is_file()), None)
            mask = folder / "mask" / (identifier + ".png")
            if image is None or not mask.is_file():
                raise FileNotFoundError(f"Image or mask missing for {identifier} under {folder}")
            rows.append(
                {
                    "id": identifier,
                    "image": str(image),
                    "mask": str(mask),
                    "class_name": sample["base_class"],
                    "label": names.index(sample["base_class"]),
                }
            )
        write_manifest(output / f"{split}.jsonl", rows)
        vocabulary = []
        for name in names:
            item = next(c for c in subset if c["name"] == name)
            descriptors = item.get("sub_class", [])
            if isinstance(descriptors, dict):
                descriptors = list(descriptors)
            if isinstance(descriptors, str):
                descriptors = [descriptors]
            vocabulary.append({"name": name, "descriptors": [str(x) for x in descriptors]})
        write_json(output / f"{split}_vocabulary.json", vocabulary)
    return output
