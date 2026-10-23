"""Portable file I/O and strict artifact validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def read_config(path=None):
    if path is None:
        path = Path(__file__).parent / "configs" / "idsam.yaml"
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_weights(path):
    """Load tensor state dictionaries without executing pickled Python objects."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError("A checkpoint must contain a state dictionary.")
    state = obj.get("state_dict", obj.get("model", obj))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint model/state_dict is not a dictionary.")
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_npz(path):
    """Caches use numeric arrays and Unicode strings, never object/pickle arrays."""
    with np.load(path, allow_pickle=False) as archive:
        try:
            result = {key: archive[key] for key in archive.files}
        except ValueError as error:
            raise ValueError(
                "Cache contains object arrays. Rebuild it with the cache commands."
            ) from error
    for name, value in result.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Non-finite values in cache field {name}.")
    return result


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise ValueError(f"Object arrays are not supported: {key}.")
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in cache field {key}.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
