import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from ovcos.io import read_config


@pytest.fixture(autouse=True)
def seeds():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    np.random.seed(12)


@pytest.fixture
def small_config():
    config = copy.deepcopy(read_config())
    config["model"]["inp_size"] = 64
    encoder = config["model"]["encoder_mode"]
    encoder.update(embed_dim=64, depth=2, num_heads=4, global_attn_indexes=[1], window_size=2)
    config["model"]["inst_prompt"]["slice_dim"] = 32
    config["model"]["structure_prior"]["phase"]["pc_size"] = 32
    config["training"].update(epochs=1, workers=0)
    return config


@pytest.fixture
def recognition_config():
    return read_config(Path(__file__).parents[1] / "ovcos/configs/recognition.yaml")


@pytest.fixture
def recognition_cache():
    from ovcos.recognition.array_ops import l2norm

    n, c, d = 12, 7, 32
    global_view = l2norm(np.random.randn(n, d).astype(np.float32), 1)
    local_view = l2norm(np.random.randn(n, d).astype(np.float32), 1)
    return dict(
        image_embeddings=l2norm(0.5 * (global_view + local_view), 1),
        global_image_embeddings=global_view,
        local_image_embeddings=local_view,
        class_embeddings=l2norm(np.random.randn(c, d).astype(np.float32), 1),
        class_names=np.asarray([f"class{i}" for i in range(c)]),
        ids=np.asarray([f"image{i}" for i in range(n)]),
        labels=np.arange(n) % c,
        subclass_embeddings=l2norm(np.random.randn(c * 2, d).astype(np.float32), 1),
        subclass_owner_indices=np.repeat(np.arange(c), 2),
    )
