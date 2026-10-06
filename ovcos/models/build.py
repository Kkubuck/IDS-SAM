"""Model construction and checkpoint loading."""

from __future__ import annotations

import torch

from ovcos.io import load_weights
from .idsam import IDSAM


def build_localizer(config, checkpoint=None, sam_checkpoint=None, device="cpu"):
    if checkpoint is not None and sam_checkpoint is not None:
        raise ValueError("Specify either trained IDS-SAM weights or SAM initialization, not both.")
    model = IDSAM(**config["model"])
    if checkpoint is not None:
        model.load_state_dict(load_weights(checkpoint), strict=True)
    if sam_checkpoint is not None:
        base = load_weights(sam_checkpoint)
        state = model.state_dict()
        selected = {k: v for k, v in base.items() if k in state and v.shape == state[k].shape}
        encoder_keys = {
            k for k in state if k.startswith("image_encoder.") and "prompt_generator" not in k
        }
        missing = encoder_keys - selected.keys()
        if missing:
            raise ValueError(
                f"SAM initialization lacks {len(missing)} encoder tensors: {sorted(missing)[:3]}"
            )
        model.load_state_dict(selected, strict=False)
    for name, parameter in model.named_parameters():
        if name.startswith("image_encoder.") and "prompt_generator" not in name:
            parameter.requires_grad_(False)
    model.to(device)
    model.device = torch.device(device)
    return model
