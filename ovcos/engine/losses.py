"""Segmentation and CoBRe objectives."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def localization_loss(outputs, target):
    logits = outputs["mask_logits"]
    probability = logits.sigmoid()
    intersection = (probability * target).sum((2, 3))
    union = (probability + target).sum((2, 3)) - intersection
    mask_loss = (
        F.binary_cross_entropy_with_logits(logits, target)
        + (1 - intersection / (union + 1e-6)).mean()
    )
    dilated = F.max_pool2d(target, 5, 1, 2)
    eroded = -F.max_pool2d(-target, 5, 1, 2)
    edge_target = (dilated - eroded > 0).to(target.dtype)
    edge = outputs["edge_prob"]
    if edge.min() < 0 or edge.max() > 1:
        edge = edge.sigmoid()
    edge = edge.clamp(0, 1).flatten(1)
    edge_target = edge_target.flatten(1)
    edge_loss = (
        1
        - (2 * (edge * edge_target).sum(1) + 1)
        / ((edge.square() + edge_target.square()).sum(1) + 1)
    ).mean()
    return mask_loss + edge_loss, {"mask": mask_loss.detach(), "edge": edge_loss.detach()}


def cobre_loss(model, fused, global_view, local_view, labels, texts, config):
    """Hardness-weighted CE, monotonic margin and counterfactual consistency."""
    fused, global_view, local_view = [
        F.normalize(x, dim=-1) for x in (fused, global_view, local_view)
    ]
    texts = F.normalize(texts, dim=-1)
    base_logits = fused @ texts.T
    rows = torch.arange(len(labels), device=labels.device)
    other = F.one_hot(labels, num_classes=texts.shape[0]).bool()
    base_margin = base_logits[rows, labels] - base_logits.masked_fill(other, -1e9).max(1).values
    corrected, aux = model(fused, global_view, local_view, base_margin[:, None], return_aux=True)
    logits = corrected @ texts.T
    new_margin = logits[rows, labels] - logits.masked_fill(other, -1e9).max(1).values
    ce = F.cross_entropy(logits, labels, reduction="none")
    floor = config["absolute_margin"] + config["margin_slope"] * (
        1 - aux["trust"].squeeze(1).detach()
    )
    monotonic = F.relu(torch.maximum(base_margin + config["margin"], floor) - new_margin)
    residual = global_view - local_view
    counterfactual = F.normalize(
        local_view
        + config["counterfactual_scale"]
        * residual[torch.randperm(len(labels), device=labels.device)],
        dim=-1,
    )
    cf_base = counterfactual @ texts.T
    cf_margin = cf_base[rows, labels] - cf_base.masked_fill(other, -1e9).max(1).values
    cf_corrected = model(counterfactual, counterfactual, local_view, cf_margin[:, None])
    cf_logits = cf_corrected @ texts.T
    temperature = config["crsc_temperature"]
    log_p, log_q = (
        F.log_softmax(logits / temperature, -1),
        F.log_softmax(cf_logits / temperature, -1),
    )
    crsc = (
        0.5
        * temperature**2
        * (
            F.kl_div(log_p, log_q.exp().detach(), reduction="none").sum(1)
            + F.kl_div(log_q, log_p.exp().detach(), reduction="none").sum(1)
        )
    )
    weights = aux["hard"].squeeze(1).detach()
    individual = (
        config["ce_weight"] * ce
        + config["monotonic_weight"] * monotonic
        + config["crsc_weight"] * crsc
    )
    loss = (weights * individual).mean() / (weights.mean() + 1e-6)
    return loss, {
        "ce": ce.mean().detach(),
        "monotonic": monotonic.mean().detach(),
        "crsc": crsc.mean().detach(),
    }
