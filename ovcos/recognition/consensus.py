"""Neighbor-consensus support signal used by CADER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .array_ops import l2norm, rank_classes_from_scores, softmax


@dataclass
class NCSRResult:
    support_scores: np.ndarray
    support_z: np.ndarray
    lambda_auto: float
    ambiguity: np.ndarray
    flip_gate: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _standardize_per_row(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def _ambiguity_from_scores(
    base_scores: np.ndarray, policy_temp: float = 1.0, mode: str = "margin_std"
) -> np.ndarray:
    """Estimate per-sample ambiguity in the unit interval."""
    if mode == "softmax":
        p = softmax(base_scores, axis=1, temp=policy_temp)
        order = np.argsort(-p, axis=1)
        p1 = np.take_along_axis(p, order[:, :1], axis=1).squeeze(1)
        p2 = np.take_along_axis(p, order[:, 1:2], axis=1).squeeze(1)
        margin = p1 - p2
        ent = -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(p.shape[1])
        return (0.5 * (1.0 - margin) + 0.5 * ent).astype(np.float32)

    s = np.sort(base_scores, axis=1)[:, ::-1].astype(np.float32)
    sd = base_scores.std(axis=1).astype(np.float32) + 1e-6
    z12 = (s[:, 0] - s[:, 1]) / sd
    z1k = (s[:, 0] - s[:, -1]) / sd
    conf12 = _sigmoid((z12 - 0.95) / 0.45)
    conf1k = _sigmoid((z1k - 2.05) / 0.70)
    conf = (0.65 * conf12 + 0.35 * conf1k).astype(np.float32)
    return np.clip(1.0 - conf, 0.0, 1.0).astype(np.float32)


def build_ncsr_scores(
    image_embeddings: np.ndarray,
    topk_indices: np.ndarray,
    base_scores: np.ndarray,
    policy_temp: float = 1.0,
    knn: Optional[int] = 8,
    knn_offset: int = 1,
    lambda_scale: float = 0.25,
    ambiguity_power: float = 0.0,
    use_ambiguity: bool = False,
    lambda_override: Optional[float] = None,
    consensus_mode: str = "hard",
    ambiguity_mode: str = "margin_std",
) -> NCSRResult:
    """Build standardized candidate support from nearest-neighbor consensus."""
    image_embeddings = l2norm(image_embeddings.astype(np.float32), axis=1)
    topk_indices = topk_indices.astype(np.int64)
    base_scores = base_scores.astype(np.float32)

    n, k = base_scores.shape
    if knn is None:
        knn = int(k + max(0, knn_offset))
    knn = int(min(max(1, knn), max(1, n - 1)))

    _, ranked = rank_classes_from_scores(base_scores, topk_indices)
    pred_top1 = ranked[:, 0]
    p_local = softmax(base_scores, axis=1, temp=policy_temp)

    sim = image_embeddings @ image_embeddings.T
    np.fill_diagonal(sim, -np.inf)
    nbr = np.argpartition(-sim, knn, axis=1)[:, :knn]
    nbr_sim = np.take_along_axis(sim, nbr, axis=1)
    nbr_w = np.maximum(nbr_sim, 0.0).astype(np.float32)
    nbr_w = nbr_w / (nbr_w.sum(axis=1, keepdims=True) + 1e-6)

    support = np.zeros_like(base_scores, dtype=np.float32)
    if consensus_mode == "hard":
        neigh_cls = pred_top1[nbr]
        for j in range(k):
            cand = topk_indices[:, j][:, None]
            match = (neigh_cls == cand).astype(np.float32)
            support[:, j] = (match * nbr_w).sum(axis=1)
    elif consensus_mode == "soft":
        neigh_cls_topk = topk_indices[nbr]
        neigh_prob_topk = p_local[nbr]
        for j in range(k):
            cand = topk_indices[:, j][:, None, None]
            match = (neigh_cls_topk == cand).astype(np.float32)
            mass = (match * neigh_prob_topk).sum(axis=2)
            support[:, j] = (mass * nbr_w).sum(axis=1)
    else:
        raise ValueError(f"Unsupported consensus_mode: {consensus_mode}")

    support_z = _standardize_per_row(support)

    if lambda_override is None:
        lambda_auto = float(lambda_scale * np.median(base_scores.std(axis=1)))
    else:
        lambda_auto = float(lambda_override)

    ambiguity = _ambiguity_from_scores(base_scores, policy_temp=policy_temp, mode=ambiguity_mode)
    if use_ambiguity:
        amb = np.power(np.clip(ambiguity, 0.0, 1.0), float(ambiguity_power)).astype(np.float32)
    else:
        amb = np.ones((n,), dtype=np.float32)

    final = (base_scores + lambda_auto * amb[:, None] * support_z).astype(np.float32)
    _ = final  # compatibility
    return NCSRResult(
        support_scores=support.astype(np.float32),
        support_z=support_z.astype(np.float32),
        lambda_auto=lambda_auto,
        ambiguity=ambiguity.astype(np.float32),
        flip_gate=np.ones((n,), dtype=np.float32),
    )
