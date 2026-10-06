"""Adaptive stage2 subclass strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .array_ops import l2norm


@dataclass
class Stage2AdaptiveResult:
    text_embeddings: np.ndarray
    scores: np.ndarray
    used_subclass_idx: np.ndarray
    alpha: np.ndarray
    ambiguity: np.ndarray
    commonality: np.ndarray
    owner_reliability: np.ndarray
    subclass_specificity: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _robust_center_scale(x: np.ndarray) -> Tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    med = float(np.median(x))
    q1 = float(np.quantile(x, 0.25))
    q3 = float(np.quantile(x, 0.75))
    sc = max(q3 - q1, 1e-6)
    return med, sc


def _ambiguity_from_scores(base_scores: np.ndarray) -> np.ndarray:
    """Estimate sample ambiguity from the standardized Top-1/Top-2 margin."""
    s = np.sort(base_scores.astype(np.float32), axis=1)[:, ::-1]
    sd = base_scores.std(axis=1).astype(np.float32) + 1e-6
    z12 = (s[:, 0] - s[:, 1]) / sd
    q1 = float(np.quantile(z12, 0.25))
    q3 = float(np.quantile(z12, 0.75))
    den = max(q3 - q1, 1e-6)
    amb = np.clip((q3 - z12) / den, 0.0, 1.0)
    return amb.astype(np.float32)


def _commonality_from_text(topk_text_embeddings: np.ndarray) -> np.ndarray:
    """Measure candidate-text commonality using mean pairwise similarity."""
    n, k, _ = topk_text_embeddings.shape
    sim = np.einsum("nkd,njd->nkj", topk_text_embeddings, topk_text_embeddings, optimize=True)
    tri = np.triu_indices(k, 1)
    c = sim[:, tri[0], tri[1]].mean(axis=1).astype(np.float32)
    q1 = float(np.quantile(c, 0.25))
    q3 = float(np.quantile(c, 0.75))
    den = max(q3 - q1, 1e-6)
    return np.clip((c - q1) / den, 0.0, 1.0).astype(np.float32)


def compute_subclass_specificity(
    class_embeddings: np.ndarray,
    subclass_embeddings: np.ndarray,
    subclass_owner_indices: np.ndarray,
) -> np.ndarray:
    """Subtract the strongest non-owner similarity from the owner's similarity."""
    cls = l2norm(class_embeddings.astype(np.float32), axis=1)
    sub = l2norm(subclass_embeddings.astype(np.float32), axis=1)
    owner = subclass_owner_indices.astype(np.int64)
    sim = sub @ cls.T
    row = np.arange(sub.shape[0], dtype=np.int64)
    owner_sim = sim[row, owner]
    sim_wo = sim.copy()
    sim_wo[row, owner] = -1e9
    non_owner_max = sim_wo.max(axis=1)
    return (owner_sim - non_owner_max).astype(np.float32)


def _build_owner_reliability(
    subclass_specificity: np.ndarray,
    subclass_owner_indices: np.ndarray,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute owner-class reliability and subclass weights."""
    spec = subclass_specificity.astype(np.float32)
    med, sc = _robust_center_scale(spec)
    sub_w = _sigmoid((spec - med) / sc).astype(np.float32)

    owner_rel = np.zeros((n_classes,), dtype=np.float32)
    for c in range(n_classes):
        idx = np.where(subclass_owner_indices == c)[0]
        if idx.size == 0:
            owner_rel[c] = 0.0
        else:
            owner_rel[c] = float(sub_w[idx].mean())
    q1 = float(np.quantile(owner_rel, 0.25))
    q3 = float(np.quantile(owner_rel, 0.75))
    den = max(q3 - q1, 1e-6)
    owner_rel = np.clip((owner_rel - q1) / den, 0.0, 1.0).astype(np.float32)
    return owner_rel, sub_w


def build_stage2_subclass_adaptive(
    image_embeddings: np.ndarray,
    topk_text_embeddings: np.ndarray,
    topk_indices: np.ndarray,
    subclass_embeddings: np.ndarray,
    subclass_owner_indices: np.ndarray,
    class_embeddings: np.ndarray,
    mode: str = "hard_positive",
    mix_text: bool = False,
    use_ambiguity: bool = True,
    use_commonality: bool = True,
) -> Stage2AdaptiveResult:
    """Adapt subclass alignment strength for each sample and candidate class."""
    image_embeddings = l2norm(image_embeddings.astype(np.float32), axis=1)
    topk_text_embeddings = l2norm(topk_text_embeddings.astype(np.float32), axis=2)
    subclass_embeddings = l2norm(subclass_embeddings.astype(np.float32), axis=1)
    class_embeddings = l2norm(class_embeddings.astype(np.float32), axis=1)
    subclass_owner_indices = subclass_owner_indices.astype(np.int64)
    topk_indices = topk_indices.astype(np.int64)

    stage2_text = topk_text_embeddings.copy()
    stage2_scores = np.einsum("nd,nkd->nk", image_embeddings, stage2_text, optimize=True).astype(
        np.float32
    )
    used_subclass_idx = np.full(stage2_scores.shape, -1, dtype=np.int64)
    alpha = np.zeros(stage2_scores.shape, dtype=np.float32)

    n_classes = class_embeddings.shape[0]
    sub_spec = compute_subclass_specificity(
        class_embeddings, subclass_embeddings, subclass_owner_indices
    )
    owner_rel, sub_w = _build_owner_reliability(sub_spec, subclass_owner_indices, n_classes)

    ambiguity = (
        _ambiguity_from_scores(stage2_scores)
        if use_ambiguity
        else np.ones((stage2_scores.shape[0],), dtype=np.float32)
    )
    commonality = (
        _commonality_from_text(topk_text_embeddings)
        if use_commonality
        else np.ones((stage2_scores.shape[0],), dtype=np.float32)
    )
    sample_rel = (ambiguity * (0.5 + 0.5 * commonality)).astype(np.float32)

    owner_to_sub: Dict[int, np.ndarray] = {}
    for cls_idx in np.unique(subclass_owner_indices):
        owner_to_sub[int(cls_idx)] = np.where(subclass_owner_indices == int(cls_idx))[0]

    n, k, _ = stage2_text.shape
    for i in range(n):
        vi = image_embeddings[i]
        sr = float(sample_rel[i])
        for j in range(k):
            owner_cls = int(topk_indices[i, j])
            sub_idx = owner_to_sub.get(owner_cls)
            if sub_idx is None or sub_idx.size == 0:
                continue
            subs = subclass_embeddings[sub_idx]
            sub_scores = subs @ vi
            best_local = int(np.argmax(sub_scores))
            best_sub = int(sub_idx[best_local])
            best_score = float(sub_scores[best_local])
            base_score = float(stage2_scores[i, j])
            if best_score <= base_score:
                continue

            if mode == "hard_positive":
                a = 1.0 if float(sub_spec[best_sub]) > 0.0 else 0.0
            elif mode == "soft_spec":
                a = float(sub_w[best_sub])
            elif mode == "soft_spec_owner":
                a = float(sub_w[best_sub] * owner_rel[owner_cls])
            else:
                a = float(sub_w[best_sub] * owner_rel[owner_cls] * sr)

            if a <= 0.0:
                continue

            uplift = best_score - base_score
            stage2_scores[i, j] = base_score + a * uplift
            alpha[i, j] = a
            used_subclass_idx[i, j] = best_sub

            if mix_text:
                t_owner = stage2_text[i, j]
                t_sub = subclass_embeddings[best_sub]
                stage2_text[i, j] = l2norm(((1.0 - a) * t_owner + a * t_sub)[None, :], axis=1)[0]

    stage2_text = l2norm(stage2_text, axis=2)
    return Stage2AdaptiveResult(
        text_embeddings=stage2_text.astype(np.float32),
        scores=stage2_scores.astype(np.float32),
        used_subclass_idx=used_subclass_idx.astype(np.int64),
        alpha=alpha.astype(np.float32),
        ambiguity=ambiguity.astype(np.float32),
        commonality=commonality.astype(np.float32),
        owner_reliability=owner_rel.astype(np.float32),
        subclass_specificity=sub_spec.astype(np.float32),
    )
