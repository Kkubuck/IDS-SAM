"""Numerical helpers for recognition."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize embeddings along the selected axis."""
    denom = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / denom


def softmax(x: np.ndarray, axis: int = -1, temp: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax."""
    z = x / max(temp, 1e-6)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def topk_accuracy(pred_ranked: np.ndarray, labels: np.ndarray, k: int) -> float:
    return float((pred_ranked[:, :k] == labels[:, None]).any(axis=1).mean())


def metrics_from_ranked(pred_ranked: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute cumulative Top-1 through Top-5 accuracy."""
    return {f"top{k}": topk_accuracy(pred_ranked, labels, k) for k in (1, 2, 3, 4, 5)}


def rank_classes_from_scores(
    scores: np.ndarray, topk_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank each sample's candidate classes by score."""
    ranked_pos = np.argsort(-scores, axis=1)
    ranked_classes = np.take_along_axis(topk_indices, ranked_pos, axis=1)
    return ranked_pos, ranked_classes
