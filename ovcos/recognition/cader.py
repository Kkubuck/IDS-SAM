"""Consensus-aware differential evidence re-ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .array_ops import l2norm
from .consensus import build_ncsr_scores


@dataclass
class CADERConfig:
    """CADER re-ranking settings."""

    policy_temp: float = 1.0
    knn: int = 8
    support_scale: float = 0.32
    diff_scale: float = 0.42
    rho: float = 0.90
    eta: float = 0.34
    use_support: bool = True
    use_diff: bool = True
    use_commonality_gate: bool = False
    use_rarity_protect: bool = False
    use_microclass_lock: bool = True
    use_dual_agreement_gate: bool = False
    use_positive_evidence_gate: bool = True
    use_adaptive_topm: bool = False
    adaptive_topm_thr: float = 0.35
    consensus_mode: str = "hard"
    challenger_topm: int = 3


@dataclass
class CADERResult:
    """CADER output and diagnostic metadata."""

    final_scores: np.ndarray
    support_z: np.ndarray
    diff_scores: np.ndarray
    challenger_pos: np.ndarray
    challenger_score: np.ndarray
    ambiguity: np.ndarray
    commonality: np.ndarray
    gate: np.ndarray
    delta: np.ndarray
    lambda_auto: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _ambiguity_margin_std(base_scores: np.ndarray) -> np.ndarray:
    """Estimate ambiguity from standardized score margins."""
    s = np.sort(base_scores, axis=1)[:, ::-1].astype(np.float32)
    sd = base_scores.std(axis=1).astype(np.float32) + 1e-6
    z12 = (s[:, 0] - s[:, 1]) / sd
    z1k = (s[:, 0] - s[:, -1]) / sd
    conf12 = _sigmoid((z12 - 0.95) / 0.45)
    conf1k = _sigmoid((z1k - 2.00) / 0.70)
    conf = (0.65 * conf12 + 0.35 * conf1k).astype(np.float32)
    return np.clip(1.0 - conf, 0.0, 1.0).astype(np.float32)


def _commonality(topk_text_embeddings: np.ndarray) -> np.ndarray:
    """Measure commonality among the candidate text embeddings."""
    _, k, _ = topk_text_embeddings.shape
    sim = np.einsum("nkd,njd->nkj", topk_text_embeddings, topk_text_embeddings, optimize=True)
    tri = np.triu_indices(k, 1)
    c = sim[:, tri[0], tri[1]].mean(axis=1).astype(np.float32)
    q1 = float(np.quantile(c, 0.25))
    q3 = float(np.quantile(c, 0.75))
    den = max(q3 - q1, 1e-6)
    return np.clip((c - q1) / den, 0.0, 1.0).astype(np.float32)


def _build_stage2_prior(base_scores: np.ndarray, topk_indices: np.ndarray) -> np.ndarray:
    """Estimate a class-prior proxy without ground-truth labels."""
    row = np.arange(base_scores.shape[0], dtype=np.int64)
    pos = np.argmax(base_scores, axis=1)
    cls = topk_indices[row, pos]
    cmax = int(topk_indices.max()) + 1
    cnt = np.bincount(cls, minlength=cmax).astype(np.float32)
    return (cnt / (cnt.sum() + 1e-6)).astype(np.float32)


def _build_diff_scores(
    image_embeddings: np.ndarray, topk_text_embeddings: np.ndarray, rho: float
) -> np.ndarray:
    """Compute differential scores after removing the common text direction."""
    mu = l2norm(topk_text_embeddings.mean(axis=1), axis=1)[:, None, :]
    d = l2norm(topk_text_embeddings - float(rho) * mu, axis=2)
    return np.einsum("nd,nkd->nk", image_embeddings, d, optimize=True).astype(np.float32)


def build_cader_scores(
    image_embeddings: np.ndarray,
    topk_text_embeddings: np.ndarray,
    topk_indices: np.ndarray,
    base_scores: np.ndarray,
    cfg: CADERConfig,
) -> CADERResult:
    """Re-rank the fixed candidate set with CADER."""
    image_embeddings = l2norm(image_embeddings.astype(np.float32), axis=1)
    topk_text_embeddings = l2norm(topk_text_embeddings.astype(np.float32), axis=2)
    topk_indices = topk_indices.astype(np.int64)
    base_scores = base_scores.astype(np.float32)

    n, k = base_scores.shape
    row = np.arange(n, dtype=np.int64)
    base_pos = np.argmax(base_scores, axis=1)
    base_cls = topk_indices[row, base_pos]

    ncsr = build_ncsr_scores(
        image_embeddings=image_embeddings,
        topk_indices=topk_indices,
        base_scores=base_scores,
        policy_temp=float(cfg.policy_temp),
        knn=int(cfg.knn),
        lambda_scale=float(cfg.support_scale),
        lambda_override=None,
        ambiguity_power=0.0,
        use_ambiguity=False,
        consensus_mode=str(cfg.consensus_mode),
        ambiguity_mode="margin_std",
    )
    support_z = ncsr.support_z.astype(np.float32)
    lambda_auto = float(ncsr.lambda_auto)

    diff_scores = _build_diff_scores(
        image_embeddings=image_embeddings,
        topk_text_embeddings=topk_text_embeddings,
        rho=float(cfg.rho),
    ).astype(np.float32)

    ambiguity = _ambiguity_margin_std(base_scores).astype(np.float32)
    commonality = _commonality(topk_text_embeddings).astype(np.float32)

    sup_gain = (support_z - support_z[row, base_pos][:, None]).astype(np.float32)
    dif_gain = (diff_scores - diff_scores[row, base_pos][:, None]).astype(np.float32)
    sup_gain[row, base_pos] = -1e9
    dif_gain[row, base_pos] = -1e9

    cand = np.zeros_like(base_scores, dtype=np.float32)
    if cfg.use_support:
        cand += sup_gain
    if cfg.use_diff:
        cand += float(cfg.diff_scale) * dif_gain

    m = max(2, min(int(cfg.challenger_topm), k))
    order_base = np.argsort(-base_scores, axis=1)
    allow = np.zeros_like(base_scores, dtype=bool)
    if cfg.use_adaptive_topm:
        m_hi = m
        m_lo = max(2, m - 1)
        thr = float(cfg.adaptive_topm_thr)
        m_row = np.where(ambiguity > thr, m_hi, m_lo).astype(np.int64)
        for i in range(n):
            allow[i, order_base[i, : m_row[i]]] = True
    else:
        allow[row[:, None], order_base[:, :m]] = True
    allow[row, base_pos] = False
    cand = np.where(allow, cand, -1e9).astype(np.float32)

    agreement_gate = np.ones((n,), dtype=np.float32)
    if cfg.use_dual_agreement_gate and cfg.use_support and cfg.use_diff:
        sup_c = np.where(allow, sup_gain, -1e9).argmax(axis=1)
        dif_c = np.where(allow, dif_gain, -1e9).argmax(axis=1)
        agree = (sup_c == dif_c).astype(np.float32)
        agreement_gate = np.where(agree > 0.5, 1.0, 0.35).astype(np.float32)

    chal_pos = np.argmax(cand, axis=1)
    chal_score = cand[row, chal_pos].astype(np.float32)

    q = _sigmoid(chal_score)
    gate = np.ones((n,), dtype=np.float32)
    gate *= (chal_score > 0.0).astype(np.float32)
    gate *= (ambiguity > 0.18).astype(np.float32)

    if cfg.use_commonality_gate:
        gate *= (commonality > 0.35).astype(np.float32)

    if cfg.use_rarity_protect or cfg.use_microclass_lock:
        prior = _build_stage2_prior(base_scores=base_scores, topk_indices=topk_indices)
        p_base = prior[base_cls]
        med = float(np.median(prior))
        if cfg.use_rarity_protect:
            rare_factor = np.clip((p_base / (med + 1e-6)), 0.55, 1.10).astype(np.float32)
            gate *= rare_factor
        if cfg.use_microclass_lock:
            q10 = float(np.quantile(prior, 0.10))
            micro = (p_base <= q10).astype(np.float32)
            gate *= 1.0 - micro

    if cfg.use_dual_agreement_gate and cfg.use_support and cfg.use_diff:
        gate *= agreement_gate

    if cfg.use_positive_evidence_gate and cfg.use_support and cfg.use_diff:
        sup_sel = sup_gain[row, chal_pos]
        dif_sel = dif_gain[row, chal_pos]
        pos_ok = ((sup_sel > 0.0) & (dif_sel > 0.0)).astype(np.float32)
        gate *= np.where(pos_ok > 0.5, 1.0, 0.25).astype(np.float32)

    gate = np.clip(gate, 0.0, 1.0).astype(np.float32)

    delta = (float(cfg.eta) * ambiguity * (0.5 + 0.5 * commonality) * q * gate).astype(np.float32)
    final = base_scores.copy().astype(np.float32)
    final[row, chal_pos] += delta
    final[row, base_pos] -= delta

    mu = final.mean(axis=1, keepdims=True)
    sd = final.std(axis=1, keepdims=True) + 1e-6
    final = ((final - mu) / sd).astype(np.float32)

    return CADERResult(
        final_scores=final.astype(np.float32),
        support_z=support_z.astype(np.float32),
        diff_scores=diff_scores.astype(np.float32),
        challenger_pos=chal_pos.astype(np.int64),
        challenger_score=chal_score.astype(np.float32),
        ambiguity=ambiguity.astype(np.float32),
        commonality=commonality.astype(np.float32),
        gate=gate.astype(np.float32),
        delta=delta.astype(np.float32),
        lambda_auto=lambda_auto,
    )
