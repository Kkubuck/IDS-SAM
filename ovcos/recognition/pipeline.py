"""Label-free recognition over a fixed fused candidate set."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from ovcos.io import load_weights
from .array_ops import l2norm, rank_classes_from_scores
from .cobre import CoBRe, top12_margin_torch
from .cader import CADERConfig, build_cader_scores
from .subclass import build_stage2_subclass_adaptive


@dataclass
class RecognitionResult:
    candidate_indices: np.ndarray
    corrected_embeddings: np.ndarray
    stage_scores: dict
    stage_rankings: dict


def validate_cache(cache):
    required = (
        "image_embeddings",
        "global_image_embeddings",
        "local_image_embeddings",
        "class_embeddings",
        "class_names",
        "ids",
    )
    for key in required:
        if key not in cache:
            raise ValueError(f"Recognition cache missing {key}.")
    image = cache["image_embeddings"]
    texts = cache["class_embeddings"]
    if image.ndim != 2 or texts.ndim != 2 or image.shape[1] != texts.shape[1]:
        raise ValueError("Image and class embeddings must be [N,D] and [C,D].")
    if len(image) == 0 or len(texts) == 0:
        raise ValueError("Image and vocabulary sets must be nonempty.")
    for name in ("global_image_embeddings", "local_image_embeddings"):
        if cache[name].shape != image.shape:
            raise ValueError(f"Shape mismatch for {name}.")
    if len(cache["class_names"]) != len(texts) or len(set(cache["class_names"].tolist())) != len(
        texts
    ):
        raise ValueError("Class names must uniquely match text embedding rows.")
    if len(cache["ids"]) != len(image) or len(set(cache["ids"].tolist())) != len(image):
        raise ValueError("Sample IDs must uniquely match image embedding rows.")


@torch.inference_mode()
def recognize(
    cache,
    config,
    checkpoint=None,
    device="cpu",
    batch_size=256,
    use_subclass=True,
    use_cader=True,
    model=None,
):
    validate_cache(cache)
    image = l2norm(cache["image_embeddings"].astype(np.float32), 1)
    texts = l2norm(cache["class_embeddings"].astype(np.float32), 1)
    top_k = min(config["inference"]["top_k"], len(texts))
    if "topk_indices" in cache:
        candidates = cache["topk_indices"].astype(np.int64)
        if (
            candidates.shape != (len(image), top_k)
            or (candidates < 0).any()
            or (candidates >= len(texts)).any()
        ):
            raise ValueError("Invalid fixed candidate-set shape or class indices.")
        if any(len(set(row.tolist())) != top_k for row in candidates):
            raise ValueError("A candidate row contains duplicate classes.")
    else:
        candidates = np.argsort(-(image @ texts.T), axis=1)[:, :top_k]
    candidate_texts = (
        l2norm(cache["topk_text_embeddings"].astype(np.float32), 2)
        if "topk_text_embeddings" in cache
        else texts[candidates]
    )
    if candidate_texts.shape != (len(image), top_k, image.shape[1]):
        raise ValueError("Invalid candidate text embedding shape.")
    if model is None:
        if not checkpoint:
            raise ValueError("Recognition requires a trained --cobre-checkpoint.")
        model = CoBRe(dim=image.shape[1], **config["model"])
        model.load_state_dict(load_weights(checkpoint), strict=True)
    model = model.to(device).eval()
    class_tensor = F.normalize(torch.from_numpy(texts).to(device), dim=-1)
    chunks = []
    for start in range(0, len(image), batch_size):
        end = start + batch_size
        fused, global_view, local_view = [
            F.normalize(torch.from_numpy(x[start:end].astype(np.float32)).to(device), dim=-1)
            for x in (image, cache["global_image_embeddings"], cache["local_image_embeddings"])
        ]
        margin = top12_margin_torch(fused @ class_tensor.T)
        chunks.append(model(fused, global_view, local_view, margin).cpu().numpy())
    corrected = l2norm(np.concatenate(chunks).astype(np.float32), 1)
    score = lambda images, prompts: np.einsum("nd,nkd->nk", images, prompts, optimize=True).astype(
        np.float32
    )
    stages = {"fused": score(image, candidate_texts), "cobre": score(corrected, candidate_texts)}
    aligned_texts = candidate_texts
    subclass_scores = stages["cobre"]
    descriptors = cache.get("subclass_embeddings", np.empty((0, image.shape[1]), np.float32))
    if use_subclass and len(descriptors) and top_k >= 2:
        owners = cache["subclass_owner_indices"].astype(np.int64)
        if len(owners) != len(descriptors) or (owners < 0).any() or (owners >= len(texts)).any():
            raise ValueError("Descriptor owners must refer to vocabulary indices.")
        alignment = build_stage2_subclass_adaptive(
            corrected,
            candidate_texts,
            candidates,
            l2norm(descriptors.astype(np.float32), 1),
            owners,
            texts,
            mode=config["inference"]["subclass_policy"],
            mix_text=config["inference"]["mix_subclass_text"],
        )
        aligned_texts, subclass_scores = alignment.text_embeddings, alignment.scores
    stages["subclass"] = subclass_scores
    if use_cader:
        if len(image) >= 2 and top_k >= 2:
            result = build_cader_scores(
                corrected,
                l2norm(aligned_texts, 2),
                candidates,
                subclass_scores,
                CADERConfig(**config["inference"]["cader"]),
            )
            stages["cader"] = result.final_scores
        else:
            stages["cader"] = subclass_scores.copy()
    rankings = {
        name: rank_classes_from_scores(value, candidates)[1] for name, value in stages.items()
    }
    return RecognitionResult(candidates, corrected, stages, rankings)
