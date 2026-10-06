"""Training and inference entry points for the recognition stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ovcos.io import load_npz, read_config, write_json
from ovcos.recognition.cobre import CoBRe, top12_margin_torch
from ovcos.recognition.pipeline import recognize, validate_cache
from .losses import cobre_loss


def train(args):
    config = read_config(args.config)
    settings = config["training"]
    cache = load_npz(args.cache)
    validate_cache(cache)
    labels = cache.get("labels")
    if labels is None or (labels < 0).any() or (labels >= len(cache["class_names"])).any():
        raise ValueError("CoBRe training requires valid labels in a seen-class cache.")
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    order = rng.permutation(len(labels))
    pivot = int(len(order) * settings["train_ratio"])
    if not 0 < pivot < len(order):
        raise ValueError("At least two samples and a nonempty train/validation split are required.")
    train_ids, validation_ids = order[:pivot], order[pivot:]
    device = torch.device(args.device)
    arrays = [
        torch.from_numpy(cache[key].astype(np.float32))
        for key in ("image_embeddings", "global_image_embeddings", "local_image_embeddings")
    ]
    target = torch.from_numpy(labels.astype(np.int64))
    dataset = TensorDataset(*(array[train_ids] for array in arrays), target[train_ids])
    loader = DataLoader(dataset, batch_size=args.batch_size or settings["batch_size"], shuffle=True)
    model = CoBRe(dim=arrays[0].shape[1], **config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    texts = F.normalize(
        torch.from_numpy(cache["class_embeddings"].astype(np.float32)).to(device), dim=-1
    )
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory for CoBRe training.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "split.json",
        {
            "seed": settings["seed"],
            "train": train_ids.tolist(),
            "validation": validation_ids.tolist(),
        },
    )
    write_json(output / "config.json", config)
    best, steps, history = -1.0, 0, []
    for epoch in range(1, (args.epochs or settings["epochs"]) + 1):
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"CoBRe epoch {epoch}"):
            fused, global_view, local_view, truth = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = cobre_loss(model, fused, global_view, local_view, truth, texts, settings)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite recognition loss.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        model.eval()
        with torch.inference_mode():
            hits = 0
            for start in range(0, len(validation_ids), 256):
                ids = validation_ids[start : start + 256]
                fused, global_view, local_view = [
                    F.normalize(x[ids].to(device), dim=-1) for x in arrays
                ]
                margin = top12_margin_torch(fused @ texts.T)
                prediction = (model(fused, global_view, local_view, margin) @ texts.T).argmax(1)
                hits += int((prediction.cpu() == target[ids]).sum())
            accuracy = hits / len(validation_ids)
        state = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "validation_top1": accuracy,
        }
        torch.save(state, output / "last.pth")
        if accuracy > best:
            best = accuracy
            torch.save(state, output / "best.pth")
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), "validation_top1": accuracy}
        )
        write_json(output / "training.json", history)
        if args.max_steps and steps >= args.max_steps:
            break


def predict(args):
    cache = load_npz(args.cache)
    config = read_config(args.config)
    result = recognize(
        cache,
        config,
        checkpoint=args.cobre_checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        use_subclass=not args.no_subclass,
        use_cader=not args.no_cader,
    )
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError("Output exists. Use a new directory or --overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    names = cache["class_names"].tolist()
    metrics = {}
    for stage, ranking in result.stage_rankings.items():
        rows = [
            {
                "id": str(cache["ids"][i]),
                "pred_name": names[int(order[0])],
                "ranked_classes": [names[int(k)] for k in order],
                "ranked_indices": order.tolist(),
            }
            for i, order in enumerate(ranking)
        ]
        write_json(output / f"{stage}.json", {"stage": stage, "predictions": rows})
        if "labels" in cache and (cache["labels"] >= 0).all():
            metrics[stage] = {
                f"top{k}": float((ranking[:, :k] == cache["labels"][:, None]).any(1).mean())
                for k in range(1, ranking.shape[1] + 1)
            }
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "protocol.json",
        {
            "cader": not args.no_cader,
            "retrieval_pool": "other images in the input cache",
            "self_excluded": True,
            "num_samples": len(cache["ids"]),
            "num_classes": len(names),
            "candidate_size": result.candidate_indices.shape[1],
        },
    )
