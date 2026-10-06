"""Single-GPU and distributed IDS-SAM training, validation and prediction."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ovcos.data.localization import InstructionCache, LocalizationDataset
from ovcos.data.manifest import load_manifest, write_manifest
from ovcos.io import read_config, read_json, write_json
from ovcos.models.build import build_localizer
from .losses import localization_loss


def _device_batch(batch, device):
    return [batch[key].to(device) for key in ("image", "instruction_images", "instruction_texts")]


def train(args):
    config = read_config(args.config)
    settings = config["training"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group("nccl")
    else:
        device = torch.device(args.device)
    torch.manual_seed(settings["seed"] + rank)
    np.random.seed(settings["seed"] + rank)
    random.seed(settings["seed"] + rank)
    rows = load_manifest(args.manifest, require_masks=True)
    cache = InstructionCache(args.instruction_cache, args.cache_split)
    dataset = LocalizationDataset(rows, cache, config["model"]["inp_size"])
    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=settings["seed"]) if world > 1 else None
    )
    batch_size = args.batch_size or settings["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    if not args.resume and not args.sam_checkpoint and not args.initialize_random:
        raise ValueError(
            "Training requires --sam-checkpoint, --resume, or explicit --initialize-random."
        )
    model = build_localizer(
        config, sam_checkpoint=args.sam_checkpoint if not args.resume else None, device=device
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    epochs = args.epochs or settings["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=settings["minimum_learning_rate"]
    )
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(
            "Output directory is not empty. Choose a new directory or use --resume."
        )
    start, best, steps = 1, float("-inf"), 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best, steps = state["epoch"] + 1, state["best_iou"], state["steps"]
        model._global_step = steps
    wrapped = (
        DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
        if world > 1
        else model
    )
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "config.json", config)
    if world > 1:
        dist.barrier()
    history_path = output / "training.json"
    history = read_json(history_path) if args.resume and history_path.is_file() else []
    for epoch in range(start, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        wrapped.train()
        total_loss, count = 0.0, 0
        for batch in tqdm(loader, desc=f"epoch {epoch}/{epochs}", disable=rank != 0):
            optimizer.zero_grad(set_to_none=True)
            outputs = wrapped(*_device_batch(batch, device))
            loss, _ = localization_loss(outputs, batch["mask"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite segmentation loss.")
            loss.backward()
            optimizer.step()
            model._global_step += 1
            steps += 1
            total_loss += float(loss.detach())
            count += 1
            if args.max_steps and steps >= args.max_steps:
                break
        scheduler.step()
        statistics = torch.tensor([total_loss, count], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(statistics)
        validation_iou = None
        if rank == 0:
            if args.val_manifest:
                validation_iou = validate(
                    model,
                    args.val_manifest,
                    args.val_instruction_cache or args.instruction_cache,
                    args.val_cache_split,
                    config["model"]["inp_size"],
                    device,
                )
            improved = validation_iou is not None and validation_iou > best
            if improved:
                best = validation_iou
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "steps": steps,
                "best_iou": best,
                "config": config,
            }
            torch.save(state, output / "last.pth")
            if improved:
                torch.save(
                    {"model": model.state_dict(), "config": config, "epoch": epoch},
                    output / "best.pth",
                )
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(statistics[0] / statistics[1]),
                    "validation_iou": validation_iou,
                }
            )
            write_json(output / "training.json", history)
        if world > 1:
            dist.barrier()
        if args.max_steps and steps >= args.max_steps:
            break
    if world > 1:
        dist.destroy_process_group()


@torch.no_grad()
def validate(model, manifest, directory, split, size, device):
    data = LocalizationDataset(
        load_manifest(manifest, True), InstructionCache(directory, split), size
    )
    model.eval()
    total = 0.0
    for batch in DataLoader(data, batch_size=1):
        pred = model(*_device_batch(batch, device))["mask_logits"].sigmoid() > 0.5
        gt = batch["mask"].to(device) > 0.5
        total += float((pred & gt).sum() / (pred | gt).sum().clamp(min=1))
    return total / len(data)


@torch.inference_mode()
def predict(args):
    config = read_config(args.config)
    device = torch.device(args.device)
    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    cache = InstructionCache(args.instruction_cache, args.cache_split)
    dataset = LocalizationDataset(rows, cache, config["model"]["inp_size"], require_masks=False)
    model = build_localizer(config, checkpoint=args.checkpoint, device=device).eval()
    output = Path(args.output).resolve()
    if (output / "manifest.jsonl").exists() and not args.overwrite:
        raise FileExistsError(
            "Prediction manifest already exists. Use a new --output or --overwrite."
        )
    (output / "masks").mkdir(parents=True, exist_ok=True)
    records, row_by_id = [], {r["id"]: r for r in rows}
    for batch in tqdm(
        DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers),
        desc="localization",
    ):
        logits = model(*_device_batch(batch, device))["mask_logits"]
        for i, identifier in enumerate(batch["id"]):
            shape = (int(batch["height"][i]), int(batch["width"][i]))
            probability = logits[i, 0].sigmoid().cpu().numpy()
            array = cv2.resize(probability, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            destination = output / "masks" / f"{identifier}.png"
            Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8)).save(destination)
            records.append({**row_by_id[identifier], "prediction": str(destination)})
    write_manifest(output / "manifest.jsonl", records)
