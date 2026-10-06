"""Command-line interface. Heavy model dependencies are imported on demand."""

from __future__ import annotations

import argparse
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(
        prog="ovcos", description="IDS-SAM / CoBRe / CADER training and inference"
    )
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-data", help="Create OVCamo train/test manifests and vocabularies"
    )
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--output", required=True)
    images = commands.add_parser(
        "manifest", help="Create an unlabeled manifest from an image directory"
    )
    images.add_argument("--images", required=True)
    images.add_argument("--output", required=True)
    for name in ("extract-instructions", "extract-recognition"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-8B")
        sub.add_argument("--output", required=True)
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
        sub.add_argument("--batch-size", type=int, default=1)
        sub.add_argument("--max-pixels", type=int, default=1800 * 32 * 32)
        sub.add_argument("--limit", type=int, default=0)
        sub.add_argument("--overwrite", action="store_true")
        if name == "extract-recognition":
            sub.add_argument("--vocabulary", required=True)
    for name in ("train-segment", "segment"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--instruction-cache", required=True)
        sub.add_argument(
            "--cache-split",
            default=None,
            help="Prefix for legacy train_/test_ instruction cache filenames",
        )
        sub.add_argument("--config", default=None)
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--batch-size", type=int, default=1)
        sub.add_argument("--workers", type=int, default=4)
        sub.add_argument("--output", required=True)
        if name == "train-segment":
            group = sub.add_mutually_exclusive_group()
            group.add_argument("--sam-checkpoint")
            group.add_argument("--resume")
            group.add_argument(
                "--initialize-random", action="store_true", help="For software smoke tests only"
            )
            sub.add_argument("--epochs", type=int, default=None)
            sub.add_argument(
                "--max-steps", type=int, default=0, help="Limit optimizer steps for a smoke test"
            )
            sub.add_argument("--val-manifest")
            sub.add_argument("--val-instruction-cache")
            sub.add_argument("--val-cache-split", default=None)
        else:
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--limit", type=int, default=0)
            sub.add_argument("--overwrite", action="store_true")
    recognition_config = str(Path(__file__).parents[1] / "configs" / "recognition.yaml")
    for name in ("train-recognizer", "recognize"):
        sub = commands.add_parser(name)
        sub.add_argument("--cache", required=True)
        sub.add_argument("--config", default=recognition_config)
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument(
            "--batch-size", type=int, default=None if name == "train-recognizer" else 256
        )
        sub.add_argument("--output", required=True)
        if name == "train-recognizer":
            sub.add_argument("--epochs", type=int, default=None)
            sub.add_argument("--max-steps", type=int, default=0)
        else:
            sub.add_argument("--cobre-checkpoint", required=True)
            sub.add_argument("--no-subclass", action="store_true")
            sub.add_argument("--no-cader", action="store_true")
            sub.add_argument("--overwrite", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--predictions")
    evaluation.add_argument("--output", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if hasattr(args, "batch_size") and args.batch_size is not None and args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    for field in ("limit", "max_steps", "workers"):
        if hasattr(args, field) and getattr(args, field) < 0:
            raise ValueError(f"{field} cannot be negative.")
    if getattr(args, "epochs", None) is not None and args.epochs < 1:
        raise ValueError("Epoch count must be positive.")
    if args.command == "prepare-data":
        from ovcos.data.manifest import prepare_ovcamo

        prepare_ovcamo(args.root, args.output)
    elif args.command == "manifest":
        from ovcos.data.manifest import image_rows, write_manifest

        write_manifest(args.output, image_rows(args.images))
    elif args.command == "extract-instructions":
        from ovcos.embeddings.extract import extract_instructions

        extract_instructions(args)
    elif args.command == "extract-recognition":
        from ovcos.embeddings.extract import extract_recognition

        extract_recognition(args)
    elif args.command in ("train-segment", "segment"):
        from ovcos.engine.segmentation import train, predict

        (train if args.command == "train-segment" else predict)(args)
    elif args.command in ("train-recognizer", "recognize"):
        from ovcos.engine.recognition import train, predict

        (train if args.command == "train-recognizer" else predict)(args)
    elif args.command == "evaluate":
        from ovcos.evaluation.run import evaluate

        evaluate(args)
