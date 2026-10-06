"""Evaluate saved masks, optionally with predicted class names."""

from __future__ import annotations

import cv2
import numpy as np
from tqdm import tqdm

from ovcos.data.manifest import load_manifest
from ovcos.io import read_json, write_json
from .ovcos_metrics import OVCOSMetricer


def evaluate(args):
    rows = load_manifest(args.manifest, require_masks=True)
    predictions = None
    if args.predictions:
        records = read_json(args.predictions)["predictions"]
        predictions = {r["id"]: r["pred_name"] for r in records}
        if len(predictions) != len(records) or set(predictions) != {r["id"] for r in rows}:
            raise ValueError(
                "Prediction IDs must exactly match the evaluation manifest, with no duplicates."
            )
        if any("class_name" not in row for row in rows):
            raise ValueError("Class-aware evaluation requires class_name in the manifest.")
    evaluator = OVCOSMetricer(sorted({r.get("class_name", "foreground") for r in rows}))
    for row in tqdm(rows, desc="evaluation"):
        if "prediction" not in row:
            raise ValueError(f"Missing predicted mask for {row['id']}")
        pred = cv2.imread(row["prediction"], cv2.IMREAD_GRAYSCALE)
        truth = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        if pred is None or truth is None:
            raise ValueError(f"Unreadable mask for {row['id']}")
        if pred.shape != truth.shape:
            pred = cv2.resize(
                pred, (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        true_class = row.get("class_name", "foreground")
        predicted_class = predictions[row["id"]] if predictions is not None else true_class
        evaluator.step(pred.astype(np.uint8), truth.astype(np.uint8), predicted_class, true_class)
    raw = evaluator._get_raw_results()
    result = {name: float(value) for name, value in raw.items()}
    write_json(
        args.output,
        {"samples": len(rows), "class_aware": predictions is not None, "metrics": result},
    )
    print(result)
