"""Create portable instruction and dual-view recognition caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ovcos.data.manifest import load_manifest, read_rgb
from ovcos.data.views import mask_conditioned_view
from ovcos.io import read_json, write_json, save_npz
from ovcos.recognition.array_ops import l2norm
from .prompts import CLASS_TEMPLATES, INSTRUCTIONS, RECOGNITION_INSTRUCTION


def create_embedder(args):
    from .qwen_model import Qwen3VLEmbedder

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        args.dtype
    ]
    return Qwen3VLEmbedder(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        attn_implementation="eager",
        max_pixels=args.max_pixels,
    )


def encode(embedder, inputs, batch_size, description):
    values = []
    for start in tqdm(range(0, len(inputs), batch_size), desc=description):
        output = embedder.process(inputs[start : start + batch_size]).float().cpu().numpy()
        if not np.isfinite(output).all():
            raise FloatingPointError(f"Non-finite embeddings: {description}")
        values.append(output)
    return l2norm(np.concatenate(values), 1)


def extract_instructions(args):
    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    output = Path(args.output)
    if (output / "image_embeddings.npy").exists() and not args.overwrite:
        raise FileExistsError("Instruction cache exists. Choose a new output or --overwrite.")
    embedder = create_embedder(args)
    output.mkdir(parents=True, exist_ok=True)
    visual = None
    textual = []
    for view, instruction in enumerate(INSTRUCTIONS):
        text = encode(
            embedder,
            [{"text": "camouflaged object", "instruction": instruction}],
            1,
            "instruction text",
        )
        textual.append(text[0])
        for start in tqdm(range(0, len(rows), args.batch_size), desc=f"view {view + 1}/5"):
            inputs = [
                {"image": row["image"], "instruction": instruction}
                for row in rows[start : start + args.batch_size]
            ]
            value = embedder.process(inputs).float().cpu().numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError("Non-finite instruction embeddings.")
            if visual is None:
                visual = np.lib.format.open_memmap(
                    output / "image_embeddings.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(rows), len(INSTRUCTIONS), value.shape[1]),
                )
            visual[start : start + len(value), view] = value
    visual.flush()
    np.save(output / "text_embeddings.npy", np.stack(textual))
    write_json(output / "image_index.json", {row["id"]: i for i, row in enumerate(rows)})
    write_json(
        output / "metadata.json",
        {
            "model": args.model,
            "instructions": INSTRUCTIONS,
            "text_query": "camouflaged object",
            "count": len(rows),
            "dtype": args.dtype,
        },
    )


def extract_recognition(args):
    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError("Recognition cache exists. Choose a new output or --overwrite.")
    if any("prediction" not in row for row in rows):
        raise ValueError(
            "Recognition extraction requires IDS-SAM prediction paths in the manifest."
        )
    vocabulary = read_json(args.vocabulary)
    vocabulary = [{"name": v, "descriptors": []} if isinstance(v, str) else v for v in vocabulary]
    names = [v["name"] for v in vocabulary]
    if not names or len(names) != len(set(names)):
        raise ValueError("Vocabulary must contain unique class names.")
    embedder = create_embedder(args)
    class_text = []
    for name in names:
        values = encode(
            embedder,
            [
                {"text": t.format(class_name=name), "instruction": RECOGNITION_INSTRUCTION}
                for t in CLASS_TEMPLATES
            ],
            args.batch_size,
            name,
        )
        class_text.append(l2norm(values.mean(0, keepdims=True), 1)[0])
    class_text = np.asarray(class_text, np.float32)
    descriptor_names, owners = [], []
    for index, entry in enumerate(vocabulary):
        for name in entry.get("descriptors", []):
            descriptor_names.append(str(name))
            owners.append(index)
    descriptor_text = []
    for name in descriptor_names:
        values = encode(
            embedder,
            [
                {"text": t.format(class_name=name), "instruction": RECOGNITION_INSTRUCTION}
                for t in CLASS_TEMPLATES
            ],
            args.batch_size,
            name,
        )
        descriptor_text.append(l2norm(values.mean(0, keepdims=True), 1)[0])
    descriptor_text = np.asarray(descriptor_text, np.float32).reshape(-1, class_text.shape[1])
    global_chunks, local_chunks = [], []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="dual-view embeddings"):
        chunk = rows[start : start + args.batch_size]
        global_inputs, local_inputs = [], []
        for row in chunk:
            image = read_rgb(row["image"])
            with Image.open(row["prediction"]) as mask:
                mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
            local = mask_conditioned_view(image, mask, blur_sigma=1.5)
            global_inputs.append({"image": image, "instruction": RECOGNITION_INSTRUCTION})
            local_inputs.append({"image": local, "instruction": RECOGNITION_INSTRUCTION})
        global_chunks.append(embedder.process(global_inputs).float().cpu().numpy())
        local_chunks.append(embedder.process(local_inputs).float().cpu().numpy())
    global_view, local_view = [
        l2norm(np.concatenate(chunks).astype(np.float32), 1)
        for chunks in (global_chunks, local_chunks)
    ]
    fused = l2norm(0.5 * global_view + 0.5 * local_view, 1)
    topk = np.argsort(-(fused @ class_text.T), axis=1)[:, : min(5, len(names))]
    payload = dict(
        image_embeddings=fused,
        global_image_embeddings=global_view,
        local_image_embeddings=local_view,
        class_embeddings=class_text,
        class_names=np.asarray(names, dtype=str),
        subclass_embeddings=descriptor_text,
        subclass_owner_indices=np.asarray(owners, np.int64),
        topk_indices=topk,
        topk_text_embeddings=class_text[topk],
        ids=np.asarray([r["id"] for r in rows], dtype=str),
    )
    if all("class_name" in r for r in rows):
        payload["labels"] = np.asarray(
            [names.index(r["class_name"]) if r["class_name"] in names else -1 for r in rows],
            np.int64,
        )
    save_npz(output, **payload)
    write_json(
        output.with_suffix(".json"),
        {
            "model": args.model,
            "instruction": RECOGNITION_INSTRUCTION,
            "templates": CLASS_TEMPLATES,
            "samples": len(rows),
            "classes": len(names),
        },
    )
