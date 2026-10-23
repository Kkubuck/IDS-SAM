# Getting started

[README](../README.md) · [Data formats](DATA_FORMATS.md)

This guide runs each stage of the pipeline separately. [`scripts/train.sh`](../scripts/train.sh) and [`scripts/eval.sh`](../scripts/eval.sh) chain the same commands. Run everything from the repository root in a Bash shell; every command is also available as `python -m idssam <command>`.

| Step | Command | Output |
|:--|:--|:--|
| 0. Manifests | `idssam prepare-data` | `data/manifests/{train,test}.jsonl`, vocabularies |
| 1. Instruction views | `idssam extract-instructions` | `cache/instructions/<split>/` |
| 2. IDS-SAM | `idssam train-segment` | `outputs/idsam/last.pth` |
| 3. Masks | `idssam segment` | `outputs/masks/<split>/` |
| 4. Recognition cache | `idssam extract-recognition` | `cache/recognition/<split>.npz` |
| 5. CoBRe + recognition | `idssam train-recognizer`, `idssam recognize` | `outputs/cobre/best.pth`, stage rankings |
| 6. Evaluation | `idssam evaluate` | metric JSON |

Commands never overwrite existing outputs unless `--overwrite` (or `--resume` for training) is given.

## 0. Data and pretrained models

Download [OVCamo](https://github.com/lartpang/OVCamo) and keep its split: 7,713 training images from 14 seen classes and 3,770 test images from 61 unseen classes.

```text
data/ovcamo/
├── dataset/
│   ├── class_info.json
│   └── sample_info.json
└── ovcamo_dataset/
    ├── train/{image,mask}/
    └── test/{image,mask}/
```

```bash
idssam prepare-data --root data/ovcamo --output data/manifests
```

This writes `train.jsonl`, `test.jsonl`, and the vocabularies `train_vocabulary.json` / `test_vocabulary.json`, whose subclass descriptors come from the `sub_class` field of the official `class_info.json`.

Pretrained models:

- **SAM ViT-H**: download [`sam_vit_h_4b8939.pth`](https://github.com/facebookresearch/segment-anything#model-checkpoints) to `checkpoints/`.
- **Qwen3-VL-Embedding-8B**: pass the Hugging Face ID `Qwen/Qwen3-VL-Embedding-8B` or a local directory to `--model`.

## 1. Instruction embeddings

Each image is embedded under the five fixed instructions in [`idssam/embeddings/prompts.py`](../idssam/embeddings/prompts.py) (foreground, texture, boundary, parts, shape/pose).

```bash
for split in train test; do
  CUDA_VISIBLE_DEVICES=0 idssam extract-instructions \
    --manifest data/manifests/$split.jsonl \
    --model Qwen/Qwen3-VL-Embedding-8B \
    --output cache/instructions/$split
done
```

## 2. Train IDS-SAM

The pretrained SAM image encoder stays frozen except for its adapters; the instruction projections, FiLM, routing bias, edge and boundary-refinement modules are trained. The paper recipe is [`idssam/configs/idsam.yaml`](../idssam/configs/idsam.yaml): 1024 × 1024 input, 20 epochs, AdamW (lr 2e-4, weight decay 1e-2, cosine schedule), batch size 1 per GPU on 3 GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 -m idssam train-segment \
  --manifest data/manifests/train.jsonl \
  --instruction-cache cache/instructions/train \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --output outputs/idsam
```

- **Single GPU:** replace `torchrun --standalone --nproc_per_node=3 -m idssam` with `idssam`.
- **Resume:** `--resume outputs/idsam/last.pth` restores model, optimizer and scheduler state.
- **Validation:** `--val-manifest` and `--val-instruction-cache` additionally save the best validation-IoU model as `best.pth`.
- **Localization ablations:** the paper uses matched 10-epoch controls (`--epochs 10`); module switches are in the `model` section of the config (pass a copy with `--config`).

## 3. Predict masks

```bash
for split in train test; do
  CUDA_VISIBLE_DEVICES=0 idssam segment \
    --manifest data/manifests/$split.jsonl \
    --instruction-cache cache/instructions/$split \
    --checkpoint outputs/idsam/last.pth \
    --output outputs/masks/$split
done
```

Each output folder holds 8-bit probability masks at the original image size and a `manifest.jsonl` that links every image to its mask. Ground truth is not needed for prediction.

## 4. Recognition caches

The global image, its mask-conditioned local view (background blurred), six class-name templates and the subclass descriptors are embedded once and cached.

```bash
for split in train test; do
  CUDA_VISIBLE_DEVICES=0 idssam extract-recognition \
    --manifest outputs/masks/$split/manifest.jsonl \
    --vocabulary data/manifests/${split}_vocabulary.json \
    --model Qwen/Qwen3-VL-Embedding-8B \
    --output cache/recognition/$split.npz
done
```

## 5. Train CoBRe and recognize

CoBRe is the only trained recognition module. It is fit on the seen-class cache with a fixed 80/20 split, and the checkpoint with the best held-out Top-1 accuracy is kept ([`idssam/configs/recognition.yaml`](../idssam/configs/recognition.yaml): 20 epochs, AdamW lr 1e-3, batch 64).

```bash
CUDA_VISIBLE_DEVICES=0 idssam train-recognizer \
  --cache cache/recognition/train.npz --output outputs/cobre

CUDA_VISIBLE_DEVICES=0 idssam recognize \
  --cache cache/recognition/test.npz \
  --cobre-checkpoint outputs/cobre/best.pth \
  --output outputs/recognition/test
```

`recognize` writes one ranking per stage (`fused.json`, `cobre.json`, `subclass.json`, `cader.json`), `metrics.json` with Top-1…Top-5 accuracy when labels are present, and `protocol.json`. Ground-truth labels are never used to make predictions.

## 6. Evaluate

```bash
# Class-agnostic localization
idssam evaluate --manifest outputs/masks/test/manifest.jsonl \
  --output outputs/metrics/localization.json

# Class-aware OVCOS (Table 1)
idssam evaluate --manifest outputs/masks/test/manifest.jsonl \
  --predictions outputs/recognition/test/cader.json \
  --output outputs/metrics/ovcos.json
```

The evaluator reports structure measure, weighted F-measure, MAE, and adaptive / mean / maximum F-measure, E-measure and IoU. In class-aware mode a wrong category scores zero on similarity and overlap metrics and one on MAE, following the OVCamo evaluator. Keys `avgfm`, `avgem` and `avgiou` are means over threshold curves; `sm`, `wfm` and `mae` are per-image averages.

## Evaluation protocol

- **Fixed candidates.** CoBRe, subclass alignment and CADER re-rank the same fused Top-5 classes, so Top-5 accuracy is identical across stages while Top-1 to Top-4 can change.
- **Batch-transductive CADER.** For each test image, CADER retrieves its k = 8 nearest neighbors among the *other* unlabeled test images (the query is excluded) and uses their predicted labels. The reported result runs `recognize` on the full 3,770-image cache at once; `--batch-size` only batches the CoBRe forward pass and does not split the retrieval pool.
- **Per-image recognition.** `--no-cader` gives the per-image path; `--no-subclass` uses class names only (Table 8). A single image falls back to the subclass-stage result.
- **Embedding backend.** The paper's caches were computed with a vLLM backend; the extraction commands here use Hugging Face Transformers with the versions pinned in the `embedding` extra. Keep model, precision, prompts and embedding dimension fixed when comparing caches.

## Custom images and vocabularies

```bash
idssam manifest --images path/to/images --output data/custom.jsonl
```

Use this manifest for `extract-instructions` and `segment`. A vocabulary is a JSON list of class names or of `{"name": ..., "descriptors": [...]}` objects; see [Data formats](DATA_FORMATS.md).

## Python API

```python
from idssam.io import read_config
from idssam.models.build import build_localizer

model = build_localizer(read_config(), checkpoint="outputs/idsam/last.pth", device="cuda:0").eval()
out = model(images, instruction_images, instruction_texts)
# images: [B,3,1024,1024], ImageNet-normalized; instruction tensors: [B,5,D] and [B,5,D] (or [5,D])
masks = out["mask_logits"].sigmoid()
```

Recognition on a cache is `idssam.recognition.pipeline.recognize(cache, config, checkpoint=...)`, which returns stage-wise scores and rankings.
