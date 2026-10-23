#!/usr/bin/env bash
# Train IDS-SAM and CoBRe on the OVCamo seen split (paper setting).
#   bash scripts/train.sh
# Override any variable inline, e.g.  GPUS=1 QWEN=/models/Qwen3-VL-Embedding-8B bash scripts/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OVCAMO=${OVCAMO:-data/ovcamo}                       # OVCamo root (dataset/, ovcamo_dataset/)
SAM_CKPT=${SAM_CKPT:-checkpoints/sam_vit_h_4b8939.pth}
QWEN=${QWEN:-Qwen/Qwen3-VL-Embedding-8B}            # HF model ID or local directory
GPUS=${GPUS:-3}                                     # GPUs for IDS-SAM training
OUT=${OUT:-outputs}

# Steps whose outputs already exist are skipped, so the script can be re-run after an interruption.
skip() { [ -e "$1" ] && echo "[skip] $1 exists"; }

# 0. Manifests and vocabularies
skip data/manifests/train.jsonl || idssam prepare-data --root "$OVCAMO" --output data/manifests

# 1. Five instruction views per image from the frozen embedding model
skip cache/instructions/train/image_embeddings.npy || idssam extract-instructions \
  --manifest data/manifests/train.jsonl --model "$QWEN" --output cache/instructions/train

# 2. IDS-SAM localizer (20 epochs, AdamW 2e-4, batch 1 per GPU)
if [ "$GPUS" -gt 1 ]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="$GPUS" -m idssam)
else
  LAUNCH=(idssam)
fi
"${LAUNCH[@]}" train-segment \
  --manifest data/manifests/train.jsonl --instruction-cache cache/instructions/train \
  --sam-checkpoint "$SAM_CKPT" --output "$OUT/idsam"

# 3. Masks for the training images, then global/local + class/subclass embeddings
skip "$OUT/masks/train/manifest.jsonl" || idssam segment \
  --manifest data/manifests/train.jsonl --instruction-cache cache/instructions/train \
  --checkpoint "$OUT/idsam/last.pth" --output "$OUT/masks/train"
skip cache/recognition/train.npz || idssam extract-recognition \
  --manifest "$OUT/masks/train/manifest.jsonl" --vocabulary data/manifests/train_vocabulary.json \
  --model "$QWEN" --output cache/recognition/train.npz

# 4. CoBRe on cached seen-class embeddings (best held-out Top-1 -> $OUT/cobre/best.pth)
idssam train-recognizer --cache cache/recognition/train.npz --output "$OUT/cobre"
