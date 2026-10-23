#!/usr/bin/env bash
# Full OVCOS inference and evaluation on OVCamo-Unseen (3,770 images, 61 classes).
#   bash scripts/eval.sh
# Uses $OUT/idsam/last.pth and $OUT/cobre/best.pth produced by scripts/train.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

OVCAMO=${OVCAMO:-data/ovcamo}
QWEN=${QWEN:-Qwen/Qwen3-VL-Embedding-8B}
OUT=${OUT:-outputs}
IDSAM_CKPT=${IDSAM_CKPT:-$OUT/idsam/last.pth}
COBRE_CKPT=${COBRE_CKPT:-$OUT/cobre/best.pth}

skip() { [ -e "$1" ] && echo "[skip] $1 exists"; }

skip data/manifests/test.jsonl || idssam prepare-data --root "$OVCAMO" --output data/manifests

# 1. Class-agnostic localization with IDS-SAM
skip cache/instructions/test/image_embeddings.npy || idssam extract-instructions \
  --manifest data/manifests/test.jsonl --model "$QWEN" --output cache/instructions/test
skip "$OUT/masks/test/manifest.jsonl" || idssam segment \
  --manifest data/manifests/test.jsonl --instruction-cache cache/instructions/test \
  --checkpoint "$IDSAM_CKPT" --output "$OUT/masks/test"

# 2. Recognition: fused Top-5 -> CoBRe -> subclass alignment -> CADER (batch-transductive)
skip cache/recognition/test.npz || idssam extract-recognition \
  --manifest "$OUT/masks/test/manifest.jsonl" --vocabulary data/manifests/test_vocabulary.json \
  --model "$QWEN" --output cache/recognition/test.npz
idssam recognize --cache cache/recognition/test.npz --cobre-checkpoint "$COBRE_CKPT" \
  --output "$OUT/recognition/test" --overwrite

# 3. Metrics: class-agnostic masks, then class-aware OVCOS (Table 1)
idssam evaluate --manifest "$OUT/masks/test/manifest.jsonl" --output "$OUT/metrics/localization.json"
idssam evaluate --manifest "$OUT/masks/test/manifest.jsonl" \
  --predictions "$OUT/recognition/test/cader.json" --output "$OUT/metrics/ovcos.json"
echo "Top-1/Top-5 per stage: $OUT/recognition/test/metrics.json"
