# Release validation

The release was reviewed in three passes: source and interface review, runtime and source-parity checks, and independent archive installation and regression checks.

## Tested environment

- Python 3.10, PyTorch 2.5.1, TorchVision 0.20.1, CUDA 12.1, NumPy 1.26.4.
- Qwen embedding extra: Transformers 4.57.6 and qwen-vl-utils 0.0.14.
- CPU tests and CUDA execution; distributed localization training on three GPUs.

## Pass 1 — Source and interfaces

- Checked the localization, embedding, recognition, and evaluation dependencies.
- Separated model code, training engines, data formats, configuration, and CLI entry points.
- Preserved model state-dictionary keys and strict trained-checkpoint loading.
- Checked manifest IDs, cache dimensions, candidate-set consistency, and required inputs.
- Checked that recognition predictions do not consume ground-truth labels.
- Checked README commands, figure paths, source attributions, and included license notices.

## Pass 2 — Runtime and source parity

- Executed a full-size SAM ViT-H/IDS-SAM optimizer step with three-GPU DDP, using pretrained SAM initialization and three real training images.
- Executed native-resolution mask export on two real images.
- Compared original and modularized IDS-SAM inference with the same checkpoint and input: maximum logit difference `0.0`, identical exported mask bytes.
- Compared recognition on a preserved 3,770-image cache: all four stages agreed with the recorded Top-1 predictions for every image. Independently executed original recognition code produced identical candidate rankings; maximum final-score difference was `2.26e-6` or less.
- Executed CoBRe training on a 128-image seen-class cache and loaded its saved checkpoint.
- Extracted five instruction-view embeddings from a real image with Qwen3-VL-Embedding-8B.
- Extracted new global/local, class, and subclass embeddings for two real images; ran recognition and class-aware evaluation on the resulting cache.
- Compared bundled saliency metrics and the class-aware evaluator with the original implementations on random and boundary-case masks: maximum difference `0.0`.

## Pass 3 — Packaging and regression

- Installed the package from a freshly extracted repository archive into an isolated package target.
- Ran all nine CLI subcommand help pages and the CPU test suite.
- Tested localization training, checkpoint resume, prediction, and evaluation through the public CLI.
- Tested CoBRe training and recognition through the public CLI.
- Tested frozen encoder gradients, batch forward/backward, strict checkpoint round-trips, singleton inputs, fixed candidates, label-free inference, neighbor self-exclusion, class-aware penalties, and invalid input handling.
- Checked formatting, Python lint, README local links, ZIP CRCs, and extracted-file hashes.
- Verified that the archive contains no model weights, dataset files, embedding caches, machine-specific paths, or experiment outputs.

These checks cover executable software behavior and source-preserving refactoring. GPU training checks are short smoke runs; the paper's result tables are presented separately in the README.
