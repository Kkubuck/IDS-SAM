<div align="center">

<h1>Uncertainty-Guided Structural Routing and<br>Consensus-Aware Re-Ranking for<br>Open-Vocabulary Camouflaged Object Segmentation</h1>

**ACCV 2026**

Jisang Lee<sup>&#42;</sup>, SeoYeon Oh<sup>&#42;</sup>, Cheoneum Park<sup>†</sup>, Haneol Jang<sup>†</sup><br>
Department of Computer Engineering, Hanbat National University<br>
<sup>&#42;</sup>Equal contribution · <sup>†</sup>Co-corresponding authors

[![Tests](https://github.com/Kkubuck/IDS-SAM/actions/workflows/tests.yml/badge.svg)](https://github.com/Kkubuck/IDS-SAM/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-555555.svg)](LICENSE)

[Getting started](docs/getting_started.md) · [Results](#results) · [Reproduction](docs/reproduction.md) · [Citation](#citation)

</div>

Official PyTorch implementation of our ACCV 2026 paper. The pipeline localizes a camouflaged object with **IDS-SAM**, then predicts its category through background-residual correction, subclass alignment, and consensus-aware re-ranking.

## Overview

![Pipeline overview: IDS-SAM localization followed by CoBRe, subclass alignment, and CADER recognition](assets/overview.png)

**IDS-SAM** (Instruction-Driven Structure-Aware SAM) combines multi-view instruction prompts with a phase-congruency prior. Instruction-view ambiguity controls structural routing, and pixel-wise uncertainty guides boundary refinement.

Recognition uses the predicted mask to form global and local image embeddings. **CoBRe** corrects their background residual, subclass alignment refines candidate text scores, and **CADER** re-ranks the fixed Top-5 candidates using neighboring images and differential evidence. Recognition does not feed back into localization.

## Results

OVCamo-Unseen: **3,770 images / 61 unseen classes**. Results below are reported in the paper and use predicted masks. The full model uses SAM ViT-H and Qwen3-VL-Embedding-8B.

### Class-aware segmentation

OVCOS methods from Table 1 of the main paper:

| Method | Venue | cS<sub>m</sub> ↑ | cF<sup>ω</sup><sub>β</sub> ↑ | cMAE ↓ | cF<sub>β</sub> ↑ | cE<sub>m</sub> ↑ | cIoU ↑ |
|:--|:--|--:|--:|--:|--:|--:|--:|
| OVCoser | ECCV 2024 | 0.579 | 0.490 | 0.336 | 0.520 | 0.616 | 0.443 |
| Classifier-centric | ICASSP 2026 | 0.658 | 0.547 | 0.239 | 0.582 | 0.696 | 0.493 |
| SuCLIP | ICCV 2025 | 0.667 | 0.594 | 0.242 | 0.633 | 0.722 | 0.540 |
| COCUS | CVM 2026 | 0.668 | 0.615 | 0.265 | 0.631 | 0.697 | 0.568 |
| **Ours (full pipeline)** | **ACCV 2026** | **0.752** | **0.691** | **0.168** | **0.711** | **0.788** | **0.635** |

### Recognition

Cumulative recognition stages with fixed IDS-SAM masks and the full CoBRe training objective (main Table 8; supplementary Tables S12 and S13):

| Stage | Top-1 (%) ↑ | Top-5 (%) ↑ |
|:--|--:|--:|
| Fused global/local baseline | 81.06 | 93.47 |
| + CoBRe | 81.80 | 93.47 |
| + Subclass alignment | 84.32 | 93.47 |
| + CADER | **85.12** | 93.47 |

All stages re-rank the same fused Top-5 set, so Top-5 accuracy remains unchanged. The full result uses official subclass metadata and **batch-transductive** CADER: each query retrieves from the other 3,769 unlabeled evaluation images. See [evaluation protocol](docs/reproduction.md#evaluation-protocol) for the per-image path.

<details>
<summary>Qualitative comparison</summary>

![Qualitative comparison on OVCamo-Unseen from Figure 3 of the paper](assets/qualitative.png)

</details>

## Installation

Python 3.10+ with a CUDA-enabled PyTorch installation. The reported core environment is PyTorch 2.5.1 / TorchVision 0.20.1 / CUDA 12.1.

```bash
git clone https://github.com/Kkubuck/IDS-SAM.git
cd IDS-SAM

python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[embedding,dev]'
ovcos --help
```

The Python package and CLI are named `ovcos`. For work with precomputed embeddings, install the core package with `pip install -e .`; the `embedding` extra is only needed to extract Qwen embeddings.

## Training and evaluation

Follow the [step-by-step guide](docs/getting_started.md) from data preparation through evaluation:

| Step | Commands |
|:--|:--|
| Prepare OVCamo | `ovcos prepare-data` |
| Cache instruction views | `ovcos extract-instructions` |
| Train IDS-SAM and predict masks | `ovcos train-segment`, `ovcos segment` |
| Cache global/local and text embeddings | `ovcos extract-recognition` |
| Train CoBRe and run recognition | `ovcos train-recognizer`, `ovcos recognize` |
| Evaluate localization or class-aware OVCOS | `ovcos evaluate` |

The guide also covers checkpoint resume, single-GPU and distributed training, and custom images and vocabularies. See [data formats](docs/data_formats.md) for manifests and caches, and [implementation](docs/implementation.md) for the module interfaces.

## Models and data

| Resource | Source / status |
|:--|:--|
| OVCamo dataset and official split | [OVCamo](https://github.com/lartpang/OVCamo) |
| Pretrained SAM ViT-H | [SAM checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| Frozen embedding model | [Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) |
| Trained IDS-SAM and CoBRe checkpoints | Not included; no pretrained download is available in this release. Follow the training guide to produce them. |

The paper used precomputed vLLM embeddings; the extraction commands in this repository use the Transformers backend. [Reproduction notes](docs/reproduction.md) document the settings and this backend difference.

## Tests

```bash
python -m pytest -q
ruff check ovcos tests
```

The CPU suite covers model gradients, checkpoints, training/inference commands, fixed candidate sets, label-free recognition, query self-exclusion, and evaluation. [Release validation](docs/validation.md) records the original runtime and source-parity checks.

## Citation

```bibtex
@inproceedings{lee2026ovcos,
  title={Uncertainty-Guided Structural Routing and Consensus-Aware Re-Ranking for Open-Vocabulary Camouflaged Object Segmentation},
  author={Lee, Jisang and Oh, SeoYeon and Park, Cheoneum and Jang, Haneol},
  booktitle={Asian Conference on Computer Vision},
  year={2026}
}
```

## Acknowledgments and license

This implementation builds on [Segment Anything](https://github.com/facebookresearch/segment-anything), [SAM-Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch), [OVCamo](https://github.com/lartpang/OVCamo), [PySODMetrics](https://github.com/lartpang/PySODMetrics), and [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding).

Code is released under [Apache-2.0](LICENSE); upstream components retain their respective licenses. See [third-party notices](THIRD_PARTY_NOTICES.md) for source attributions and the separate terms covering dataset images in the paper figures.
