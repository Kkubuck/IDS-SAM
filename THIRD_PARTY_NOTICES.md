# Third-party notices

The code in this repository is released under the [Apache License 2.0](LICENSE), with the following upstream components retaining their original notices and licenses.

| Component | Files | Source | License |
|:--|:--|:--|:--|
| SAM and ViTDet-derived transformer components | `idssam/models/sam/` | [Segment Anything](https://github.com/facebookresearch/segment-anything) | Apache-2.0, [`licenses/SAM-Apache-2.0.txt`](licenses/SAM-Apache-2.0.txt) |
| SAM encoder adapters | `idssam/models/sam/encoder.py` | [SAM-Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch) | MIT, [`licenses/SAM-Adapter-MIT.txt`](licenses/SAM-Adapter-MIT.txt) |
| Class-aware evaluation | `idssam/evaluation/ovcos_metrics.py` | [OVCamo](https://github.com/lartpang/OVCamo) | MIT, [`licenses/OVCamo-MIT.txt`](licenses/OVCamo-MIT.txt) |
| Saliency metric algorithms | `idssam/evaluation/saliency.py` | [PySODMetrics](https://github.com/lartpang/PySODMetrics) | MIT, [`licenses/PySODMetrics-MIT.txt`](licenses/PySODMetrics-MIT.txt) |
| Qwen multimodal embedding backend | `idssam/embeddings/qwen_model.py` | [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) | Apache-2.0, [`licenses/Qwen-Apache-2.0.txt`](licenses/Qwen-Apache-2.0.txt) |

The model/decoder files contain task-specific modifications for IDS-SAM. The Qwen backend includes stricter error handling for invalid image input. The repository keeps upstream copyright headers in the relevant source files.

`assets/overview.png` and `assets/qualitative.png` are taken from Figures 2 and 3 of the authors' camera-ready paper. OVCamo example photographs remain covered by the dataset's terms, including its CC BY-NC-SA 4.0 license. Apart from the examples embedded in these paper figures, dataset images are not distributed here. Model weights and embedding caches are not included.
