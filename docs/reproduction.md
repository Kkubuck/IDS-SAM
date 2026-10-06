# Reproduction notes

[README](../README.md) · [Commands](getting_started.md) · [Data formats](data_formats.md)

## Paper settings

The README tables reproduce the submitted paper's reported results. The default full pipeline uses IDS-SAM with SAM ViT-H, Qwen3-VL-Embedding-8B, CoBRe, official subclass descriptors, and CADER.

| Setting | Paper configuration | Release entry point |
|:--|:--|:--|
| Dataset | OVCamo: 7,713 training images / 14 seen classes; 3,770 evaluation images / 61 unseen classes | `ovcos prepare-data` |
| IDS-SAM input | 1024 × 1024 | [`idsam.yaml`](../ovcos/configs/idsam.yaml) |
| Instruction prompting | 5 fixed views; 4 active views; 8 condition tokens | [`prompts.py`](../ovcos/embeddings/prompts.py), `inst_prompt.top_k: 4` |
| IDS-SAM optimization | AdamW, 20 epochs, learning rate 2e-4, weight decay 1e-2; batch size 1 per GPU | `ovcos train-segment` |
| Localization ablations | Matched 10-epoch controls | `--epochs 10` |
| CoBRe optimization | AdamW, 20 epochs, learning rate 1e-3, weight decay 1e-4, batch size 64 | [`recognition.yaml`](../ovcos/configs/recognition.yaml) |
| CoBRe validation | 80/20 split of cached seen-class embeddings; best validation Top-1 checkpoint | `ovcos train-recognizer` |
| Candidate set | Fused Top-5; fixed through every recognition stage | `inference.top_k: 5` |
| CADER | 8 neighbors; support/differential weights 0.32 / 0.42 | `inference.cader` |

The paper's distributed localization training used three GPUs. The guide provides the corresponding launch command; `--workers` controls data-loader workers per process. GPU model and memory requirements depend on the embedding backend, precision, image size, and batch size.

## Evaluation protocol

**Localization.** IDS-SAM predicts a binary foreground mask without category names. The pretrained SAM encoder weights stay frozen while encoder adapters and the adaptation modules are trained. Recognition does not feed back into the mask prediction.

**Recognition.** CoBRe is trained on seen classes. The vocabulary and optional subclass descriptors define the unseen-class candidates at evaluation. Ground-truth test labels are used only to calculate metrics, never to generate recognition predictions.

**CADER.** The reported full result is batch-transductive. Run recognition on the complete 3,770-image evaluation cache: each query retrieves from the other 3,769 images, excludes itself, and uses neighbors' predicted labels. Processing separate subsets changes the retrieval pool and therefore the experiment. The `--batch-size` option batches the CoBRe forward pass; it does not partition the CADER retrieval pool.

For per-image recognition, pass `--no-cader`. Pass `--no-subclass` to use class names only. These switches define different experimental settings from the full result. A single-image input falls back to the sample-wise subclass result because no neighbors are available.

**Fixed candidates.** Fused, CoBRe, subclass, and CADER outputs rank the same Top-5 classes. Their Top-5 accuracy is consequently identical; Top-1 through Top-4 can change.

**Metrics.** Omitting `--predictions` from `ovcos evaluate` gives class-agnostic localization scores. Supplying recognition predictions enables the OVCamo class-aware convention: a wrong category receives zero for similarity/overlap metrics and one for MAE. Adaptive, mean, and maximum threshold statistics are exported separately; see the [metric keys](implementation.md#evaluation).

## Embedding backend and checkpoints

The supplementary material reports cached embeddings from a **vLLM** backend. This release's `extract-instructions` and `extract-recognition` commands use **Transformers**, with the dependency versions pinned by the `embedding` extra. Both use Qwen3-VL-Embedding, but this release does not establish numerical equivalence between the two extraction backends. Keep model version, precision, preprocessing, prompts, and embedding dimensions consistent when producing or comparing caches.

Pretrained SAM and Qwen weights are obtained from their upstream projects. Trained IDS-SAM and CoBRe weights and the paper's original embedding caches are not included or linked in this release. The training commands produce new checkpoints; exact table reproduction also depends on the original model artifacts and evaluation protocol.

The published CPU tests check software behavior on small inputs. The earlier GPU smoke runs and source-parity checks are described in [release validation](validation.md); they are separate from a full retraining of the paper's reported experiments.
