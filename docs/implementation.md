# Implementation guide

## Localization

`ovcos.models.build.build_localizer` constructs IDS-SAM and handles strict checkpoint loading. The SAM encoder and learned high-pass adapters are in `models/sam/encoder.py`. `models/priors.py` computes image-derived structural maps. `models/layers.py` contains spatial FiLM and the edge/boundary refinement heads.

`IDSAM.forward(images, instruction_images, instruction_texts)` returns mask logits, the edge output, and instruction-view weights. The input image tensor is normalized with ImageNet mean and standard deviation. The two instruction tensors have shapes `[B,5,D]` and `[B,5,D]` (or `[5,D]` for shared texts). Four views contribute visual and text tokens, giving eight condition tokens.

The decoder uses the shared FiLM-modulated image features. Its routing bias acts on the mask-query attention path. Pixel uncertainty weights the final boundary residual. The model exposes all trainable modules through a standard `torch.nn.Module.forward`, so distributed training uses PyTorch DDP directly.

Parameter names are retained for compatibility with existing checkpoints. Tensor dimensions and strict loading are checked before inference.

## Recognition

`recognition.cobre.CoBRe` estimates a background prototype and a foreground residual from the global/local embedding pair. It blends the corrected vector with the fused anchor through learned gates. The training objective is implemented in `engine/losses.py`.

`recognition.subclass` uses class-owned descriptors within the fused candidate set. The default hard-positive policy retains a class name when a descriptor does not improve its score or lacks positive class specificity.

`recognition.consensus` computes label-free neighbor support with query self-exclusion. `recognition.cader` combines neighborhood and differential evidence to update candidate scores. `recognition.pipeline.recognize` returns stage-wise scores and complete candidate rankings without using ground-truth labels.

The training and prediction engines handle file formats and batching. Model functions can also be imported directly:

```python
from ovcos.io import read_config
from ovcos.models.build import build_localizer

config = read_config()
model = build_localizer(config, checkpoint="checkpoints/idsam.pth", device="cuda:0")
model.eval()
```

## Evaluation

The evaluator follows the OVCamo class-aware convention. For a category error, similarity and overlap scores are zero, and MAE is one. Omitting `--predictions` evaluates masks independently of recognition.

The returned keys `avgfm`, `avgem`, and `avgiou` are means over the threshold curves; `adp*` and `max*` are separately reported. `sm`, `wfm`, and `mae` are direct sample averages.
