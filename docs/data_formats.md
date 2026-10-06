# Data and artifact formats

## Dataset manifest

A manifest is UTF-8 JSON Lines, with one image per line. Paths may be absolute or relative to the manifest's directory. IDs must be unique, nonempty filenames without directory components.

```json
{"id":"sample_001","image":"images/sample_001.jpg"}
```

For segmentation training and evaluation, add `mask`. For class-aware evaluation, add `class_name`. Predicted masks are recorded as `prediction` by the `segment` command. Labels are needed for recognition training or accuracy calculation only.

```json
{"id":"sample_001","image":"images/sample_001.jpg","mask":"masks/sample_001.png","class_name":"moth","label":0}
```

Images and ground-truth masks must have matching dimensions. Use binary masks encoded as `0` and `255`. Training uses `value > 127`; the original OVCamo evaluation convention uses `value > 128`.

## Vocabulary

```json
[
  {"name":"moth","descriptors":["moth","peppered moth"]},
  {"name":"frog","descriptors":["frog","tree frog"]}
]
```

A list of strings is also accepted. Vocabulary order determines class indices. `prepare-data` derives vocabularies from the official class-level metadata; the unseen vocabulary is not used to train the recognition adapter.

## Instruction cache

| File | Shape or content |
|:--|:--|
| `image_embeddings.npy` | Float array `[N,5,D]` |
| `text_embeddings.npy` | Float array `[5,D]` |
| `image_index.json` | Exact mapping from image ID to row number |
| `metadata.json` | Model identifier, fixed instructions, text query, sample count |

The image index is mandatory. Missing images raise an error. The localizer normalizes each vector before selecting its first 1,024 dimensions and applying the learned projections. `--cache-split train` or `--cache-split test` supports caches whose three filenames have the corresponding prefix.

## Recognition cache

The NPZ archive contains numeric arrays and Unicode strings. It does not require pickle.

| Field | Shape | Role |
|:--|:--|:--|
| `image_embeddings` | `[N,D]` | Normalized fused global/local embedding |
| `global_image_embeddings` | `[N,D]` | Global image view |
| `local_image_embeddings` | `[N,D]` | Mask-conditioned image view |
| `class_embeddings` | `[C,D]` | Averaged class-template embeddings |
| `class_names` | `[C]` | Unique vocabulary strings |
| `ids` | `[N]` | Unique sample IDs |
| `topk_indices` | `[N,K]` | Optional cached fused candidate indices |
| `topk_text_embeddings` | `[N,K,D]` | Optional candidate text vectors |
| `subclass_embeddings` | `[S,D]` | Optional descriptor vectors |
| `subclass_owner_indices` | `[S]` | Owner class of each descriptor |
| `labels` | `[N]` | Optional ground-truth class indices |

`K = min(5,C)`. Once selected, the candidate set remains fixed across recognition stages. Class names and image IDs must be unique. Missing global/local embeddings are errors rather than implicit substitutions.

## Checkpoints

`segment` and `recognize` accept a plain tensor state dictionary, or a dictionary containing `model` or `state_dict`. A distributed `module.` prefix is removed automatically. Loading is strict: missing keys, unexpected keys, and tensor-shape differences produce an error. Checkpoints are loaded with `weights_only=True`.

The segmentation trainer's `last.pth` includes optimizer, scheduler, epoch, and step state for resume. The CoBRe trainer writes both `last.pth` and `best.pth`, with the best checkpoint selected by held-out validation Top-1 accuracy.

## Predictions and evaluation

Recognition writes one JSON file for each stage. Each row contains `id`, `pred_name`, `ranked_classes`, and `ranked_indices`. Class-aware evaluation requires an exact match between prediction IDs and manifest IDs. It never silently skips unmatched samples.

Localization PNGs store foreground probabilities quantized to 8-bit grayscale. Probability maps are resized to the original image dimensions with bilinear interpolation before quantization.
