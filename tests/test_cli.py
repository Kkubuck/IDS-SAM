"""CPU smoke tests of the public training, inference, and evaluation commands."""

import json

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from ovcos.cli.main import main
from ovcos.data.manifest import load_manifest, write_manifest
from ovcos.io import read_json, save_npz
from ovcos.recognition.consensus import build_ncsr_scores


def test_localization_cli_train_resume_predict_evaluate(tmp_path, small_config):
    image = tmp_path / "sample.png"
    mask = tmp_path / "truth.png"
    Image.new("RGB", (64, 48), (128, 100, 110)).save(image)
    array = np.zeros((48, 64), np.uint8)
    array[12:36, 16:48] = 255
    Image.fromarray(array).save(mask)
    manifest = tmp_path / "samples.jsonl"
    write_manifest(manifest, [{"id": "sample", "image": str(image), "mask": str(mask)}])
    cache = tmp_path / "instructions"
    cache.mkdir()
    np.save(cache / "image_embeddings.npy", np.random.randn(1, 5, 32).astype(np.float32))
    np.save(cache / "text_embeddings.npy", np.random.randn(5, 32).astype(np.float32))
    (cache / "image_index.json").write_text(json.dumps({"sample": 0}))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(small_config))
    trained = tmp_path / "trained"
    common = [
        "--manifest",
        str(manifest),
        "--instruction-cache",
        str(cache),
        "--config",
        str(config),
        "--device",
        "cpu",
        "--workers",
        "0",
    ]
    main(
        [
            "train-segment",
            *common,
            "--initialize-random",
            "--output",
            str(trained),
            "--epochs",
            "1",
            "--max-steps",
            "1",
        ]
    )
    main(
        [
            "train-segment",
            *common,
            "--resume",
            str(trained / "last.pth"),
            "--output",
            str(trained),
            "--epochs",
            "2",
            "--max-steps",
            "2",
        ]
    )
    state = torch.load(trained / "last.pth", weights_only=True)
    assert state["epoch"] == state["steps"] == 2
    assert len(read_json(trained / "training.json")) == 2
    predicted = tmp_path / "predicted"
    main(
        ["segment", *common, "--checkpoint", str(trained / "last.pth"), "--output", str(predicted)]
    )
    with Image.open(predicted / "masks/sample.png") as result:
        assert result.size == (64, 48)
    metrics = tmp_path / "metrics.json"
    main(["evaluate", "--manifest", str(predicted / "manifest.jsonl"), "--output", str(metrics)])
    assert read_json(metrics)["samples"] == 1
    assert all(np.isfinite(v) for v in read_json(metrics)["metrics"].values())


def test_recognition_cli_train_predict(tmp_path, recognition_cache, recognition_config):
    cache = tmp_path / "cache.npz"
    save_npz(cache, **recognition_cache)
    config = tmp_path / "recognition.yaml"
    config.write_text(yaml.safe_dump(recognition_config))
    trained = tmp_path / "trained"
    main(
        [
            "train-recognizer",
            "--cache",
            str(cache),
            "--config",
            str(config),
            "--output",
            str(trained),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--max-steps",
            "1",
        ]
    )
    predicted = tmp_path / "predicted"
    main(
        [
            "recognize",
            "--cache",
            str(cache),
            "--config",
            str(config),
            "--output",
            str(predicted),
            "--device",
            "cpu",
            "--cobre-checkpoint",
            str(trained / "best.pth"),
        ]
    )
    assert len(read_json(predicted / "cader.json")["predictions"]) == 12
    assert read_json(predicted / "protocol.json")["self_excluded"]


def test_neighbor_support_excludes_query():
    # With identical embeddings and opposing predictions, support must come
    # entirely from the other image, including at the maximum legal k = n - 1.
    result = build_ncsr_scores(
        np.ones((2, 4), np.float32),
        np.array([[0, 1], [0, 1]]),
        np.array([[1.0, 0.0], [0.0, 1.0]], np.float32),
        knn=8,
    )
    np.testing.assert_allclose(result.support_scores, [[0.0, 1.0], [1.0, 0.0]], atol=2e-6)


@pytest.mark.parametrize("identifier", ["../x", "..", ".", "a/b", "a\\b"])
def test_unsafe_manifest_ids_rejected(tmp_path, identifier):
    path = tmp_path / "samples.jsonl"
    write_manifest(path, [{"id": identifier, "image": "unused.png"}])
    with pytest.raises(ValueError):
        load_manifest(path)


def test_nonfinite_cache_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_npz(tmp_path / "cache.npz", features=np.array([np.nan]))
