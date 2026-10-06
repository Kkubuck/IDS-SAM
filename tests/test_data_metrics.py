import json
import numpy as np
import pytest
from PIL import Image

from ovcos.io import load_npz, save_npz
from ovcos.data.localization import InstructionCache
from ovcos.data.manifest import load_manifest, write_manifest
from ovcos.evaluation.ovcos_metrics import OVCOSMetricer


def test_cache_has_no_pickle(tmp_path):
    path = tmp_path / "cache.npz"
    save_npz(path, ids=np.array(["x"]), features=np.ones((1, 3), np.float32))
    assert load_npz(path)["ids"][0] == "x"
    with pytest.raises(ValueError):
        save_npz(path, bad=np.array([{}], dtype=object))


def test_missing_instruction_is_an_error(tmp_path):
    np.save(tmp_path / "image_embeddings.npy", np.ones((1, 5, 32), np.float32))
    np.save(tmp_path / "text_embeddings.npy", np.ones((5, 32), np.float32))
    (tmp_path / "image_index.json").write_text(json.dumps({"x": 0}))
    cache = InstructionCache(tmp_path)
    assert cache.get("x")[0].shape == (5, 32)
    with pytest.raises(KeyError):
        cache.get("unknown")


def test_manifest_does_not_require_labels(tmp_path):
    image = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(image)
    path = tmp_path / "images.jsonl"
    write_manifest(path, [{"id": "x", "image": "x.png"}])
    assert load_manifest(path)[0]["id"] == "x"
    write_manifest(path, [{"id": "x", "image": "x.png"}] * 2)
    with pytest.raises(ValueError):
        load_manifest(path)


def test_class_aware_wrong_label_penalty():
    mask = np.zeros((16, 16), np.uint8)
    mask[4:12, 4:12] = 255
    meter = OVCOSMetricer(["cat", "dog"])
    meter.step(mask, mask, "dog", "cat")
    result = meter._get_raw_results()
    assert result["mae"] == 1
    assert result["sm"] == result["wfm"] == result["avgiou"] == 0


def test_perfect_localization():
    mask = np.zeros((16, 16), np.uint8)
    mask[4:12, 4:12] = 255
    meter = OVCOSMetricer(["foreground"])
    meter.step(mask, mask, "foreground", "foreground")
    result = meter._get_raw_results()
    assert result["mae"] == 0
    assert result["sm"] == pytest.approx(1, abs=1e-12)
    assert result["wfm"] > 0.999


def test_single_foreground_pixel_has_finite_metrics():
    mask = np.zeros((16, 16), np.uint8)
    mask[8, 8] = 255
    meter = OVCOSMetricer(["foreground"])
    meter.step(mask, mask, "foreground", "foreground")
    assert all(np.isfinite(value) for value in meter._get_raw_results().values())
