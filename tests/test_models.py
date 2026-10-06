import numpy as np
import torch
import pytest

from ovcos.models.build import build_localizer
from ovcos.engine.losses import localization_loss, cobre_loss
from ovcos.recognition.cobre import CoBRe
from ovcos.recognition.pipeline import recognize


def test_localization_backward_and_checkpoint(small_config, tmp_path):
    model = build_localizer(small_config)
    model.train()
    output = model(torch.randn(1, 3, 64, 64), torch.randn(1, 5, 32), torch.randn(1, 5, 32))
    assert output["mask_logits"].shape == (1, 1, 64, 64)
    assert output["view_weights"].shape == (1, 5)
    loss, _ = localization_loss(output, (torch.rand(1, 1, 64, 64) > 0.5).float())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad is None
        for n, p in model.named_parameters()
        if n.startswith("image_encoder.") and "prompt_generator" not in n
    )
    assert model.inst_proj_vis.weight.grad is not None
    checkpoint = tmp_path / "model.pth"
    torch.save({"model": model.state_dict()}, checkpoint)
    restored = build_localizer(small_config, checkpoint=checkpoint).eval()
    model.eval()
    inputs = (torch.randn(1, 3, 64, 64), torch.randn(1, 5, 32), torch.randn(1, 5, 32))
    with torch.no_grad():
        torch.testing.assert_close(
            model(*inputs)["mask_logits"], restored(*inputs)["mask_logits"], rtol=0, atol=0
        )


def test_batch_forward(small_config):
    model = build_localizer(small_config).eval()
    with torch.no_grad():
        result = model(torch.randn(2, 3, 64, 64), torch.randn(2, 5, 32), torch.randn(2, 5, 32))
    assert result["mask_logits"].shape == (2, 1, 64, 64)


def test_cobre_objective_has_finite_gradients(recognition_cache, recognition_config):
    cache = recognition_cache
    model = CoBRe(dim=32, **recognition_config["model"])
    views = [
        torch.from_numpy(cache[k])
        for k in ("image_embeddings", "global_image_embeddings", "local_image_embeddings")
    ]
    loss, _ = cobre_loss(
        model,
        *views,
        torch.from_numpy(cache["labels"]),
        torch.from_numpy(cache["class_embeddings"]),
        recognition_config["training"],
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.proto_bg.grad is not None and torch.isfinite(model.proto_bg.grad).all()
    assert model.gate[0].weight.grad.abs().sum() > 0


def test_recognition_uses_no_ground_truth_and_preserves_candidates(
    recognition_cache, recognition_config
):
    cache = recognition_cache
    model = CoBRe(dim=32, **recognition_config["model"])
    result = recognize(cache, recognition_config, model=model)
    cache_without_truth = {k: v for k, v in cache.items() if k != "labels"}
    unlabeled = recognize(cache_without_truth, recognition_config, model=model)
    for stage, ranked in result.stage_rankings.items():
        np.testing.assert_array_equal(ranked, unlabeled.stage_rankings[stage])
        np.testing.assert_array_equal(np.sort(ranked, 1), np.sort(result.candidate_indices, 1))
        assert np.isfinite(result.stage_scores[stage]).all()


@pytest.mark.parametrize("num_images,num_classes", [(1, 1), (1, 7), (2, 1), (2, 2)])
def test_recognition_small_inputs(recognition_cache, recognition_config, num_images, num_classes):
    cache = recognition_cache
    cache = {
        k: (
            v[:num_images]
            if k
            in (
                "image_embeddings",
                "global_image_embeddings",
                "local_image_embeddings",
                "ids",
                "labels",
            )
            else v
        )
        for k, v in cache.items()
    }
    cache["class_embeddings"] = cache["class_embeddings"][:num_classes]
    cache["class_names"] = cache["class_names"][:num_classes]
    cache.pop("subclass_embeddings")
    cache.pop("subclass_owner_indices")
    model = CoBRe(dim=32, **recognition_config["model"])
    result = recognize(cache, recognition_config, model=model)
    assert result.stage_rankings["cader"].shape == (num_images, min(5, num_classes))
    assert np.isfinite(result.stage_scores["cader"]).all()


def test_checkpoint_shape_errors_are_not_silenced(small_config, tmp_path):
    path = tmp_path / "incomplete.pth"
    torch.save({"model": {"wrong": torch.zeros(1)}}, path)
    with pytest.raises(RuntimeError):
        build_localizer(small_config, checkpoint=path)
