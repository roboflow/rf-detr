# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for stripping training state out of a saved checkpoint with ``strip_checkpoint``."""

from types import SimpleNamespace

import pytest
import torch

from rfdetr.utilities.state_dict import strip_checkpoint


class TestStripCheckpoint:
    """strip_checkpoint reduces a training checkpoint to the entries inference needs."""

    @pytest.mark.parametrize(
        ("extra", "expected"),
        [
            pytest.param({}, {"model", "args"}, id="no-model-name"),
            pytest.param({"model_name": "RFDETRSmall"}, {"model", "args", "model_name"}, id="model-name-preserved"),
        ],
    )
    def test_keeps_only_model_args_and_model_name(self, tmp_path, extra: dict, expected: set) -> None:
        """Optimizer state is dropped; ``model_name`` survives when present and is not invented when absent."""
        checkpoint_path = tmp_path / "checkpoint_best_total.pth"
        torch.save(
            {
                "model": {"weight": torch.tensor([1.0])},
                "args": SimpleNamespace(class_names=["a"]),
                "optimizer": {"lr": 1e-4},
                **extra,
            },
            checkpoint_path,
        )

        strip_checkpoint(str(checkpoint_path))

        stripped = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert set(stripped.keys()) == expected
        assert stripped.get("model_name") == extra.get("model_name")

    def test_is_atomic_when_save_fails(self, tmp_path, monkeypatch) -> None:
        """A failed save of the stripped copy leaves the original checkpoint on disk untouched."""
        checkpoint_path = tmp_path / "checkpoint_best_total.pth"
        original_checkpoint = {
            "model": {"weight": torch.tensor([1.0])},
            "args": SimpleNamespace(class_names=["a"]),
            "optimizer": {"lr": 1e-4},
        }
        torch.save(original_checkpoint, checkpoint_path)

        original_torch_save = torch.save

        def failing_torch_save(obj, destination, *args, **kwargs):
            if str(destination) != str(checkpoint_path):
                raise RuntimeError("simulated save failure")
            return original_torch_save(obj, destination, *args, **kwargs)

        monkeypatch.setattr(torch, "save", failing_torch_save)

        with pytest.raises(RuntimeError) as exc_info:
            strip_checkpoint(str(checkpoint_path))

        assert "simulated save failure" in str(exc_info.value)
        recovered = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert set(recovered.keys()) == set(original_checkpoint.keys())
        assert recovered["model"]["weight"].equal(original_checkpoint["model"]["weight"])
        assert recovered["optimizer"] == original_checkpoint["optimizer"]
