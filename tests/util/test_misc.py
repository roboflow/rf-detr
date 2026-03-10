# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from types import SimpleNamespace

import pytest
import torch

from rfdetr.utilities.state_dict import strip_checkpoint


class TestStripCheckpoint:
    def test_strip_checkpoint_keeps_only_model_and_args(self, tmp_path):
        checkpoint_path = tmp_path / "checkpoint_best_total.pth"
        torch.save(
            {
                "model": {"weight": torch.tensor([1.0])},
                "args": SimpleNamespace(class_names=["a"]),
                "optimizer": {"lr": 1e-4},
            },
            checkpoint_path,
        )

        strip_checkpoint(str(checkpoint_path))

        stripped = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert set(stripped.keys()) == {"model", "args"}

    def test_strip_checkpoint_is_atomic_when_save_fails(self, tmp_path, monkeypatch):
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

        with pytest.raises(RuntimeError, match="simulated save failure"):
            strip_checkpoint(str(checkpoint_path))

        recovered = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert set(recovered.keys()) == set(original_checkpoint.keys())
        assert recovered["model"]["weight"].equal(original_checkpoint["model"]["weight"])
        assert recovered["optimizer"] == original_checkpoint["optimizer"]
