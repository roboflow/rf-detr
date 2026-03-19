# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for checkpoint compatibility utilities."""

import os

import pytest
import pytorch_lightning
import torch

from rfdetr.training.checkpoint import convert_legacy_checkpoint


class TestConvertLegacyCheckpoint:
    """Tests for convert_legacy_checkpoint."""

    def _make_legacy_pth(self, path: str, num_classes: int = 2, epoch: int = 5) -> None:
        """Write a minimal legacy .pth checkpoint to *path*."""
        state = {
            "model": {
                "class_embed.bias": torch.zeros(num_classes),
                "class_embed.weight": torch.zeros(num_classes, 4),
            },
            "epoch": epoch,
        }
        torch.save(state, path)

    def test_converted_checkpoint_has_ptl_version(self, tmp_path):
        """convert_legacy_checkpoint output must contain pytorch-lightning_version."""
        src = str(tmp_path / "model.pth")
        dst = str(tmp_path / "model.ckpt")
        self._make_legacy_pth(src)

        convert_legacy_checkpoint(src, dst)

        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert "pytorch-lightning_version" in ckpt

    def test_converted_checkpoint_ptl_version_matches_installed(self, tmp_path):
        """pytorch-lightning_version in converted checkpoint must match the installed version."""
        src = str(tmp_path / "model.pth")
        dst = str(tmp_path / "model.ckpt")
        self._make_legacy_pth(src)

        convert_legacy_checkpoint(src, dst)

        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["pytorch-lightning_version"] == pytorch_lightning.__version__

    def test_converted_checkpoint_preserves_epoch(self, tmp_path):
        """Epoch stored in legacy .pth must be carried into the converted checkpoint."""
        src = str(tmp_path / "model.pth")
        dst = str(tmp_path / "model.ckpt")
        self._make_legacy_pth(src, epoch=7)

        convert_legacy_checkpoint(src, dst)

        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 7

    def test_missing_model_key_raises(self, tmp_path):
        """Legacy .pth without 'model' key should raise ValueError."""
        src = str(tmp_path / "bad.pth")
        dst = str(tmp_path / "bad.ckpt")
        torch.save({"weights": {}}, src)

        with pytest.raises(ValueError, match="'model' key"):
            convert_legacy_checkpoint(src, dst)


class TestEnsurePtlCheckpoint:
    """Tests for _ensure_ptl_checkpoint in rfdetr.detr."""

    def test_ptl_checkpoint_returned_unchanged(self, tmp_path):
        """A checkpoint that already has pytorch-lightning_version is returned as-is."""
        from rfdetr.detr import _ensure_ptl_checkpoint

        path = str(tmp_path / "native.ckpt")
        torch.save({"pytorch-lightning_version": pytorch_lightning.__version__, "state_dict": {}}, path)

        result_path, tmp = _ensure_ptl_checkpoint(path)

        assert result_path == path
        assert tmp is None

    def test_raw_checkpoint_gets_patched(self, tmp_path):
        """A raw checkpoint without pytorch-lightning_version gets a temp patched copy."""
        from rfdetr.detr import _ensure_ptl_checkpoint

        path = str(tmp_path / "raw.pth")
        torch.save({"model": {"w": torch.zeros(1)}}, path)

        result_path, tmp = _ensure_ptl_checkpoint(path)

        try:
            assert result_path != path
            assert tmp is not None
            assert os.path.exists(tmp)
            patched = torch.load(tmp, map_location="cpu", weights_only=False)
            assert "pytorch-lightning_version" in patched
            assert patched["pytorch-lightning_version"] == pytorch_lightning.__version__
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    def test_unreadable_checkpoint_returned_unchanged(self, tmp_path):
        """An unreadable/corrupt checkpoint path is returned as-is (let PTL handle the error)."""
        from rfdetr.detr import _ensure_ptl_checkpoint

        path = str(tmp_path / "nonexistent.ckpt")

        result_path, tmp = _ensure_ptl_checkpoint(path)

        assert result_path == path
        assert tmp is None
