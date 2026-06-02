# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for ``RFDETR.export_for_roboflow``.

``export_for_roboflow`` is the extracted, network-free core of ``deploy_to_roboflow``: it writes ``weights.pt`` (a dict
with ``"model"`` and ``"args"`` keys) and ``class_names.txt`` into a target directory.  A lightweight stub stands in for
``self.model`` so the file-writing contract is exercised without building a real model or downloading weights.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from rfdetr.detr import RFDETR


def _make_stub_model(class_names: list[str]) -> RFDETR:
    """Build an RFDETR instance whose model/state are stubbed for export_for_roboflow.

    ``RFDETR.__init__`` is bypassed; only the attributes ``export_for_roboflow`` reads are populated.
    """
    instance = RFDETR.__new__(RFDETR)
    args = SimpleNamespace(resolution=560)
    inner_module = SimpleNamespace(state_dict=lambda: {"weight": torch.zeros(2, 2)})
    instance.model = SimpleNamespace(model=inner_module, args=args, class_names=class_names)
    return instance


class TestExportForRoboflow:
    """export_for_roboflow writes a deploy-ready bundle into a directory."""

    def test_writes_weights_pt_with_model_and_args(self, tmp_path: Path) -> None:
        """weights.pt is a dict with 'model' and 'args', and args carries resolution."""
        model = _make_stub_model(["cat", "dog"])

        model.export_for_roboflow(str(tmp_path))

        bundle = torch.load(tmp_path / "weights.pt", map_location="cpu", weights_only=False)
        assert set(bundle) == {"model", "args"}
        assert "weight" in bundle["model"]
        assert bundle["args"].resolution == 560

    def test_writes_class_names_txt(self, tmp_path: Path) -> None:
        """class_names.txt lists one class name per line."""
        model = _make_stub_model(["cat", "dog"])

        model.export_for_roboflow(str(tmp_path))

        assert (tmp_path / "class_names.txt").read_text(encoding="utf-8") == "cat\ndog"

    def test_embeds_class_names_in_args(self, tmp_path: Path) -> None:
        """class_names are embedded in the saved args namespace when absent."""
        model = _make_stub_model(["cat", "dog"])

        model.export_for_roboflow(str(tmp_path))

        bundle = torch.load(tmp_path / "weights.pt", map_location="cpu", weights_only=False)
        assert bundle["args"].class_names == ["cat", "dog"]

    def test_does_not_overwrite_existing_args_class_names(self, tmp_path: Path) -> None:
        """args.class_names already set on the model is preserved in the saved bundle."""
        model = _make_stub_model(["cat", "dog"])
        model.model.args.class_names = ["pre_existing"]

        model.export_for_roboflow(str(tmp_path))

        bundle = torch.load(tmp_path / "weights.pt", map_location="cpu", weights_only=False)
        assert bundle["args"].class_names == ["pre_existing"]

    def test_empty_class_names_writes_empty_file(self, tmp_path: Path) -> None:
        """Empty class_names list produces an empty class_names.txt (no trailing newline)."""
        model = _make_stub_model([])

        model.export_for_roboflow(str(tmp_path))

        assert (tmp_path / "class_names.txt").read_text(encoding="utf-8") == ""

    def test_creates_output_dir_when_missing(self, tmp_path: Path) -> None:
        """output_dir is created if it does not already exist."""
        model = _make_stub_model(["cat", "dog"])
        target = tmp_path / "nested" / "bundle"

        model.export_for_roboflow(str(target))

        assert (target / "weights.pt").exists()
        assert (target / "class_names.txt").exists()
