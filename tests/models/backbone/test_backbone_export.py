# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for backbone export behavior."""

import sys
from types import ModuleType

from rfdetr.models.backbone.backbone import Backbone


class TestBackboneExport:
    """Tests for ``Backbone.export``."""

    def test_export_without_lora_encoder_skips_peft_import_and_warning(self, monkeypatch) -> None:
        """Non-LoRA exports should not warn just because peft is unavailable."""
        backbone = object.__new__(Backbone)
        backbone.encoder = object()
        warning_messages: list[str] = []

        monkeypatch.delitem(sys.modules, "peft", raising=False)
        monkeypatch.setattr("rfdetr.models.backbone.backbone.logger.warning", warning_messages.append)

        backbone.export()

        assert warning_messages == []

    def test_export_replaces_peft_encoder_with_merged_encoder(self, monkeypatch) -> None:
        """Export should replace PEFT wrapper with merged base encoder."""

        class _MergedEncoder:
            pass

        class _FakePeftModel:
            def __init__(self) -> None:
                self._merged = _MergedEncoder()

            def merge_and_unload(self):
                return self._merged

        peft_module = ModuleType("peft")
        peft_module.PeftModel = _FakePeftModel
        monkeypatch.setitem(sys.modules, "peft", peft_module)

        backbone = object.__new__(Backbone)
        backbone.encoder = _FakePeftModel()

        backbone.export()

        assert isinstance(backbone.encoder, _MergedEncoder)
