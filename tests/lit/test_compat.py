# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for rfdetr.lit.compat.evaluate — the engine.evaluate() bridge."""

from unittest.mock import MagicMock, patch

import torch

from rfdetr.lit.compat import evaluate


def _make_module() -> MagicMock:
    """Return a MagicMock shaped like RFDETRModule (model, criterion, postprocess, _args)."""
    module = MagicMock(name="RFDETRModule")
    module.model = MagicMock(name="model")
    module.criterion = MagicMock(name="criterion")
    module.postprocess = MagicMock(name="postprocess")
    module._args = MagicMock(name="_args")
    return module


class TestEvaluate:
    """evaluate() delegates to engine.evaluate with the correct arguments."""

    def test_delegates_to_engine_evaluate(self) -> None:
        """evaluate() calls _engine_evaluate with all module components and extra args."""
        module = _make_module()
        data_loader = MagicMock(name="data_loader")
        base_ds = MagicMock(name="base_ds")
        device = torch.device("cpu")
        sentinel = ({"results_json": {"map": 0.5}}, MagicMock())

        with patch("rfdetr.lit.compat._engine_evaluate", return_value=sentinel) as mock_eval:
            result = evaluate(module, data_loader, base_ds, device)

        mock_eval.assert_called_once_with(
            module.model,
            module.criterion,
            module.postprocess,
            data_loader,
            base_ds,
            device,
            args=module._args,
        )
        assert result is sentinel

    def test_return_value_passthrough(self) -> None:
        """evaluate() returns exactly what _engine_evaluate returns."""
        stats = {"results_json": {"map": 0.42, "f1_score": 0.55}, "loss": 1.2}
        coco_eval = MagicMock(name="coco_evaluator")

        with patch("rfdetr.lit.compat._engine_evaluate", return_value=(stats, coco_eval)):
            returned_stats, returned_coco_eval = evaluate(_make_module(), MagicMock(), MagicMock(), torch.device("cpu"))

        assert returned_stats is stats
        assert returned_coco_eval is coco_eval

    def test_extracts_model_from_module(self) -> None:
        """evaluate() forwards module.model as the first positional arg, not the module itself."""
        module = _make_module()
        sentinel_model = object()
        module.model = sentinel_model

        with patch("rfdetr.lit.compat._engine_evaluate", return_value=({}, None)) as mock_eval:
            evaluate(module, MagicMock(), MagicMock(), torch.device("cpu"))

        assert mock_eval.call_args[0][0] is sentinel_model

    def test_uses_module_args_as_keyword(self) -> None:
        """evaluate() passes module._args as the 'args' keyword argument."""
        module = _make_module()
        sentinel_args = object()
        module._args = sentinel_args

        with patch("rfdetr.lit.compat._engine_evaluate", return_value=({}, None)) as mock_eval:
            evaluate(module, MagicMock(), MagicMock(), torch.device("cpu"))

        assert mock_eval.call_args[1]["args"] is sentinel_args

    def test_segmentation_stats_passthrough(self) -> None:
        """evaluate() passes through results_json_masks for segmentation models."""
        stats = {
            "results_json": {"map": 0.6, "f1_score": 0.65},
            "results_json_masks": {"map": 0.58, "f1_score": 0.62},
            "coco_eval_bbox": [0.6, 0.4],
            "coco_eval_masks": [0.58, 0.38],
        }

        with patch("rfdetr.lit.compat._engine_evaluate", return_value=(stats, None)):
            returned_stats, _ = evaluate(_make_module(), MagicMock(), MagicMock(), torch.device("cpu"))

        assert "results_json" in returned_stats
        assert "results_json_masks" in returned_stats
        assert returned_stats["results_json_masks"]["map"] == 0.58
