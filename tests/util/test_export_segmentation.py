# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Integration test for segmentation model export.

Tests the fix for the bug where exporting a segmentation model would crash
with 'AttributeError: dict object has no attribute shape' because pred_masks
can be either a tensor or a dictionary depending on the model configuration.
"""

import shutil
import tempfile

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for export test")
class TestSegmentationModelExport:
    """
    Integration test that actually loads a segmentation model and tests export.

    Requires CUDA and ONNX dependencies.
    """

    @pytest.fixture
    def output_dir(self):
        """Create a temporary directory for export output."""
        tmp_dir = tempfile.mkdtemp()
        yield tmp_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_segmentation_model_export_no_crash(self, output_dir):
        """
        Test that exporting a segmentation model does not crash.

        This is the actual integration test that exercises the full export path.
        """
        try:
            from rfdetr import RFDETRSegNano
        except ImportError:
            pytest.skip("rfdetr not installed")

        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx not installed, run: pip install rfdetr[onnxexport]")

        # Create model without pretrained weights (random initialization)
        model = RFDETRSegNano()

        # This should not crash with "AttributeError: 'dict' object has no attribute 'shape'"
        model.export(output_dir=output_dir, simplify=False)

        # Verify export produced output files
        output_path = Path(output_dir)
        onnx_files = list(output_path.glob("*.onnx"))
        assert len(onnx_files) > 0, "Export should produce ONNX file(s)"
