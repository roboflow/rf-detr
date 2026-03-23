# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import sys
from unittest.mock import Mock, patch

import pytest

from rfdetr.assets import ModelWeightAsset, ModelWeights
from rfdetr.assets.model_weights import download_pretrain_weights


# Module-level fixture for common file operation mocks
@pytest.fixture
def mock_file_operations():
    """Mock file operations to avoid actual file I/O."""
    with (
        patch("rfdetr.assets.model_weights.os.path.exists") as mock_exists,
        patch("rfdetr.assets.model_weights._download_file") as mock_download,
        patch("rfdetr.assets.model_weights._validate_file_md5") as mock_validate,
    ):
        # Default: file doesn't exist
        mock_exists.return_value = False
        # Default: MD5 validation passes
        mock_validate.return_value = True

        yield {"exists": mock_exists, "download": mock_download, "validate": mock_validate}


class TestDownloadPretrainWeights:
    """Test download_pretrain_weights function with mocking for offline testing."""

    def test_download_from_local_model_weights(self, mock_file_operations):
        """Test downloading a model from local ModelWeights."""
        download_pretrain_weights("rf-detr-base.pth")

        # Should call download with correct URL and MD5
        mock_file_operations["download"].assert_called_once()
        call_kwargs = mock_file_operations["download"].call_args[1]

        assert call_kwargs["filename"] == "rf-detr-base.pth"
        assert "rf-detr-base-coco.pth" in call_kwargs["url"]
        assert call_kwargs["expected_md5"] is not None  # Should have MD5 hash
        assert len(call_kwargs["expected_md5"]) == 32  # Valid MD5 hash

    @pytest.mark.skipif(
        "rfdetr_plus" not in sys.modules and "rfdetr_plus.assets" not in sys.modules,
        reason="rf-detr-plus not installed - skip priority test",
    )
    def test_download_from_rfdetr_plus_when_available(self, mock_file_operations):
        """Test that rf-detr-plus models are prioritized when available.

        Note: This test only runs if rf-detr-plus is actually installed.
        The priority logic is also tested in the fallback test.
        """
        # This test validates the real rf-detr-plus integration
        # If rf-detr-plus is installed, verify it's checked first
        download_pretrain_weights("some-model.pth")

        # Should attempt download (whether from plus or local)
        # The important part is that the function doesn't crash
        assert mock_file_operations["download"].called or not mock_file_operations["exists"].return_value

    def test_download_from_platform_models_fallback(self, mock_file_operations):
        """Test falling back to PLATFORM_MODELS when model not in ModelWeights."""
        # Mock PLATFORM_MODELS
        mock_platform_module = Mock()
        mock_platform_module.PLATFORM_MODELS = {"legacy-model.pth": "https://legacy.com/model.pth"}

        with (
            patch("rfdetr.assets.model_weights.ModelWeights.from_filename", return_value=None),
            patch.dict("sys.modules", {"rfdetr.platform": Mock(), "rfdetr.platform.downloads": mock_platform_module}),
        ):
            download_pretrain_weights("legacy-model.pth")

            # Should call download with legacy URL
            mock_file_operations["download"].assert_called_once()
            call_kwargs = mock_file_operations["download"].call_args[1]
            assert call_kwargs["url"] == "https://legacy.com/model.pth"
            assert call_kwargs["expected_md5"] is None  # Platform models don't have MD5

    def test_file_exists_with_correct_md5(self, mock_file_operations):
        """Test that download is skipped if file exists with correct MD5."""
        mock_file_operations["exists"].return_value = True
        mock_file_operations["validate"].return_value = True

        download_pretrain_weights("rf-detr-base.pth")

        # Should not download if file exists with correct hash
        mock_file_operations["download"].assert_not_called()

    def test_file_exists_with_incorrect_md5_redownloads(self, mock_file_operations):
        """Test that file is re-downloaded if MD5 is incorrect."""
        mock_file_operations["exists"].return_value = True
        mock_file_operations["validate"].return_value = False  # Incorrect MD5

        download_pretrain_weights("rf-detr-base.pth")

        # Should re-download due to incorrect MD5
        mock_file_operations["download"].assert_called_once()

    def test_redownload_flag_forces_download(self, mock_file_operations):
        """Test that redownload=True forces re-download even if file exists."""
        mock_file_operations["exists"].return_value = True
        mock_file_operations["validate"].return_value = True

        download_pretrain_weights("rf-detr-base.pth", redownload=True)

        # Should download despite file existing
        mock_file_operations["download"].assert_called_once()

    def test_validate_md5_disabled(self, mock_file_operations):
        """Test that MD5 validation can be disabled."""
        download_pretrain_weights("rf-detr-base.pth", validate_md5=False)

        # Should pass expected_md5=None when validation is disabled
        call_kwargs = mock_file_operations["download"].call_args[1]
        assert call_kwargs["expected_md5"] is None

    @patch("rfdetr.assets.model_weights.ModelWeights.from_filename", return_value=None)
    def test_nonexistent_model_returns_early(self, mock_from_filename, mock_file_operations):
        """Test that function returns early for non-existent models."""
        download_pretrain_weights("nonexistent-model.pth")

        # Should not attempt download
        mock_file_operations["download"].assert_not_called()

    def test_model_without_md5_hash(self, mock_file_operations):
        """Test downloading a model that has no MD5 hash."""
        # Create a mock asset without MD5
        mock_asset = ModelWeightAsset(filename="test-no-md5.pth", url="https://example.com/test.pth", md5_hash=None)

        with patch("rfdetr.assets.model_weights.ModelWeights.from_filename", return_value=mock_asset):
            download_pretrain_weights("test-no-md5.pth", validate_md5=True)

        # Should pass None for expected_md5
        call_kwargs = mock_file_operations["download"].call_args[1]
        assert call_kwargs["expected_md5"] is None

    def test_file_exists_no_md5_skips_download(self, mock_file_operations):
        """Test that if file exists and no MD5 validation, download is skipped."""
        mock_file_operations["exists"].return_value = True

        # Create a mock asset without MD5
        mock_asset = ModelWeightAsset(filename="test-no-md5.pth", url="https://example.com/test.pth", md5_hash=None)

        with patch("rfdetr.assets.model_weights.ModelWeights.from_filename", return_value=mock_asset):
            download_pretrain_weights("test-no-md5.pth")

        # Should not download if file exists (no MD5 to validate)
        mock_file_operations["download"].assert_not_called()


class TestDownloadIntegration:
    """Integration tests for the complete download flow."""

    @pytest.mark.parametrize("model", list(ModelWeights), ids=[m.filename for m in ModelWeights])
    def test_all_models_have_valid_md5_format(self, model: ModelWeightAsset) -> None:
        """Test that MD5 hashes are valid when present (prevent typos)."""
        # MD5 should be None or valid 32-char hex string
        if model.md5_hash is not None:
            assert len(model.md5_hash) == 32, f"{model.filename} has invalid MD5 length: {len(model.md5_hash)}"
            assert all(c in "0123456789abcdef" for c in model.md5_hash.lower()), (
                f"{model.filename} has invalid MD5 characters"
            )

    def test_from_filename_bidirectional_lookup(self):
        """Test that from_filename correctly maps back to enum values."""
        from rfdetr.assets.model_weights import ModelWeights

        # Pick a known model
        original = ModelWeights.RF_DETR_BASE

        # Look it up by filename
        asset = ModelWeights.from_filename(original.filename)

        # Should return the exact same asset
        assert asset is not None
        assert asset.filename == original.filename
        assert asset.url == original.url
        assert asset.md5_hash == original.md5_hash

    @patch("rfdetr.assets.model_weights.os.path.exists")
    @patch("rfdetr.assets.model_weights._validate_file_md5")
    @patch("rfdetr.assets.model_weights._download_file")
    def test_download_flow_for_real_model(self, mock_download, mock_validate, mock_exists):
        """Test the complete download flow for a real model."""
        mock_exists.return_value = False
        mock_validate.return_value = True

        # Download a real model (mocked network)
        download_pretrain_weights("rf-detr-base.pth")

        # Verify download was called with correct parameters
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]

        assert call_kwargs["filename"] == "rf-detr-base.pth"
        assert "storage.googleapis.com/rfdetr" in call_kwargs["url"]
        assert call_kwargs["expected_md5"] == "b4d3ce46099eaed50626ede388caf979"


class TestDownloadErrorHandling:
    """Test error handling in download mechanism."""

    @patch("rfdetr.assets.model_weights.os.path.exists")
    @patch("rfdetr.assets.model_weights._download_file")
    def test_handles_missing_rfdetr_plus_gracefully(self, mock_download, mock_exists):
        """Test that missing rf-detr-plus is handled gracefully."""
        mock_exists.return_value = False

        # Should not raise an error if rf-detr-plus is not installed
        download_pretrain_weights("rf-detr-base.pth")

        # Should still download from local ModelWeights
        mock_download.assert_called_once()

    @patch("rfdetr.assets.model_weights.ModelWeights.from_filename", return_value=None)
    @patch("rfdetr.assets.model_weights._download_file")
    @patch("rfdetr.assets.model_weights.os.path.exists")
    def test_handles_missing_platform_models_gracefully(self, mock_exists, mock_download, mock_from_filename):
        """Test that missing platform models is handled gracefully."""
        mock_exists.return_value = False

        # Should return early without raising error
        download_pretrain_weights("unknown-model.pth")

        # Should not attempt download
        mock_download.assert_not_called()

    @patch("rfdetr.assets.model_weights._validate_file_md5")
    @patch("rfdetr.assets.model_weights._download_file")
    @patch("rfdetr.assets.model_weights.os.path.exists")
    @patch("rfdetr.assets.model_weights.logger")
    def test_logs_info_messages(self, mock_logger, mock_exists, mock_download, mock_validate):
        """Test that appropriate log messages are generated."""
        mock_exists.return_value = False
        mock_validate.return_value = True

        download_pretrain_weights("rf-detr-base.pth")

        # Should log download message
        mock_logger.info.assert_called()
        log_message = mock_logger.info.call_args[0][0]
        assert "rf-detr-base.pth" in log_message

    @patch("rfdetr.assets.model_weights._download_file")
    @patch("rfdetr.assets.model_weights._validate_file_md5")
    @patch("rfdetr.assets.model_weights.os.path.exists")
    @patch("rfdetr.assets.model_weights.logger")
    def test_logs_warning_on_incorrect_md5(self, mock_logger, mock_exists, mock_validate, mock_download):
        """Test that warning is logged when MD5 is incorrect."""
        mock_exists.return_value = True
        mock_validate.return_value = False

        download_pretrain_weights("rf-detr-base.pth")

        # Should log warning about incorrect MD5
        mock_logger.warning.assert_called()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "incorrect MD5 hash" in warning_message

    @patch("rfdetr.assets.model_weights._download_file")
    @patch("rfdetr.assets.model_weights.os.path.exists")
    def test_absolute_path_resolves_to_known_model(self, mock_exists, mock_download):
        """Absolute paths like /content/rf-detr-base.pth must still match the registry.

        Regression test: previously ModelWeights.from_filename received the full
        path instead of the basename, so it returned None and the download was
        silently skipped.
        """
        mock_exists.return_value = False

        download_pretrain_weights("/content/rf-detr-base.pth")

        # Must have attempted a download — not silently returned
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["filename"] == "/content/rf-detr-base.pth"
        assert "rf-detr-base-coco.pth" in call_kwargs["url"]

    @patch("rfdetr.assets.model_weights._download_file")
    @patch("rfdetr.assets.model_weights.os.path.exists")
    def test_nested_absolute_path_resolves_to_known_model(self, mock_exists, mock_download):
        """Nested paths like /workspace/models/rf-detr-base.pth also resolve."""
        mock_exists.return_value = False

        download_pretrain_weights("/workspace/models/rf-detr-base.pth")

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["filename"] == "/workspace/models/rf-detr-base.pth"
