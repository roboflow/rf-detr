# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for pretrained-weight downloading: registry lookup, MD5 handling, and RF_HOME cache resolution."""

from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from rfdetr.assets import ModelWeightAsset, ModelWeights
from rfdetr.assets.model_weights import download_pretrain_weights
from rfdetr.platform import _IS_RFDETR_PLUS_AVAILABLE


@pytest.fixture
def mock_file_operations(mocker):
    """Mock file operations to avoid actual file I/O.

    Defaults to a missing file whose MD5 validation would pass.
    """
    mock_exists = mocker.patch("rfdetr.assets.model_weights.os.path.exists", return_value=False)
    mock_download = mocker.patch("rfdetr.assets.model_weights._download_file")
    mock_validate = mocker.patch("rfdetr.assets.model_weights._validate_file_md5", return_value=True)
    return {"exists": mock_exists, "download": mock_download, "validate": mock_validate}


class TestDownloadPretrainWeights:
    """Test download_pretrain_weights function with mocking for offline testing."""

    def test_download_from_local_model_weights(self, mock_file_operations):
        """A model registered in ModelWeights is downloaded with its registry URL and a 32-character MD5."""
        download_pretrain_weights("rf-detr-base.pth")

        # Should call download with correct URL and MD5
        mock_file_operations["download"].assert_called_once()
        call_kwargs = mock_file_operations["download"].call_args[1]

        assert call_kwargs["filename"] == "rf-detr-base.pth"
        assert "rf-detr-base-coco.pth" in call_kwargs["url"]
        assert call_kwargs["expected_md5"] is not None  # Should have MD5 hash
        assert len(call_kwargs["expected_md5"]) == 32  # Valid MD5 hash

    @pytest.mark.skipif(not _IS_RFDETR_PLUS_AVAILABLE, reason="rf-detr-plus not installed - skip priority test")
    def test_download_from_rfdetr_plus_when_available(self, mocker, mock_file_operations):
        """The rf-detr-plus URL is used for models absent from local ModelWeights.

        Verifies the priority contract: a model not registered in local ModelWeights but
        present in rf-detr-plus triggers _download_file with the plus URL, not a fallback
        or silent no-op.
        """
        plus_url = "https://plus.example.com/plus-only-model.pth"
        plus_asset = ModelWeightAsset(filename="plus-only-model.pth", url=plus_url, md5_hash=None)
        mocker.patch("rfdetr.assets.model_weights.ModelWeights.from_filename", return_value=None)
        mocker.patch("rfdetr_plus.assets.ModelWeights.from_filename", return_value=plus_asset)

        download_pretrain_weights("plus-only-model.pth")

        mock_file_operations["download"].assert_called_once()
        call_kwargs = mock_file_operations["download"].call_args[1]
        assert call_kwargs["url"] == plus_url

    def test_download_from_platform_models_fallback(self, mocker, mock_file_operations):
        """A model missing from ModelWeights falls back to PLATFORM_MODELS, with no MD5 to validate."""
        mock_platform_module = Mock()
        mock_platform_module.PLATFORM_MODELS = {"legacy-model.pth": "https://legacy.com/model.pth"}
        mocker.patch("rfdetr.assets.model_weights.ModelWeights.from_filename", return_value=None)
        mocker.patch.dict("sys.modules", {"rfdetr.platform": Mock(), "rfdetr.platform.downloads": mock_platform_module})

        download_pretrain_weights("legacy-model.pth")

        # Should call download with legacy URL
        mock_file_operations["download"].assert_called_once()
        call_kwargs = mock_file_operations["download"].call_args[1]
        assert call_kwargs["url"] == "https://legacy.com/model.pth"
        assert call_kwargs["expected_md5"] is None  # Platform models don't have MD5

    def test_private_platform_models_registry_imports_without_eager_fallback(self):
        """A private-only registry imports cleanly without evaluating PLATFORM_MODELS early."""
        from importlib import reload

        import rfdetr.platform as platform_pkg
        import rfdetr.platform.downloads as platform_downloads

        mock_rfdetr_plus_pkg = ModuleType("rfdetr_plus")
        mock_rfdetr_plus_pkg.__path__ = []
        mock_platforms_pkg = ModuleType("rfdetr_plus.models")
        mock_platforms_pkg.__path__ = []
        mock_downloads_module = ModuleType("rfdetr_plus.models.downloads")
        mock_downloads_module._PLATFORM_MODELS = {"legacy-model.pth": "https://legacy.com/model.pth"}
        mock_platforms_pkg.downloads = mock_downloads_module
        mock_rfdetr_plus_pkg.models = mock_platforms_pkg

        # Patches must end before the final reload so the real module state is restored.
        with (
            patch.dict(
                "sys.modules",
                {
                    "rfdetr_plus": mock_rfdetr_plus_pkg,
                    "rfdetr_plus.models": mock_platforms_pkg,
                    "rfdetr_plus.models.downloads": mock_downloads_module,
                },
            ),
            patch.object(platform_pkg, "_IS_RFDETR_PLUS_AVAILABLE", True),
        ):
            reloaded = reload(platform_downloads)
            assert reloaded.PLATFORM_MODELS == {"legacy-model.pth": "https://legacy.com/model.pth"}

        reload(platform_downloads)

    def test_file_exists_with_correct_md5(self, mock_file_operations):
        """An existing file with a matching MD5 is not downloaded again."""
        mock_file_operations["exists"].return_value = True
        mock_file_operations["validate"].return_value = True

        download_pretrain_weights("rf-detr-base.pth")

        # Should not download if file exists with correct hash
        mock_file_operations["download"].assert_not_called()

    def test_file_exists_with_incorrect_md5_warns_and_skips(self, mocker, mock_file_operations):
        """A file with a wrong MD5 is kept and a warning is logged when redownload=False.

        This protects fine-tuned checkpoints that share the same filename as a registry model (e.g. rf-detr-nano.pth)
        from being silently overwritten.
        """
        mock_warning = mocker.patch("rfdetr.assets.model_weights.logger.warning")
        mock_file_operations["exists"].return_value = True
        mock_file_operations["validate"].return_value = False  # Incorrect MD5

        download_pretrain_weights("rf-detr-base.pth")

        # Should NOT re-download — the user's file must be preserved
        mock_file_operations["download"].assert_not_called()
        mock_warning.assert_called_once()
        warning_msg = mock_warning.call_args[0][0]
        assert "incorrect MD5 hash" in warning_msg
        assert "skipping re-download to avoid overwriting it" in warning_msg

    @pytest.mark.parametrize("md5_valid", [True, False])
    def test_redownload_flag_forces_download(self, mock_file_operations, md5_valid):
        """Redownload=True downloads even when the file exists, whether or not its MD5 matches.

        With a wrong MD5 this is the force path where the user explicitly overwrites an existing file (e.g. a fine-tuned
        checkpoint) with the original registry weights, overriding the skip-on-existing-file guard.
        """
        mock_file_operations["exists"].return_value = True
        mock_file_operations["validate"].return_value = md5_valid

        download_pretrain_weights("rf-detr-base.pth", redownload=True)

        mock_file_operations["download"].assert_called_once()

    def test_validate_md5_disabled(self, mock_file_operations):
        """validate_md5=False passes expected_md5=None to the download."""
        download_pretrain_weights("rf-detr-base.pth", validate_md5=False)

        call_kwargs = mock_file_operations["download"].call_args[1]
        assert call_kwargs["expected_md5"] is None

    @pytest.mark.parametrize("filename", ["nonexistent-model.pth", "unknown-model.pth"])
    def test_unregistered_model_returns_early_without_download(self, mocker, mock_file_operations, filename):
        """A model absent from every registry returns early without raising or attempting a download."""
        mocker.patch("rfdetr.assets.model_weights.ModelWeights.from_filename", return_value=None)

        download_pretrain_weights(filename)

        mock_file_operations["download"].assert_not_called()

    def test_model_without_md5_hash(self, mocker, mock_file_operations):
        """A registry asset with no MD5 hash is downloaded with expected_md5=None even when validation is requested."""
        mock_asset = ModelWeightAsset(filename="test-no-md5.pth", url="https://example.com/test.pth", md5_hash=None)
        mocker.patch("rfdetr.assets.model_weights.ModelWeights.from_filename", return_value=mock_asset)

        download_pretrain_weights("test-no-md5.pth", validate_md5=True)

        call_kwargs = mock_file_operations["download"].call_args[1]
        assert call_kwargs["expected_md5"] is None

    def test_file_exists_no_md5_skips_download(self, mocker, mock_file_operations):
        """An existing file whose registry asset has no MD5 to validate is not downloaded again."""
        mock_file_operations["exists"].return_value = True
        mock_asset = ModelWeightAsset(filename="test-no-md5.pth", url="https://example.com/test.pth", md5_hash=None)
        mocker.patch("rfdetr.assets.model_weights.ModelWeights.from_filename", return_value=mock_asset)

        download_pretrain_weights("test-no-md5.pth")

        mock_file_operations["download"].assert_not_called()


class TestDownloadIntegration:
    """Integration tests for the complete download flow."""

    @pytest.mark.parametrize("model", [pytest.param(m, id=m.filename) for m in ModelWeights])
    def test_all_models_have_valid_md5_format(self, model: ModelWeightAsset) -> None:
        """Each registry MD5 is None or a 32-character hex string (prevents typos)."""
        md5 = model.md5_hash
        assert md5 is None or (len(md5) == 32 and set(md5.lower()) <= set("0123456789abcdef")), (
            f"{model.filename} has invalid MD5: {md5!r}"
        )

    @pytest.mark.parametrize("model", [pytest.param(m, id=m.filename) for m in ModelWeights])
    def test_all_models_download_from_weights_cdn(self, model: ModelWeightAsset) -> None:
        """Every built-in weight file is served from the RF-DETR weights CDN host."""
        assert model.url.startswith("https://repo.roboflow.com/rfdetr/"), model.url

    def test_from_filename_bidirectional_lookup(self):
        """from_filename maps a registered filename back to the identical enum asset."""
        original = ModelWeights.RF_DETR_BASE

        asset = ModelWeights.from_filename(original.filename)

        assert asset is not None
        assert asset.filename == original.filename
        assert asset.url == original.url
        assert asset.md5_hash == original.md5_hash

    def test_download_flow_for_real_model(self, mock_file_operations):
        """A registered model downloads from its exact CDN URL with its pinned MD5 (network mocked)."""
        download_pretrain_weights("rf-detr-base.pth")

        mock_file_operations["download"].assert_called_once()
        call_kwargs = mock_file_operations["download"].call_args[1]

        assert call_kwargs["filename"] == "rf-detr-base.pth"
        assert call_kwargs["url"] == "https://repo.roboflow.com/rfdetr/rf-detr-base-coco.pth"
        assert call_kwargs["expected_md5"] == "b4d3ce46099eaed50626ede388caf979"


class TestDownloadErrorHandling:
    """Test error handling in download mechanism."""

    def test_handles_missing_rfdetr_plus_gracefully(self, mock_file_operations):
        """A registered model still downloads from local ModelWeights when rf-detr-plus is not installed."""
        download_pretrain_weights("rf-detr-base.pth")

        mock_file_operations["download"].assert_called_once()

    def test_logs_info_messages(self, mocker, mock_file_operations):
        """Downloading a model logs an info message that names the file."""
        mock_logger = mocker.patch("rfdetr.assets.model_weights.logger")

        download_pretrain_weights("rf-detr-base.pth")

        mock_logger.info.assert_called()
        log_message = mock_logger.info.call_args[0][0]
        assert "rf-detr-base.pth" in log_message

    def test_logs_warning_on_incorrect_md5(self, mocker, mock_file_operations):
        """A wrong MD5 on an existing file logs a warning and does not re-download, preserving fine-tuned
        checkpoints."""
        mock_logger = mocker.patch("rfdetr.assets.model_weights.logger")
        mock_file_operations["exists"].return_value = True
        mock_file_operations["validate"].return_value = False

        download_pretrain_weights("rf-detr-base.pth")

        mock_logger.warning.assert_called()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "incorrect MD5 hash" in warning_message
        mock_file_operations["download"].assert_not_called()

    @pytest.mark.parametrize("path", ["/content/rf-detr-base.pth", "/workspace/models/rf-detr-base.pth"])
    def test_absolute_path_resolves_to_known_model(self, mock_file_operations, path):
        """Absolute paths still match the registry by basename and are downloaded under the full path.

        Regression test: previously ModelWeights.from_filename received the full path instead of the basename, so it
        returned None and the download was silently skipped.
        """
        download_pretrain_weights(path)

        # Must have attempted a download — not silently returned
        mock_file_operations["download"].assert_called_once()
        call_kwargs = mock_file_operations["download"].call_args[1]
        assert call_kwargs["filename"] == path
        assert "rf-detr-base-coco.pth" in call_kwargs["url"]


# ---------------------------------------------------------------------------
# maybe_download_pretrain_weights — RF_HOME cache-dir path resolution
# ---------------------------------------------------------------------------


class TestMaybeDownloadPretrainWeightsCacheDir:
    """Verify that RFDETR.maybe_download_pretrain_weights resolves paths via RF_HOME."""

    def _make_rfdetr(self, pretrain_weights):
        """Return an RFDETR shell backed by a fully validated RFDETRBaseConfig.

        Uses RFDETRBaseConfig (which supplies required field defaults) so the expand_path field validator on
        pretrain_weights is exercised end-to-end.

        Args:
            pretrain_weights: Raw value to pass to RFDETRBaseConfig; the pydantic
                validator resolves it before assigning to model_config.pretrain_weights.
        """
        from rfdetr.config import RFDETRBaseConfig
        from rfdetr.detr import RFDETR

        model = object.__new__(RFDETR)
        model.model_config = RFDETRBaseConfig(pretrain_weights=pretrain_weights)
        return model

    def test_bare_filename_resolved_to_rf_home(self, monkeypatch, tmp_path):
        """Bare filename (no directory separator) is joined with RF_HOME before download."""
        monkeypatch.setenv("RF_HOME", str(tmp_path))
        downloaded = []
        monkeypatch.setattr("rfdetr.detr.download_pretrain_weights", lambda p, **kw: downloaded.append(p))

        self._make_rfdetr("rf-detr-base.pth").maybe_download_pretrain_weights()

        assert downloaded == [str(tmp_path / "rf-detr-base.pth")]

    def test_path_with_directory_used_as_is(self, monkeypatch, tmp_path):
        """Path containing a directory separator is not modified by RF_HOME."""
        monkeypatch.setenv("RF_HOME", str(tmp_path / "should_not_be_used"))
        downloaded = []
        monkeypatch.setattr("rfdetr.detr.download_pretrain_weights", lambda p, **kw: downloaded.append(p))

        explicit = str(tmp_path / "custom" / "my_weights.pth")
        self._make_rfdetr(explicit).maybe_download_pretrain_weights()

        assert downloaded == [explicit]

    def test_none_pretrain_weights_skips_download(self, monkeypatch):
        """None pretrain_weights returns without calling download."""
        called = []
        monkeypatch.setattr("rfdetr.detr.download_pretrain_weights", lambda *a, **kw: called.append(True))

        self._make_rfdetr(None).maybe_download_pretrain_weights()

        assert called == []

    def test_cache_dir_created_when_absent(self, monkeypatch, tmp_path):
        """RF_HOME directory is created if it does not already exist."""
        cache_dir = tmp_path / "new_cache"
        monkeypatch.setenv("RF_HOME", str(cache_dir))
        monkeypatch.setattr("rfdetr.detr.download_pretrain_weights", lambda *a, **kw: None)

        self._make_rfdetr("rf-detr-base.pth").maybe_download_pretrain_weights()

        assert cache_dir.is_dir()
