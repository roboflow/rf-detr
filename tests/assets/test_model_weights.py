# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the pretrained-weights registry and the model cache directory resolution."""

import os

import pytest

from rfdetr.assets import ModelWeightAsset, ModelWeights, ModelWeightsBase
from rfdetr.assets.model_weights import get_model_cache_dir


class TestGetModelCacheDir:
    """Verify get_model_cache_dir reads RF_HOME with correct default."""

    def test_default_when_rf_home_not_set(self, monkeypatch):
        """Returns ~/.roboflow/models when RF_HOME env var is absent."""
        monkeypatch.delenv("RF_HOME", raising=False)
        monkeypatch.delenv("ROBOFLOW_HOME", raising=False)
        expected = os.path.normpath(os.path.expanduser("~/.roboflow/models"))
        assert get_model_cache_dir() == expected

    def test_custom_rf_home_absolute_path(self, monkeypatch, tmp_path):
        """Returns exact RF_HOME value when set to an absolute path."""
        monkeypatch.setenv("RF_HOME", str(tmp_path))
        assert get_model_cache_dir() == str(tmp_path)

    def test_tilde_in_rf_home_is_expanded(self, monkeypatch):
        """Tilde in RF_HOME is expanded; result contains no literal tilde."""
        monkeypatch.setenv("RF_HOME", "~/custom_rfdetr_cache")
        result = get_model_cache_dir()
        assert result == os.path.normpath(os.path.expanduser("~/custom_rfdetr_cache"))
        assert "~" not in result

    def test_returns_string(self, monkeypatch):
        """Return value is always a str, not a Path."""
        monkeypatch.delenv("RF_HOME", raising=False)
        monkeypatch.delenv("ROBOFLOW_HOME", raising=False)
        assert isinstance(get_model_cache_dir(), str)

    def test_roboflow_home_alias_used_when_rf_home_absent(self, monkeypatch, tmp_path):
        """ROBOFLOW_HOME acts as an alias for RF_HOME when RF_HOME is unset."""
        monkeypatch.delenv("RF_HOME", raising=False)
        monkeypatch.setenv("ROBOFLOW_HOME", str(tmp_path))
        assert get_model_cache_dir() == str(tmp_path)

    def test_rf_home_wins_when_both_set(self, monkeypatch, tmp_path):
        """RF_HOME is canonical and takes precedence over ROBOFLOW_HOME when both are set."""
        rf_home = tmp_path / "rf_home"
        monkeypatch.setenv("RF_HOME", str(rf_home))
        monkeypatch.setenv("ROBOFLOW_HOME", str(tmp_path / "roboflow_home"))
        assert get_model_cache_dir() == str(rf_home)

    def test_empty_rf_home_falls_through_to_roboflow_home(self, monkeypatch, tmp_path):
        """An empty RF_HOME is treated as unset and falls through to ROBOFLOW_HOME."""
        monkeypatch.setenv("RF_HOME", "")
        monkeypatch.setenv("ROBOFLOW_HOME", str(tmp_path))
        assert get_model_cache_dir() == str(tmp_path)

    def test_empty_roboflow_home_falls_through_to_default(self, monkeypatch):
        """An empty ROBOFLOW_HOME is treated as unset and falls back to the default."""
        monkeypatch.delenv("RF_HOME", raising=False)
        monkeypatch.setenv("ROBOFLOW_HOME", "")
        expected = os.path.normpath(os.path.expanduser("~/.roboflow/models"))
        assert get_model_cache_dir() == expected


def test_from_filename_found():
    """from_filename returns the registered asset, with its filename and a download URL, for a known filename."""
    asset = ModelWeights.from_filename("rf-detr-base.pth")

    assert asset is not None
    assert isinstance(asset, ModelWeightAsset)
    assert asset.filename == "rf-detr-base.pth"
    assert asset.url.startswith("http")
    assert "rf-detr-base-coco.pth" in asset.url


def test_from_filename_not_found():
    """from_filename returns None for a filename that is not registered."""
    asset = ModelWeights.from_filename("nonexistent-model.pth")
    assert asset is None


def test_get_url():
    """get_url returns the download URL string for a known filename."""
    url = ModelWeights.get_url("rf-detr-base.pth")

    assert url is not None
    assert isinstance(url, str)
    assert url.startswith("http")
    assert "rf-detr-base-coco.pth" in url


def test_get_url_not_found():
    """get_url returns None for a filename that is not registered."""
    url = ModelWeights.get_url("nonexistent-model.pth")
    assert url is None


def test_get_md5():
    """get_md5 returns either None (hash not yet computed) or a 32-character lowercase-hex digest string."""
    md5 = ModelWeights.get_md5("rf-detr-base.pth")

    assert md5 is None or (len(md5) == 32 and set(md5.lower()) <= set("0123456789abcdef"))


def test_list_models():
    """list_models returns a non-empty list of filename strings that includes the base and large checkpoints."""
    models = ModelWeights.list_models()

    assert isinstance(models, list)
    assert len(models) > 0
    assert "rf-detr-base.pth" in models
    assert "rf-detr-large.pth" in models

    # All entries should be strings
    assert all(isinstance(m, str) for m in models)


@pytest.mark.parametrize("asset", [pytest.param(a, id=a.filename) for a in ModelWeights])
def test_all_assets_have_valid_urls(asset: ModelWeightAsset) -> None:
    """Every registered asset has an http(s) URL of plausible length."""
    assert asset.url.startswith("http")
    assert len(asset.url) > 20  # Reasonable minimum URL length


@pytest.mark.parametrize("asset", [pytest.param(a, id=a.filename) for a in ModelWeights])
def test_all_assets_have_valid_filenames(asset: ModelWeightAsset) -> None:
    """Every registered asset has a non-empty filename ending in a checkpoint extension (.pth or .pt)."""
    assert len(asset.filename) > 0
    assert asset.filename.endswith((".pth", ".pt"))


def test_filenames_are_unique():
    """No two registered assets share a filename, which would shadow one of them in lookups."""
    filenames = [asset.filename for asset in ModelWeights]
    assert len(filenames) == len(set(filenames)), "Duplicate filenames detected"


def test_model_weight_asset_optional_md5():
    """ModelWeightAsset can be built without an MD5 hash, which defaults to None (new models have none yet)."""
    asset = ModelWeightAsset(filename="test-model.pth", url="https://example.com/test-model.pth")

    assert asset.md5_hash is None, "MD5 hash should be optional"


def test_model_weights_inherits_from_base():
    """ModelWeights subclasses ModelWeightsBase, the compile-time safety contract for the registry."""
    assert issubclass(ModelWeights, ModelWeightsBase), (
        "ModelWeights must inherit from ModelWeightsBase for compatibility"
    )


def test_rfdetr_large_deprecated_emits_future_warning(mocker) -> None:
    """RFDETRLargeDeprecated emits FutureWarning on instantiation via the pyDeprecate class decorator."""
    from rfdetr.variants import RFDETRLargeDeprecated

    mocker.patch("rfdetr.detr.RFDETR.__init__", return_value=None)
    # The proxy uses num_warns=1; reset counter so this test is order-independent
    # regardless of whether another test already triggered the warning in this session.
    RFDETRLargeDeprecated._cfg.warned = 0
    with pytest.warns(FutureWarning):
        RFDETRLargeDeprecated()
