# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest

from rfdetr.assets import ModelWeightAsset, ModelWeights, ModelWeightsBase


def test_from_filename_found():
    """Test from_filename with valid filename."""
    asset = ModelWeights.from_filename("rf-detr-base.pth")

    assert asset is not None
    assert isinstance(asset, ModelWeightAsset)
    assert asset.filename == "rf-detr-base.pth"
    assert asset.url.startswith("http")
    assert "rf-detr-base-coco.pth" in asset.url


def test_from_filename_not_found():
    """Test from_filename with invalid filename."""
    asset = ModelWeights.from_filename("nonexistent-model.pth")
    assert asset is None


def test_get_url():
    """Test get_url class method."""
    url = ModelWeights.get_url("rf-detr-base.pth")

    assert url is not None
    assert isinstance(url, str)
    assert url.startswith("http")
    assert "rf-detr-base-coco.pth" in url


def test_get_url_not_found():
    """Test get_url with invalid filename."""
    url = ModelWeights.get_url("nonexistent-model.pth")
    assert url is None


def test_get_md5():
    """Test get_md5 class method."""
    md5 = ModelWeights.get_md5("rf-detr-base.pth")

    # MD5 may be None if not yet computed
    assert md5 is None or isinstance(md5, str)

    # If MD5 exists, verify format
    if md5 is not None:
        assert len(md5) == 32
        assert all(c in "0123456789abcdef" for c in md5.lower())


def test_list_models():
    """Test list_models returns all model filenames."""
    models = ModelWeights.list_models()

    assert isinstance(models, list)
    assert len(models) > 0
    assert "rf-detr-base.pth" in models
    assert "rf-detr-large.pth" in models

    # All entries should be strings
    assert all(isinstance(m, str) for m in models)


@pytest.mark.parametrize("asset", list(ModelWeights), ids=[a.filename for a in ModelWeights])
def test_all_assets_have_valid_urls(asset: ModelWeightAsset) -> None:
    """Test that all assets have valid URLs."""
    assert asset.url.startswith("http")
    assert len(asset.url) > 20  # Reasonable minimum URL length


@pytest.mark.parametrize("asset", list(ModelWeights), ids=[a.filename for a in ModelWeights])
def test_all_assets_have_valid_filenames(asset: ModelWeightAsset) -> None:
    """Test that all assets have valid filenames."""
    assert len(asset.filename) > 0
    assert asset.filename.endswith((".pth", ".pt"))


def test_filenames_are_unique():
    """Test that all filenames are unique (prevent accidental duplicates)."""
    filenames = [asset.filename for asset in ModelWeights]
    assert len(filenames) == len(set(filenames)), "Duplicate filenames detected"


def test_model_weight_asset_optional_md5():
    """Test that MD5 hash is optional (important for new models)."""
    asset = ModelWeightAsset(filename="test-model.pth", url="https://example.com/test-model.pth")

    assert asset.md5_hash is None, "MD5 hash should be optional"


def test_model_weights_inherits_from_base():
    """Test inheritance for compile-time safety contract."""
    assert issubclass(ModelWeights, ModelWeightsBase), (
        "ModelWeights must inherit from ModelWeightsBase for compatibility"
    )
