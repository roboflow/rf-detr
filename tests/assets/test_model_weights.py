# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the pretrained-weights registry and the model cache directory resolution."""

import os
import sys
from pathlib import Path

import pytest

from rfdetr.assets import ModelWeightAsset, ModelWeights, ModelWeightsBase
from rfdetr.assets.model_weights import _log_legacy_cache_hint, get_model_cache_dir

_requires_windows = pytest.mark.skipif(sys.platform != "win32", reason="drive-less rooted paths are Windows-only")

_INVALID_XDG_CACHE_HOMES = [pytest.param("", id="empty"), "relative/cache", "./local_cache", "~/custom_xdg"]


class TestGetModelCacheDir:
    """Verify get_model_cache_dir follows RF_HOME, ROBOFLOW_HOME, XDG_CACHE_HOME, and legacy fallback."""

    def test_default_uses_xdg_cache_when_legacy_dir_absent(self, isolated_home: Path) -> None:
        """Returns ~/.cache/roboflow/models on a clean home directory when no env vars are set."""
        expected = os.path.normpath(str(isolated_home / ".cache" / "roboflow" / "models"))
        assert get_model_cache_dir() == expected

    def test_falls_back_to_existing_legacy_dir_when_xdg_unset(self, isolated_home: Path) -> None:
        """Preserves ~/.roboflow/models when that directory already exists on disk and XDG_CACHE_HOME is unset."""
        legacy_dir = isolated_home / ".roboflow" / "models"
        legacy_dir.mkdir(parents=True)
        assert get_model_cache_dir() == os.path.normpath(str(legacy_dir))

    def test_ignores_legacy_path_when_not_a_directory(self, isolated_home: Path) -> None:
        """A regular file at ~/.roboflow/models does not trigger legacy directory fallback."""
        legacy_parent = isolated_home / ".roboflow"
        legacy_parent.mkdir(parents=True)
        (legacy_parent / "models").write_text("not a dir", encoding="utf-8")
        expected = os.path.normpath(str(isolated_home / ".cache" / "roboflow" / "models"))
        assert get_model_cache_dir() == expected

    def test_xdg_cache_home_absolute_path(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Appends roboflow/models to an absolute XDG_CACHE_HOME."""
        xdg_root = tmp_path / "xdg_cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_root))
        expected = os.path.normpath(str(xdg_root / "roboflow" / "models"))
        assert get_model_cache_dir() == expected

    def test_existing_legacy_dir_wins_while_xdg_target_is_absent(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An existing ~/.roboflow/models is kept while the directory under a valid XDG_CACHE_HOME does not exist."""
        legacy_dir = isolated_home / ".roboflow" / "models"
        legacy_dir.mkdir(parents=True)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
        assert get_model_cache_dir() == os.path.normpath(str(legacy_dir))

    def test_existing_xdg_target_wins_over_legacy_dir(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Once the directory under a valid XDG_CACHE_HOME exists it wins over an existing ~/.roboflow/models."""
        (isolated_home / ".roboflow" / "models").mkdir(parents=True)
        xdg_target = tmp_path / "xdg_cache" / "roboflow" / "models"
        xdg_target.mkdir(parents=True)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
        assert get_model_cache_dir() == os.path.normpath(str(xdg_target))

    def test_existing_default_target_wins_over_legacy_dir(self, isolated_home: Path) -> None:
        """With XDG_CACHE_HOME unset, an existing ~/.cache/roboflow/models wins over ~/.roboflow/models."""
        (isolated_home / ".roboflow" / "models").mkdir(parents=True)
        default_target = isolated_home / ".cache" / "roboflow" / "models"
        default_target.mkdir(parents=True)
        assert get_model_cache_dir() == os.path.normpath(str(default_target))

    @pytest.mark.parametrize("invalid_xdg", _INVALID_XDG_CACHE_HOMES)
    def test_invalid_xdg_cache_home_is_ignored(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, invalid_xdg: str
    ) -> None:
        """Empty, relative or tilde-prefixed XDG_CACHE_HOME values are ignored as the XDG specification requires."""
        monkeypatch.setenv("XDG_CACHE_HOME", invalid_xdg)
        expected = os.path.normpath(str(isolated_home / ".cache" / "roboflow" / "models"))
        assert get_model_cache_dir() == expected

    @pytest.mark.parametrize("invalid_xdg", _INVALID_XDG_CACHE_HOMES)
    def test_invalid_xdg_cache_home_falls_back_to_legacy_dir(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, invalid_xdg: str
    ) -> None:
        """An ignored XDG_CACHE_HOME leaves an existing ~/.roboflow/models in use."""
        legacy_dir = isolated_home / ".roboflow" / "models"
        legacy_dir.mkdir(parents=True)
        monkeypatch.setenv("XDG_CACHE_HOME", invalid_xdg)
        assert get_model_cache_dir() == os.path.normpath(str(legacy_dir))

    @_requires_windows
    def test_drive_less_rooted_xdg_cache_home_is_ignored(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rooted value without a drive letter is not absolute on any Python version, so it is ignored."""
        monkeypatch.setenv("XDG_CACHE_HOME", "/cache")
        expected = os.path.normpath(str(isolated_home / ".cache" / "roboflow" / "models"))
        assert get_model_cache_dir() == expected

    @pytest.mark.parametrize("env_var", ["RF_HOME", "ROBOFLOW_HOME"])
    def test_explicit_home_wins_over_existing_legacy_dir(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, env_var: str
    ) -> None:
        """RF_HOME and ROBOFLOW_HOME stay the exact directory even when ~/.roboflow/models exists."""
        (isolated_home / ".roboflow" / "models").mkdir(parents=True)
        explicit_dir = tmp_path / "explicit_home"
        monkeypatch.setenv(env_var, str(explicit_dir))
        assert get_model_cache_dir() == str(explicit_dir)

    def test_legacy_hint_is_logged_once_per_directory_pair(self, isolated_home: Path, mocker) -> None:
        """The legacy-cache hint is logged a single time however often the directory is resolved."""
        _log_legacy_cache_hint.cache_clear()
        logger = mocker.patch("rfdetr.assets.model_weights.logger")
        (isolated_home / ".roboflow" / "models").mkdir(parents=True)

        get_model_cache_dir()
        get_model_cache_dir()

        logger.info.assert_called_once()

    def test_legacy_hint_names_the_ignored_legacy_dir_when_target_exists(self, isolated_home: Path, mocker) -> None:
        """When the XDG default exists beside a legacy directory, the single hint says the legacy one is ignored."""
        _log_legacy_cache_hint.cache_clear()
        logger = mocker.patch("rfdetr.assets.model_weights.logger")
        (isolated_home / ".roboflow" / "models").mkdir(parents=True)
        (isolated_home / ".cache" / "roboflow" / "models").mkdir(parents=True)

        get_model_cache_dir()
        get_model_cache_dir()

        logger.info.assert_called_once()
        assert "ignored" in logger.info.call_args.args[0]

    def test_no_legacy_hint_without_legacy_dir(self, isolated_home: Path, mocker) -> None:
        """Nothing is logged when ~/.roboflow/models does not exist."""
        _log_legacy_cache_hint.cache_clear()
        logger = mocker.patch("rfdetr.assets.model_weights.logger")

        get_model_cache_dir()

        logger.info.assert_not_called()

    def test_custom_rf_home_absolute_path(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Returns exact RF_HOME value when set to an absolute path."""
        rf_home = tmp_path / "rf_home"
        monkeypatch.setenv("RF_HOME", str(rf_home))
        assert get_model_cache_dir() == str(rf_home)

    def test_tilde_in_rf_home_is_expanded(self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tilde in RF_HOME is expanded; result contains no literal tilde."""
        monkeypatch.setenv("RF_HOME", "~/custom_rfdetr_cache")
        result = get_model_cache_dir()
        assert result == os.path.normpath(str(isolated_home / "custom_rfdetr_cache"))
        assert "~" not in result

    def test_returns_string(self, isolated_home: Path) -> None:
        """Return value is always a str, not a Path."""
        assert isinstance(get_model_cache_dir(), str)

    def test_roboflow_home_alias_used_when_rf_home_absent(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """ROBOFLOW_HOME acts as an alias for RF_HOME and overrides XDG_CACHE_HOME."""
        roboflow_home = tmp_path / "roboflow_home"
        monkeypatch.setenv("ROBOFLOW_HOME", str(roboflow_home))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
        assert get_model_cache_dir() == str(roboflow_home)

    def test_rf_home_wins_when_all_set(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """RF_HOME is canonical and takes precedence over ROBOFLOW_HOME and XDG_CACHE_HOME."""
        rf_home = tmp_path / "rf_home"
        monkeypatch.setenv("RF_HOME", str(rf_home))
        monkeypatch.setenv("ROBOFLOW_HOME", str(tmp_path / "roboflow_home"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
        assert get_model_cache_dir() == str(rf_home)

    def test_empty_rf_home_falls_through_to_roboflow_home(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An empty RF_HOME is treated as unset and falls through to ROBOFLOW_HOME."""
        roboflow_home = tmp_path / "roboflow_home"
        monkeypatch.setenv("RF_HOME", "")
        monkeypatch.setenv("ROBOFLOW_HOME", str(roboflow_home))
        assert get_model_cache_dir() == str(roboflow_home)

    def test_empty_roboflow_home_falls_through_to_xdg_default(
        self, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty ROBOFLOW_HOME is treated as unset and falls back to the XDG default."""
        monkeypatch.setenv("ROBOFLOW_HOME", "")
        expected = os.path.normpath(str(isolated_home / ".cache" / "roboflow" / "models"))
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
