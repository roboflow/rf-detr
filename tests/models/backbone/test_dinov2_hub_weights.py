# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for loading DINOv2 hub weights without ``transformers``: pinned revision, strict load, install hints."""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import torch

from rfdetr.models.backbone.dinov2 import _load_dinov2_hub_weights
from tests._markers import _requires_huggingface_hub, _requires_safetensors
from tests.models.backbone._dinov2_goldens import tiny_backbone
from tests.models.backbone.test_dinov2_characterization import _DINOV2_SMALL_SHA, _run_child

#: Child process that loads the tiny backbone's weights from a fake offline cache through the hub loader.
#: ``huggingface_hub`` reads ``HF_HUB_CACHE`` and ``HF_HUB_OFFLINE`` when it is imported, so only a fresh process sees
#: the fake cache.
_OFFLINE_CHILD = textwrap.dedent(
    """
    import torch

    from rfdetr.models.backbone.dinov2 import _load_dinov2_hub_weights
    from rfdetr.models.backbone.dinov2_with_windowed_attn import (
        WindowedDinov2WithRegistersBackbone,
        WindowedDinov2WithRegistersConfig,
    )

    config = WindowedDinov2WithRegistersConfig(
        hidden_size=8, num_hidden_layers=2, num_attention_heads=2, image_size=16, patch_size=4, num_register_tokens=0
    )
    with torch.device("meta"):
        backbone = WindowedDinov2WithRegistersBackbone(config)
    _load_dinov2_hub_weights(backbone, "facebook/dinov2-small")
    print("loaded")
    """
)


def _save_weights(state_dict: dict[str, torch.Tensor], path: Path) -> Path:
    """Write a state dict as a safetensors file, the format of the DINOv2 hub checkpoints.

    Args:
        state_dict: Tensors to write.
        path: Target file.

    Returns:
        ``path``.

    Examples:
        >>> import tempfile
        >>> target = Path(tempfile.mkdtemp()) / "model.safetensors"
        >>> _save_weights({"w": torch.zeros(1)}, target).name
        'model.safetensors'
    """
    from safetensors.torch import save_file  # optional dependency, gated by the callers' markers

    save_file({key: value.contiguous() for key, value in state_dict.items()}, str(path), metadata={"format": "pt"})
    return path


__doctest_requires__ = {"_save_weights": ["safetensors"]}


@pytest.fixture
def tiny_weights() -> dict[str, torch.Tensor]:
    """State dict of a randomly initialized tiny backbone, standing in for a hub checkpoint.

    Examples:
        A pytest fixture cannot be called directly.

        >>> tiny_weights()  # doctest: +SKIP
    """
    return {key: value.clone() for key, value in tiny_backbone().state_dict().items()}


@pytest.fixture
def meta_backbone() -> Any:
    """Tiny backbone built on the meta device, the way the hub path builds it before loading.

    Examples:
        A pytest fixture cannot be called directly.

        >>> meta_backbone()  # doctest: +SKIP
    """
    with torch.device("meta"):
        return tiny_backbone()


@_requires_huggingface_hub
@_requires_safetensors
class TestLoadDinov2HubWeights:
    """The loader downloads ``model.safetensors`` at a pinned revision and loads it strictly into the backbone."""

    def test_fills_every_parameter(
        self, mocker: Any, tmp_path: Path, tiny_weights: dict[str, torch.Tensor], meta_backbone: Any
    ) -> None:
        """Every parameter of a meta-device backbone takes the file's value.

        The hub path builds the backbone on the meta device, so a parameter the file does not fill would stay without
        storage; the loaded state dict must equal the file exactly.
        """
        path = _save_weights(tiny_weights, tmp_path / "model.safetensors")
        mocker.patch("huggingface_hub.hf_hub_download", autospec=True, return_value=str(path))

        _load_dinov2_hub_weights(meta_backbone, "facebook/dinov2-small")

        torch.testing.assert_close(meta_backbone.state_dict(), tiny_weights, rtol=0.0, atol=0.0)

    def test_downloads_the_pinned_revision(
        self, mocker: Any, tmp_path: Path, tiny_weights: dict[str, torch.Tensor], meta_backbone: Any
    ) -> None:
        """The download asks for ``model.safetensors`` at the commit whose keys match the backbone.

        Pinning the revision keeps a later hub update from changing the weights, and it is the commit a cache filled
        from today's ``main`` already holds, so offline users keep hitting their cache.
        """
        path = _save_weights(tiny_weights, tmp_path / "model.safetensors")
        download = mocker.patch("huggingface_hub.hf_hub_download", autospec=True, return_value=str(path))

        _load_dinov2_hub_weights(meta_backbone, "facebook/dinov2-small")

        download.assert_called_once_with("facebook/dinov2-small", "model.safetensors", revision=_DINOV2_SMALL_SHA)

    @pytest.mark.parametrize(
        ("edit", "message"),
        [
            pytest.param(lambda weights: weights.pop("layernorm.bias"), "Missing key", id="missing-key"),
            pytest.param(
                lambda weights: weights.update({"extra.weight": torch.zeros(1)}), "Unexpected key", id="unexpected-key"
            ),
        ],
    )
    def test_key_mismatch_raises(
        self,
        mocker: Any,
        tmp_path: Path,
        tiny_weights: dict[str, torch.Tensor],
        meta_backbone: Any,
        edit: Any,
        message: str,
    ) -> None:
        """A file whose keys differ from the backbone's raises instead of leaving parameters uninitialized.

        Hugging Face warned and randomly initialized missing keys; every supported hub checkpoint matches the backbone
        exactly, so a mismatch now means a wrong file and must fail.
        """
        edit(tiny_weights)
        path = _save_weights(tiny_weights, tmp_path / "model.safetensors")
        mocker.patch("huggingface_hub.hf_hub_download", autospec=True, return_value=str(path))

        with pytest.raises(RuntimeError, match=message):
            _load_dinov2_hub_weights(meta_backbone, "facebook/dinov2-small")


class TestLoadDinov2HubWeightsMissingPackage:
    """Without ``huggingface_hub`` or ``safetensors`` the loader names what triggered it and how to install them."""

    @pytest.mark.parametrize("module", ["huggingface_hub", "safetensors.torch"])
    def test_raises_install_hint(self, monkeypatch: pytest.MonkeyPatch, meta_backbone: Any, module: str) -> None:
        """A blocked package raises ``ImportError`` with the install command.

        A ``None`` entry in ``sys.modules`` makes the import fail even though the test session has the package, the
        state a core-only install is in.
        """
        monkeypatch.setitem(sys.modules, module, None)

        with pytest.raises(ImportError, match=r"pip install huggingface_hub safetensors"):
            _load_dinov2_hub_weights(meta_backbone, "facebook/dinov2-small")


@_requires_huggingface_hub
@_requires_safetensors
class TestLoadDinov2HubWeightsOffline:
    """With ``HF_HUB_OFFLINE=1`` the loader reads the pinned snapshot from the cache, or fails clearly without it."""

    def test_cache_at_pinned_revision_loads(self, tmp_path: Path, tiny_weights: dict[str, torch.Tensor]) -> None:
        """A cache holding only the pinned snapshot, with no ``refs/main``, loads offline.

        This proves the loader asks for the pinned commit: a request for ``main`` would need the ``refs/main`` pointer.
        """
        snapshot = tmp_path / "models--facebook--dinov2-small" / "snapshots" / _DINOV2_SMALL_SHA
        snapshot.mkdir(parents=True)
        _save_weights(tiny_weights, snapshot / "model.safetensors")
        env = {**os.environ, "HF_HUB_CACHE": str(tmp_path), "HF_HUB_OFFLINE": "1"}

        result = _run_child(_OFFLINE_CHILD, env=env)

        assert result.returncode == 0, result.stderr[-4000:]

    def test_missing_cache_raises(self, tmp_path: Path) -> None:
        """An empty cache offline fails with an error that names the offline mode, not a later shape or key error."""
        env = {**os.environ, "HF_HUB_CACHE": str(tmp_path), "HF_HUB_OFFLINE": "1"}

        result = _run_child(_OFFLINE_CHILD, env=env)

        assert result.returncode != 0
        assert "HF_HUB_OFFLINE" in result.stderr, result.stderr[-4000:]
