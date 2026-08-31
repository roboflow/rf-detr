# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr.models.position_encoding.build_position_encoding.

Covers:
- Supported aliases ``"sine"`` / ``"v2"`` return a ``PositionEmbeddingSine`` instance.
- Unsupported but previously accepted aliases ``"learned"`` / ``"v3"`` now raise
  ``ValueError`` with a message that names supported alternatives.
- Fully unsupported values raise ``ValueError`` with the same pattern.
- ``PositionEmbeddingSine`` caching for batches flagged ``no_padding``.
"""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.models.position_encoding import (
    _POS_CACHE_MAX_ENTRIES,
    PositionEmbeddingSine,
    build_position_encoding,
)
from rfdetr.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list


class TestBuildPositionEncodingSupportedValues:
    """build_position_encoding returns valid modules for supported aliases."""

    @pytest.mark.parametrize(
        "alias",
        [
            pytest.param("sine", id="sine"),
            pytest.param("v2", id="v2"),
        ],
    )
    def test_returns_sine_embedding(self, alias: str) -> None:
        """Supported aliases produce a PositionEmbeddingSine with normalized=True."""
        enc = build_position_encoding(hidden_dim=256, position_embedding=alias)
        assert isinstance(enc, PositionEmbeddingSine)
        assert enc.normalize is True

    @pytest.mark.parametrize(
        "hidden_dim, expected_num_pos_feats",
        [
            pytest.param(256, 128, id="dim256"),
            pytest.param(512, 256, id="dim512"),
        ],
    )
    def test_num_pos_feats_is_half_hidden_dim(self, hidden_dim: int, expected_num_pos_feats: int) -> None:
        """The sine encoding uses hidden_dim // 2 positional feature dimensions."""
        enc = build_position_encoding(hidden_dim=hidden_dim, position_embedding="sine")
        assert enc.num_pos_feats == expected_num_pos_feats


class TestBuildPositionEncodingUnsupportedValues:
    """build_position_encoding raises ValueError for broken or unknown aliases."""

    @pytest.mark.parametrize(
        "alias",
        [
            pytest.param("learned", id="learned"),
            pytest.param("v3", id="v3"),
        ],
    )
    def test_learned_raises_value_error(self, alias: str) -> None:
        """'learned' and 'v3' are doubly broken and must raise ValueError immediately.

        The PositionEmbeddingLearned class has two bugs:
        1. forward() signature is incompatible with Joiner.forward() (no align_dim_orders param).
        2. h, w = x.shape[:2] unpacks batch and channels instead of height and width.
        Rejecting them at build time is preferable to a silent or confusing runtime failure.
        """
        with pytest.raises(ValueError, match="not supported"):
            build_position_encoding(hidden_dim=256, position_embedding=alias)

    def test_unknown_value_raises_value_error(self) -> None:
        """A fully unknown alias raises ValueError naming the supported alternatives."""
        with pytest.raises(ValueError, match="not supported"):
            build_position_encoding(hidden_dim=256, position_embedding="unknown_variant")

    @pytest.mark.parametrize(
        "alias",
        [
            pytest.param("learned", id="learned"),
            pytest.param("v3", id="v3"),
        ],
    )
    def test_error_message_mentions_supported_alternatives(self, alias: str) -> None:
        """Error message for 'learned'/'v3' mentions at least one supported alternative."""
        with pytest.raises(ValueError, match="sine"):
            build_position_encoding(hidden_dim=256, position_embedding=alias)


class TestPositionEmbeddingSineNoPaddingCache:
    """The embedding is cached only for unpadded batches, and only where that is exact.

    ``PositionEmbeddingSine`` holds no parameters, so for an all-False mask the embedding is a pure function of the
    mask's shape and device and the cached value can never go stale.  The cache must therefore reproduce the recomputed
    embedding bit for bit, stay empty for batches that carry real padding, and not grow without bound.
    """

    @staticmethod
    def _nested(height: int, width: int, batch: int = 1, no_padding: bool = True) -> NestedTensor:
        """Return an unpadded ``NestedTensor`` of the given spatial size.

        Examples:
            >>> TestPositionEmbeddingSineNoPaddingCache._nested(4, 6).mask.shape
            torch.Size([1, 4, 6])
        """
        return NestedTensor(
            torch.rand(batch, 3, height, width),
            torch.zeros(batch, height, width, dtype=torch.bool),
            no_padding,
        )

    def test_module_holds_no_parameters_or_buffers(self) -> None:
        """The cache is only sound because the embedding depends on no learnable state.

        If this module ever gains a parameter or buffer, a cached embedding could outlive the value it was derived from,
        so the cache in ``forward`` would have to be invalidated.
        """
        module = PositionEmbeddingSine(num_pos_feats=16, normalize=True)
        assert list(module.parameters()) == []
        assert list(module.buffers()) == []
        assert module.state_dict() == {}

    def test_cache_is_not_in_the_state_dict(self) -> None:
        """A populated cache must not leak into checkpoints."""
        module = PositionEmbeddingSine(num_pos_feats=16, normalize=True)
        module(self._nested(12, 20))
        assert module._pos_cache != {}
        assert module.state_dict() == {}

    def test_cached_value_matches_recomputation_bitwise(self) -> None:
        """Flagging a batch must not change the embedding by a single bit."""
        module = PositionEmbeddingSine(num_pos_feats=16, normalize=True)
        flagged = self._nested(12, 20)
        unflagged = NestedTensor(flagged.tensors, flagged.mask, False)
        assert torch.equal(module(flagged), module(unflagged))

    def test_repeated_calls_return_identical_values(self) -> None:
        """A cache hit must return the same embedding the first call produced."""
        module = PositionEmbeddingSine(num_pos_feats=16, normalize=True)
        nested = self._nested(12, 20)
        first = module(nested).clone()
        assert torch.equal(module(nested), first)
        assert torch.equal(module(self._nested(12, 20)), first)

    def test_align_dim_orders_is_part_of_the_key(self) -> None:
        """The two layouts differ; one must not be served from the other's entry."""
        module = PositionEmbeddingSine(num_pos_feats=16, normalize=True)
        nested = self._nested(12, 20)
        aligned = module(nested, align_dim_orders=True)
        unaligned = module(nested, align_dim_orders=False)
        assert aligned.shape != unaligned.shape
        assert len(module._pos_cache) == 2

    def test_padded_batch_is_not_cached(self) -> None:
        """Without the flag the mask contents matter, so nothing may be reused."""
        module = PositionEmbeddingSine(num_pos_feats=16, normalize=True)
        mask = torch.zeros(1, 12, 20, dtype=torch.bool)
        mask[:, 8:, :] = True
        padded = NestedTensor(torch.rand(1, 3, 12, 20), mask, False)
        recomputed = module(padded)
        assert module._pos_cache == {}
        assert torch.equal(module(padded), recomputed)

    def test_same_shape_different_padding_is_not_confused(self) -> None:
        """An unpadded batch's entry must not be served to a padded batch of equal shape."""
        module = PositionEmbeddingSine(num_pos_feats=16, normalize=True)
        unpadded = module(self._nested(12, 20)).clone()
        mask = torch.zeros(1, 12, 20, dtype=torch.bool)
        mask[:, 8:, :] = True
        padded = module(NestedTensor(torch.rand(1, 3, 12, 20), mask, False))
        assert not torch.equal(unpadded, padded)

    def test_cache_is_bounded(self) -> None:
        """Multi-scale training cycles through shapes; retained device memory must stay bounded."""
        module = PositionEmbeddingSine(num_pos_feats=16, normalize=True)
        for i in range(_POS_CACHE_MAX_ENTRIES + 4):
            module(self._nested(8 + i, 8 + i))
        assert len(module._pos_cache) <= _POS_CACHE_MAX_ENTRIES

    def test_matches_unflagged_after_default_config_batch_uniform_resize(self) -> None:
        """The cache still agrees with recomputation after the real default-training mutation.

        The default training config (``square_resize_div_64=True``, ``multi_scale=True``,
        ``do_random_resize_via_padding=False``) resizes every sample to one fixed square scale before collate, so the
        batch is flagged. ``RFDETRLightningModule.on_train_batch_start`` then resizes the whole batch uniformly to a
        randomly chosen scale via the same two in-place ``F.interpolate`` calls reproduced below, without touching
        ``no_padding``. Nearest-neighbour resampling of an all-False mask stays all-False at any output size, so the
        cached embedding must still match the recomputed one on the mutated batch.
        """
        images = [torch.rand(3, 12, 12) for _ in range(2)]
        nested = nested_tensor_from_tensor_list(images)
        with torch.no_grad():
            nested.tensors = F.interpolate(nested.tensors, size=(20, 20), mode="bilinear", align_corners=False)
            nested.mask = (
                F.interpolate(nested.mask.unsqueeze(1).float(), size=(20, 20), mode="nearest").squeeze(1).bool()
            )
        assert nested.no_padding is True

        module = PositionEmbeddingSine(num_pos_feats=16, normalize=True)
        unflagged_twin = NestedTensor(nested.tensors, nested.mask, False)
        assert torch.equal(module(nested), module(unflagged_twin))
