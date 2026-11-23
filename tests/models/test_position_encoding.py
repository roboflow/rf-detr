# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr.models.position_encoding.build_position_encoding.

Covers:
- Supported aliases ``"sine"`` / ``"v2"`` return a ``PositionEmbeddingSine`` instance.
- Unsupported but previously accepted aliases ``"learned"`` / ``"v3"`` now raise ``ValueError`` naming supported
  alternatives.
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

    @pytest.mark.parametrize("alias", ["sine", "v2"])
    def test_returns_sine_embedding(self, alias: str) -> None:
        """Supported aliases produce a PositionEmbeddingSine with normalized=True."""
        enc = build_position_encoding(hidden_dim=256, position_embedding=alias)
        assert isinstance(enc, PositionEmbeddingSine)
        assert enc.normalize is True

    @pytest.mark.parametrize(("hidden_dim", "expected_num_pos_feats"), [(256, 128), (512, 256)])
    def test_num_pos_feats_is_half_hidden_dim(self, hidden_dim: int, expected_num_pos_feats: int) -> None:
        """The sine encoding uses hidden_dim // 2 positional feature dimensions."""
        enc = build_position_encoding(hidden_dim=hidden_dim, position_embedding="sine")
        assert enc.num_pos_feats == expected_num_pos_feats


class TestBuildPositionEncodingUnsupportedValues:
    """build_position_encoding raises ValueError for broken or unknown aliases."""

    @pytest.mark.parametrize("alias", ["learned", "v3"])
    def test_learned_raises_value_error(self, alias: str) -> None:
        """'learned' and 'v3' are doubly broken and must raise ValueError immediately.

        The PositionEmbeddingLearned class has two bugs:
        1. forward() signature is incompatible with Joiner.forward() (no align_dim_orders param).
        2. h, w = x.shape[:2] unpacks batch and channels instead of height and width.
        Rejecting them at build time is preferable to a silent or confusing runtime failure.
        """
        with pytest.raises(ValueError) as exc_info:
            build_position_encoding(hidden_dim=256, position_embedding=alias)
        assert "not supported" in str(exc_info.value)

    def test_unknown_value_raises_value_error(self) -> None:
        """A fully unknown alias raises ValueError stating that it is not supported."""
        with pytest.raises(ValueError) as exc_info:
            build_position_encoding(hidden_dim=256, position_embedding="unknown_variant")
        assert "not supported" in str(exc_info.value)

    @pytest.mark.parametrize("alias", ["learned", "v3"])
    def test_error_message_mentions_supported_alternatives(self, alias: str) -> None:
        """Error message for 'learned'/'v3' mentions at least one supported alternative."""
        with pytest.raises(ValueError) as exc_info:
            build_position_encoding(hidden_dim=256, position_embedding=alias)
        assert "sine" in str(exc_info.value)


class TestPositionEmbeddingSineNoPaddingCache:
    """The embedding is cached only for unpadded batches, and only where that is exact.

    ``PositionEmbeddingSine`` holds no parameters. For an all-False mask its evaluation output is a pure function of the
    mask shape, device, requested layout, and encoding configuration recorded in the key. The cache must reproduce
    recomputation bit for bit, stay empty during training and for padded batches, release entries on lifecycle/device
    transitions, and not grow without bound.
    """

    @pytest.fixture
    def eval_module(self) -> PositionEmbeddingSine:
        """Return an evaluation-mode sine embedding with 16 positional features.

        Examples:
            Pytest fixtures cannot be called directly outside fixture injection.

            >>> TestPositionEmbeddingSineNoPaddingCache.eval_module(None)  # doctest: +SKIP
        """
        return PositionEmbeddingSine(num_pos_feats=16, normalize=True).eval()

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

    @staticmethod
    def _padded_nested() -> NestedTensor:
        """Return a 12x20 ``NestedTensor`` whose rows from 8 on are padding and which is not flagged ``no_padding``.

        Examples:
            >>> TestPositionEmbeddingSineNoPaddingCache._padded_nested().mask[0, :, 0].int().tolist()
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        """
        mask = torch.zeros(1, 12, 20, dtype=torch.bool)
        mask[:, 8:, :] = True
        return NestedTensor(torch.rand(1, 3, 12, 20), mask, False)

    def test_module_holds_no_parameters_or_buffers(self) -> None:
        """The cache is only sound because the embedding depends on no learnable state.

        If this module ever gains a parameter or buffer, a cached embedding could outlive the value it was derived from,
        so the cache in ``forward`` would have to be invalidated.
        """
        module = PositionEmbeddingSine(num_pos_feats=16, normalize=True)
        assert list(module.parameters()) == []
        assert list(module.buffers()) == []
        assert module.state_dict() == {}

    def test_cache_is_not_in_the_state_dict(self, eval_module: PositionEmbeddingSine) -> None:
        """A populated cache must not leak into checkpoints."""
        eval_module(self._nested(12, 20))
        assert eval_module._pos_cache != {}
        assert eval_module.state_dict() == {}

    def test_cached_value_matches_recomputation_bitwise(self, eval_module: PositionEmbeddingSine) -> None:
        """Flagging a batch must not change the embedding by a single bit."""
        flagged = self._nested(12, 20)
        unflagged = NestedTensor(flagged.tensors, flagged.mask, False)
        assert torch.equal(eval_module(flagged), eval_module(unflagged))

    def test_repeated_calls_return_identical_values(self, eval_module: PositionEmbeddingSine) -> None:
        """A cache hit must return the same embedding the first call produced."""
        nested = self._nested(12, 20)
        first = eval_module(nested).clone()
        assert torch.equal(eval_module(nested), first)
        assert torch.equal(eval_module(self._nested(12, 20)), first)

    def test_align_dim_orders_is_part_of_the_key(self, eval_module: PositionEmbeddingSine) -> None:
        """The two layouts differ; one must not be served from the other's entry."""
        nested = self._nested(12, 20)
        aligned = eval_module(nested, align_dim_orders=True)
        unaligned = eval_module(nested, align_dim_orders=False)
        assert aligned.shape != unaligned.shape
        assert len(eval_module._pos_cache) == 2

    def test_padded_batch_is_not_cached(self, eval_module: PositionEmbeddingSine) -> None:
        """Without the flag the mask contents matter, so nothing may be reused."""
        padded = self._padded_nested()
        recomputed = eval_module(padded)
        assert eval_module._pos_cache == {}
        assert torch.equal(eval_module(padded), recomputed)

    def test_same_shape_different_padding_is_not_confused(self, eval_module: PositionEmbeddingSine) -> None:
        """An unpadded batch's entry must not be served to a padded batch of equal shape."""
        unpadded = eval_module(self._nested(12, 20)).clone()
        padded = eval_module(self._padded_nested())
        assert not torch.equal(unpadded, padded)

    def test_cache_is_bounded(self, eval_module: PositionEmbeddingSine) -> None:
        """Variable-resolution inference cannot retain an unbounded number of embeddings."""
        for i in range(_POS_CACHE_MAX_ENTRIES + 4):
            eval_module(self._nested(8 + i, 8 + i))
        assert len(eval_module._pos_cache) <= _POS_CACHE_MAX_ENTRIES

    def test_training_does_not_cache_after_default_config_batch_uniform_resize(self) -> None:
        """Training recomputes embeddings after the real default-training batch mutation.

        The default training config (``square_resize_div_64=True``, ``multi_scale="per-batch"``) resizes every sample to
        one fixed square scale before collate, so the batch is flagged. ``RFDETRLightningModule.on_train_batch_start``
        then resizes the whole batch uniformly to a randomly chosen scale via the same two in-place ``F.interpolate``
        calls reproduced below, without touching ``no_padding``. Nearest-neighbour resampling of an all-False mask stays
        all-False at any output size, so the training result must still match the unflagged path without retaining the
        embedding.
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
        assert module._pos_cache == {}

    def test_entering_training_clears_evaluation_cache(self, eval_module: PositionEmbeddingSine) -> None:
        """Switching an evaluated model back to training releases retained embeddings."""
        eval_module(self._nested(12, 20))
        assert eval_module._pos_cache != {}

        eval_module.train()

        assert eval_module._pos_cache == {}

    def test_device_transfer_clears_cache(self, eval_module: PositionEmbeddingSine) -> None:
        """The public module transfer path releases tensors owned by the previous device."""
        eval_module(self._nested(12, 20))
        assert eval_module._pos_cache != {}

        eval_module.to(torch.device("cpu"))

        assert eval_module._pos_cache == {}

    def test_cache_key_tracks_encoding_configuration(self, eval_module: PositionEmbeddingSine) -> None:
        """Changing a public encoding attribute cannot reuse an entry built from the old value."""
        nested = self._nested(12, 20)
        initial = eval_module(nested).clone()

        eval_module.scale = 3.0
        updated = eval_module(nested)
        expected = PositionEmbeddingSine(num_pos_feats=16, normalize=True, scale=3.0).eval()(nested)

        assert not torch.equal(initial, updated)
        assert torch.equal(updated, expected)
