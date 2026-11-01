# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for rfdetr.utilities.state_dict."""

import logging
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning import LightningModule, Trainer

from rfdetr.config import RFDETRNanoConfig
from rfdetr.utilities.state_dict import (
    _make_fit_loop_state,
    remap_projector_to_cross_attn,
    strip_checkpoint,
    validate_checkpoint_compatibility,
)
from tests._markers import _requires_posix_permissions

#: Checkpoint keys under which the patch-embedding projection weight is saved: plain, and with ``backbone_lora=True``.
_PROJECTION_KEYS = [
    "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight",
    "backbone.0.encoder.base_model.model.encoder.embeddings.patch_embeddings.projection.weight",
]
_PROJECTION_KEY = _PROJECTION_KEYS[0]


def _projection_checkpoint(*shape: int, key: str = _PROJECTION_KEY, **extra: object) -> dict:
    """Build a checkpoint whose patch-embedding projection weight has the given shape.

    Args:
        *shape: Shape of the projection weight.
        key: State-dict key the weight is stored under.
        **extra: Additional top-level checkpoint entries, such as ``args``.

    Returns:
        A checkpoint dict with the weight under ``model[key]``.

    Examples:
        >>> _projection_checkpoint(384, 3, 16, 16)["model"][_PROJECTION_KEY].shape
        torch.Size([384, 3, 16, 16])
    """
    return {**extra, "model": {key: torch.zeros(shape)}}


def _logged_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the messages the ``rf-detr`` logger recorded in ``caplog``.

    Args:
        caplog: The pytest log-capture fixture.

    Returns:
        Formatted messages of the records emitted by the ``rf-detr`` logger.

    Examples:
        >>> record = logging.LogRecord("rf-detr", logging.WARNING, "f.py", 1, "hi", (), None)
        >>> _logged_messages(SimpleNamespace(records=[record]))
        ['hi']
    """
    return [r.getMessage() for r in caplog.records if r.name == "rf-detr"]


def _load_checkpoint(path: Path) -> dict:
    """Load a checkpoint saved by ``torch.save`` onto the CPU, unpickling arbitrary objects.

    Args:
        path: Checkpoint file.

    Returns:
        The saved checkpoint.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "ckpt.pth"
        >>> torch.save({"a": 1}, path)
        >>> _load_checkpoint(path)
        {'a': 1}
    """
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# _make_fit_loop_state
# ---------------------------------------------------------------------------


class TestMakeFitLoopState:
    """Tests for _make_fit_loop_state epoch counter encoding."""

    @pytest.mark.parametrize(
        "epoch,expected_n",
        [
            pytest.param(0, 1, id="epoch_0"),
            pytest.param(4, 5, id="epoch_4"),
            pytest.param(9, 10, id="epoch_9"),
        ],
    )
    def test_epoch_progress_completed_is_epoch_plus_one(self, epoch: int, expected_n: int) -> None:
        """epoch_progress.current.completed == epoch + 1 so PTL sets current_epoch correctly."""
        state = _make_fit_loop_state(epoch)
        assert state["epoch_progress"]["current"]["completed"] == expected_n
        assert state["epoch_progress"]["total"]["completed"] == expected_n

    def test_epoch_progress_all_counters_equal(self) -> None:
        """All four counters in epoch_progress should be equal (epoch fully completed)."""
        state = _make_fit_loop_state(7)
        for scope in ("total", "current"):
            ep = state["epoch_progress"][scope]
            vals = [ep["ready"], ep["started"], ep["processed"], ep["completed"]]
            assert len(set(vals)) == 1, f"epoch_progress.{scope} counters differ: {ep}"

    def test_batches_that_stepped_is_zero(self) -> None:
        """Optimizer/scheduler state should start fresh; _batches_that_stepped must be 0."""
        state = _make_fit_loop_state(3)
        assert state["epoch_loop.state_dict"]["_batches_that_stepped"] == 0

    def test_batch_progress_is_zero(self) -> None:
        """Batch progress counters should be zeroed out (not mid-batch resume)."""
        state = _make_fit_loop_state(5)
        for key in ("epoch_loop.batch_progress", "epoch_loop.val_loop.batch_progress"):
            bp = state[key]
            assert bp["is_last_batch"] is False
            for scope in ("total", "current"):
                assert all(v == 0 for v in bp[scope].values()), f"{key}.{scope} not zero: {bp[scope]}"

    def test_ptl_accepts_fit_loop_state(self) -> None:
        """PTL's _FitLoop.load_state_dict must not raise with our synthesised state dict."""

        class _DummyModule(LightningModule):
            def training_step(self, batch, idx):
                return torch.tensor(0.0, requires_grad=True)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=1e-3)

        trainer = Trainer(max_epochs=10, accelerator="cpu", enable_progress_bar=False, logger=False)
        trainer.strategy.connect(_DummyModule())

        epoch = 4
        state = _make_fit_loop_state(epoch)
        trainer.fit_loop.load_state_dict(state)
        assert trainer.current_epoch == epoch + 1

    def test_required_top_level_keys_present(self) -> None:
        """State dict must contain all keys the FitLoop accesses during load."""
        required = {
            "state_dict",
            "epoch_loop.state_dict",
            "epoch_loop.batch_progress",
            "epoch_loop.scheduler_progress",
            "epoch_loop.automatic_optimization.state_dict",
            "epoch_loop.automatic_optimization.optim_progress",
            "epoch_loop.manual_optimization.state_dict",
            "epoch_loop.manual_optimization.optim_step_progress",
            "epoch_loop.val_loop.state_dict",
            "epoch_loop.val_loop.batch_progress",
            "epoch_progress",
        }
        state = _make_fit_loop_state(0)
        missing = required - set(state.keys())
        assert not missing, f"Missing keys: {missing}"


# ---------------------------------------------------------------------------
# validate_checkpoint_compatibility
# ---------------------------------------------------------------------------


class TestValidateCheckpointCompatibility:
    """Direct unit tests for validate_checkpoint_compatibility."""

    @pytest.fixture
    def rf_detr_warnings(self, caplog, monkeypatch):
        """Capture ``rf-detr`` logger records, which do not propagate to the root logger by default."""
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            yield caplog

    # ------------------------------------------------------------------
    # Early-return / silent-skip cases
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("checkpoint", "model_args"),
        [
            pytest.param(
                {"model": {}},
                SimpleNamespace(segmentation_head=False, patch_size=14),
                id="no-args-key",
            ),
            pytest.param(
                {"args": SimpleNamespace(segmentation_head=True, patch_size=14)},
                SimpleNamespace(patch_size=14),  # no segmentation_head attribute
                id="ckpt-only-segmentation-head",
            ),
            pytest.param(
                {"args": SimpleNamespace(segmentation_head=False)},  # no patch_size
                SimpleNamespace(segmentation_head=False, patch_size=14),
                id="ckpt-without-patch-size",
            ),
            pytest.param(
                {"args": SimpleNamespace(segmentation_head=False, patch_size=14)},
                SimpleNamespace(segmentation_head=False, patch_size=14),
                id="matching-detection",
            ),
            pytest.param(
                {"args": SimpleNamespace(segmentation_head=True, patch_size=16)},
                SimpleNamespace(segmentation_head=True, patch_size=16),
                id="matching-segmentation",
            ),
        ],
    )
    def test_compatible_or_incomplete_args_do_not_raise(self, checkpoint: dict, model_args: SimpleNamespace) -> None:
        """Matching args, or an attribute missing on one side only, pass silently instead of raising."""
        validate_checkpoint_compatibility(checkpoint, model_args)  # must not raise

    # ------------------------------------------------------------------
    # segmentation_head mismatch
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("ckpt_segmentation_head", "model_segmentation_head"),
        [
            pytest.param(True, False, id="segmentation-ckpt-into-detection-model"),
            pytest.param(False, True, id="detection-ckpt-into-segmentation-model"),
        ],
    )
    def test_segmentation_head_mismatch_raises(
        self, ckpt_segmentation_head: bool, model_segmentation_head: bool
    ) -> None:
        """A checkpoint and model that disagree on the segmentation head must raise ValueError."""
        checkpoint = {"args": SimpleNamespace(segmentation_head=ckpt_segmentation_head, patch_size=14)}
        model_args = SimpleNamespace(segmentation_head=model_segmentation_head, patch_size=14)

        with pytest.raises(ValueError) as exc_info:
            validate_checkpoint_compatibility(checkpoint, model_args)

        assert "segmentation head" in str(exc_info.value)

    # ------------------------------------------------------------------
    # patch_size mismatch
    # ------------------------------------------------------------------

    def test_patch_size_mismatch_raises_with_both_sizes(self) -> None:
        """patch_size mismatch must raise ValueError and mention both sizes."""
        checkpoint = {"args": SimpleNamespace(segmentation_head=False, patch_size=12)}
        model_args = SimpleNamespace(segmentation_head=False, patch_size=16)

        with pytest.raises(ValueError) as exc_info:
            validate_checkpoint_compatibility(checkpoint, model_args)

        assert "patch_size=12" in str(exc_info.value)
        assert "patch_size=16" in str(exc_info.value)

    # ------------------------------------------------------------------
    # patch_size inferred from projection weight (no "args" key)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("proj_key", _PROJECTION_KEYS)
    @pytest.mark.parametrize(
        "ckpt_patch_size,model_patch_size",
        [
            pytest.param(16, 12, id="ckpt_16_model_12"),
            pytest.param(14, 16, id="ckpt_14_model_16"),
        ],
    )
    def test_projection_weight_patch_size_mismatch_raises(
        self, ckpt_patch_size: int, model_patch_size: int, proj_key: str
    ) -> None:
        """Projection weight shape infers the ckpt patch_size when 'args' is absent, and a mismatch raises.

        Regression test for #965 — pretrained COCO weights lack 'args', so the shape-based fallback must fire before
        load_state_dict raises a cryptic RuntimeError. The second key is where a ``backbone_lora=True`` run saves the
        same weight (#1540); its checkpoint args are a ``TrainConfig`` dump without ``patch_size``.
        """
        checkpoint = _projection_checkpoint(384, 3, ckpt_patch_size, ckpt_patch_size, key=proj_key)  # no "args" key
        model_args = SimpleNamespace(patch_size=model_patch_size)

        with pytest.raises(ValueError) as exc_info:
            validate_checkpoint_compatibility(checkpoint, model_args)

        assert f"patch_size={ckpt_patch_size}" in str(exc_info.value)
        assert f"patch_size={model_patch_size}" in str(exc_info.value)

    @pytest.mark.parametrize("proj_key", _PROJECTION_KEYS)
    def test_projection_weight_patch_size_match_does_not_raise(self, proj_key: str) -> None:
        """A projection weight whose inferred patch_size equals the model's passes (see the mismatch test for #965)."""
        checkpoint = _projection_checkpoint(384, 3, 16, 16, key=proj_key)  # no "args" key
        model_args = SimpleNamespace(patch_size=16)

        validate_checkpoint_compatibility(checkpoint, model_args)  # must not raise

    @pytest.mark.parametrize(
        "checkpoint,model_args_kwargs",
        [
            pytest.param({}, {"patch_size": 16}, id="no_model_key_skips"),
            pytest.param({"model": {}}, {"patch_size": 16}, id="no_projection_key_skips"),
            pytest.param(_projection_checkpoint(384, 3, 16, 16), {}, id="model_no_patch_size_attr_skips"),
            # 2D/3D/5D: not a Conv2d weight, the rank guard (== 4) must skip cleanly
            pytest.param(_projection_checkpoint(384, 3), {"patch_size": 16}, id="proj_weight_2d_skips"),
            pytest.param(_projection_checkpoint(384, 3, 16), {"patch_size": 16}, id="proj_weight_3d_skips"),
            pytest.param(_projection_checkpoint(384, 3, 16, 16, 16), {"patch_size": 8}, id="proj_weight_5d_skips"),
            # Hypothetical backbones with non-square kernels, where shape[-1] would not equal patch_size
            pytest.param(_projection_checkpoint(384, 3, 16, 14), {"patch_size": 16}, id="non_square_kernel_skips"),
            # Projection suggests 16, but an explicit args.patch_size=14 (namespace or PTL-style dict) takes precedence
            pytest.param(
                _projection_checkpoint(384, 3, 16, 16, args=SimpleNamespace(patch_size=14)),
                {"patch_size": 14},
                id="args_patch_size_suppresses_projection_inference",
            ),
            pytest.param(
                _projection_checkpoint(384, 3, 16, 16, args={"patch_size": 14}),
                {"patch_size": 14},
                id="dict_args_patch_size_suppresses_projection_inference",
            ),
        ],
    )
    def test_projection_inference_silently_skips_when_incomplete(
        self, checkpoint: dict, model_args_kwargs: dict
    ) -> None:
        """Shape-based patch_size check is skipped when key or attribute is absent.

        Verifies backward compatibility: missing projection key, missing model key, model_args without patch_size
        attribute, non-4D or non-square projection weights, or an explicit args.patch_size (SimpleNamespace or dict)
        must all be handled without error.
        """
        model_args = SimpleNamespace(**model_args_kwargs)
        validate_checkpoint_compatibility(checkpoint, model_args)  # must not raise

    # ------------------------------------------------------------------
    # class-count mismatch warnings
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("ckpt_rows", "num_classes", "expected"),
        [
            pytest.param(91, 2, "re-initialized to 2 classes", id="backbone-pretrain-reinitializes-head"),
            pytest.param(3, 90, "Pass num_classes=2", id="finetune-checkpoint-hints-num-classes"),
        ],
    )
    def test_class_count_mismatch_warns(self, rf_detr_warnings, ckpt_rows: int, num_classes: int, expected: str):
        """A checkpoint head of a different size warns, with the re-init notice or the num_classes hint."""
        checkpoint = {
            "args": SimpleNamespace(segmentation_head=False, patch_size=14),
            "model": {"class_embed.bias": torch.randn(ckpt_rows)},
        }
        model_args = SimpleNamespace(segmentation_head=False, patch_size=14, num_classes=num_classes)

        validate_checkpoint_compatibility(checkpoint, model_args)

        warning_msgs = _logged_messages(rf_detr_warnings)
        assert any(expected in msg for msg in warning_msgs), f"Expected {expected!r} warning, got: {warning_msgs}"

    @pytest.mark.parametrize(
        "model_entry",
        [
            pytest.param({"model": {"class_embed.bias": torch.zeros(91)}}, id="matching-class-count"),
            pytest.param({}, id="no-model-key"),  # backward compat: nothing to compare against
        ],
    )
    def test_class_count_without_mismatch_is_silent(self, rf_detr_warnings, model_entry: dict) -> None:
        """A matching class count, or a checkpoint without a 'model' key, emits no warning."""
        checkpoint = {"args": SimpleNamespace(segmentation_head=False, patch_size=14), **model_entry}
        model_args = SimpleNamespace(segmentation_head=False, patch_size=14, num_classes=90)

        validate_checkpoint_compatibility(checkpoint, model_args)

        warning_msgs = _logged_messages(rf_detr_warnings)
        assert not warning_msgs, f"Expected no warnings, got: {warning_msgs}"

    def test_class_count_mismatch_explicit_larger_num_classes_warns_head_resize(self, rf_detr_warnings):
        """A set num_classes larger than the checkpoint: the warning says the configured count is kept.

        `load_pretrain_weights` honors a set `num_classes` and re-expands the head after loading; the warning must not
        claim the checkpoint class count is used.
        """
        checkpoint = {"model": {"class_embed.bias": torch.randn(91)}}
        model_args = RFDETRNanoConfig(num_classes=93)

        validate_checkpoint_compatibility(checkpoint, model_args)

        warning_msgs = _logged_messages(rf_detr_warnings)
        assert any(
            "Keeping num_classes=93" in msg and "grows from 90 to 93 classes" in msg and "3 added class(es)" in msg
            for msg in warning_msgs
        ), f"Expected a 'Keeping num_classes=93' resize warning, got: {warning_msgs}"
        assert not any("Using checkpoint class count" in msg for msg in warning_msgs), warning_msgs

    @pytest.mark.parametrize(
        ("model_args", "ckpt_rows", "expected", "forbidden"),
        [
            pytest.param(
                RFDETRNanoConfig(num_classes=10),
                91,
                ("re-initialized to 10 classes",),
                ("Keeping num_classes", "Using checkpoint class count"),
                id="set-smaller-trims-head",
            ),
            pytest.param(
                RFDETRNanoConfig(),
                4,
                ("Using checkpoint class count (3)", "Pass num_classes=3"),
                ("Keeping num_classes",),
                id="unset-larger-uses-checkpoint",
            ),
            pytest.param(
                RFDETRNanoConfig(num_classes=90),
                4,
                ("Keeping num_classes=90",),
                ("Using checkpoint class count",),
                id="set-to-class-default-is-kept",
            ),
            pytest.param(
                RFDETRNanoConfig(num_classes=3),
                4,
                (),
                ("Checkpoint has",),
                id="set-matching-rows-is-silent",
            ),
        ],
    )
    def test_class_count_warning_with_pydantic_config(
        self, rf_detr_warnings, model_args, ckpt_rows, expected, forbidden
    ):
        """The class-count warning follows whether a pydantic config has ``num_classes`` set, not its value.

        The sibling class-count tests pass a ``SimpleNamespace``, which has no ``model_fields_set`` and so always takes
        the unset branch. A real config distinguishes a set ``num_classes`` (even one equal to the class default) from
        an unset one, and that decides whether the warning says the configured count is kept or the checkpoint's is
        used.
        """
        checkpoint = {"model": {"class_embed.bias": torch.randn(ckpt_rows)}}

        validate_checkpoint_compatibility(checkpoint, model_args)

        warning_msgs = _logged_messages(rf_detr_warnings)
        assert all(any(text in msg for msg in warning_msgs) for text in expected), warning_msgs
        assert not any(text in msg for text in forbidden for msg in warning_msgs), warning_msgs

    def test_class_count_warning_for_num_classes_assigned_after_construction(self, rf_detr_warnings):
        """A ``num_classes`` assigned after the config is built counts as set and is kept.

        Dataset alignment assigns ``num_classes`` on an existing config instead of passing it to the constructor; the
        assignment adds the field to ``model_fields_set``, so a checkpoint with fewer classes gets the keep-and-resize
        warning rather than "Using checkpoint class count".
        """
        checkpoint = {"model": {"class_embed.bias": torch.randn(91)}}
        model_args = RFDETRNanoConfig()
        model_args.num_classes = 93

        validate_checkpoint_compatibility(checkpoint, model_args)

        warning_msgs = _logged_messages(rf_detr_warnings)
        assert any("Keeping num_classes=93" in msg for msg in warning_msgs), warning_msgs
        assert not any("Using checkpoint class count" in msg for msg in warning_msgs), warning_msgs


class TestRemapProjectorToCrossAttn:
    """Tests for dual-projector checkpoint key remapping."""

    def test_clones_projector_weights_when_dual_projector_enabled(self) -> None:
        """Dual-projector models clone projector keys into cross_attn_projector when missing."""
        state_dict = {
            "backbone.0.projector.0.weight": torch.randn(4, 4, 1, 1),
            "backbone.0.projector.0.bias": torch.randn(4),
        }
        model = SimpleNamespace(backbone=[SimpleNamespace(dual_projector=True)])

        remapped = remap_projector_to_cross_attn(state_dict, model)

        assert remapped is state_dict
        assert "backbone.0.cross_attn_projector.0.weight" in remapped
        assert "backbone.0.cross_attn_projector.0.bias" in remapped
        assert torch.equal(
            remapped["backbone.0.cross_attn_projector.0.weight"],
            state_dict["backbone.0.projector.0.weight"],
        )
        assert torch.equal(
            remapped["backbone.0.cross_attn_projector.0.bias"],
            state_dict["backbone.0.projector.0.bias"],
        )

    def test_skips_when_cross_attn_keys_already_present(self) -> None:
        """No remap is applied when cross_attn_projector keys already exist."""
        state_dict = {
            "backbone.0.projector.0.weight": torch.randn(4, 4, 1, 1),
            "backbone.0.cross_attn_projector.0.weight": torch.randn(4, 4, 1, 1),
        }
        model = SimpleNamespace(backbone=[SimpleNamespace(dual_projector=True)])

        remapped = remap_projector_to_cross_attn(state_dict, model)

        assert remapped is state_dict
        assert len([key for key in remapped if key.startswith("backbone.0.cross_attn_projector.")]) == 1


class TestStripCheckpoint:
    """Tests for strip_checkpoint loop-stub backfill."""

    def _make_minimal_ckpt(self, tmp_path, extra: dict | None = None) -> Path:
        """Write a minimal checkpoint to a temp file."""
        payload = {"model": {"w": torch.tensor(1.0)}, "args": {"lr": 1e-4}}
        if extra:
            payload.update(extra)
        ckpt_path = Path(tmp_path) / "ckpt.pth"
        torch.save(payload, ckpt_path)
        return ckpt_path

    def test_adds_validate_and_test_loop_stubs_when_loops_lack_them(self, tmp_path) -> None:
        """Old checkpoints with loops but no validate_loop/test_loop get stubs backfilled."""
        ckpt_path = self._make_minimal_ckpt(
            tmp_path,
            extra={"loops": {"fit_loop": {"state_dict": {}}}},
        )
        strip_checkpoint(ckpt_path)
        result = _load_checkpoint(ckpt_path)
        assert result["loops"]["validate_loop"] == {"state_dict": {}}
        assert result["loops"]["test_loop"] == {"state_dict": {}}

    def test_preserves_existing_validate_loop_stub(self, tmp_path) -> None:
        """Checkpoints with validate_loop already present are not overwritten."""
        original_stub = {"state_dict": {"some_key": 1}}
        ckpt_path = self._make_minimal_ckpt(
            tmp_path,
            extra={"loops": {"fit_loop": {"state_dict": {}}, "validate_loop": original_stub}},
        )
        strip_checkpoint(ckpt_path)
        result = _load_checkpoint(ckpt_path)
        assert result["loops"]["validate_loop"] == original_stub

    def test_leaves_loops_absent_without_loops_key(self, tmp_path) -> None:
        """Checkpoints without a loops key must not gain one after stripping."""
        ckpt_path = self._make_minimal_ckpt(tmp_path)
        strip_checkpoint(ckpt_path)
        result = _load_checkpoint(ckpt_path)
        assert "loops" not in result

    def test_writes_extra_metadata(self, tmp_path) -> None:
        """extra_metadata key/value pairs are merged into the stripped checkpoint."""
        ckpt_path = self._make_minimal_ckpt(tmp_path)
        strip_checkpoint(ckpt_path, extra_metadata={"best_total_source": "ema"})
        result = _load_checkpoint(ckpt_path)
        assert result["best_total_source"] == "ema"

    def test_extra_metadata_overrides_preserved_key(self, tmp_path) -> None:
        """extra_metadata wins over a preserved same-named key."""
        ckpt_path = self._make_minimal_ckpt(tmp_path, extra={"rfdetr_version": "1.0.0"})
        strip_checkpoint(ckpt_path, extra_metadata={"rfdetr_version": "override"})
        result = _load_checkpoint(ckpt_path)
        assert result["rfdetr_version"] == "override"

    @_requires_posix_permissions
    @pytest.mark.parametrize("process_umask", [0o022, 0o027], indirect=True, ids=oct)
    def test_mode_follows_umask(self, tmp_path: Path, process_umask: int) -> None:
        """A checkpoint saved and stripped under one umask ends with that umask's mode, not a ``0o600`` temp mode."""
        ckpt_path = self._make_minimal_ckpt(tmp_path)
        strip_checkpoint(ckpt_path)
        assert stat.S_IMODE(ckpt_path.stat().st_mode) == 0o666 & ~process_umask

    @_requires_posix_permissions
    @pytest.mark.usefixtures("process_umask")  # the default 0o022, so both modes differ from a fresh file's 0o644
    @pytest.mark.parametrize("mode", [pytest.param(0o600, id="0o600"), pytest.param(0o640, id="0o640")])
    def test_keeps_existing_mode(self, tmp_path: Path, mode: int) -> None:
        """Stripping keeps the checkpoint's own permission bits instead of widening them to the umask's ``0o644``."""
        ckpt_path = self._make_minimal_ckpt(tmp_path)
        ckpt_path.chmod(mode)
        strip_checkpoint(ckpt_path)
        assert stat.S_IMODE(ckpt_path.stat().st_mode) == mode

    def test_cleanup_failure_does_not_mask_save_error(self, tmp_path: Path, mocker) -> None:
        """A temp file that cannot be removed after a failed save must not hide the save's own error."""
        ckpt_path = self._make_minimal_ckpt(tmp_path)
        mocker.patch("torch.save", side_effect=RuntimeError("save failed"))
        mocker.patch("rfdetr.utilities.state_dict.os.remove", side_effect=PermissionError("temp file locked"))

        with pytest.raises(RuntimeError) as exc_info:
            strip_checkpoint(ckpt_path)

        assert "save failed" in str(exc_info.value)
