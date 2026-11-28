# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""A Lightning ``.ckpt`` written by ``train()`` describes the model it holds, like the best ``.pth`` files (#1552).

``last.ckpt`` and ``checkpoint_<epoch>.ckpt`` come from plain ``ModelCheckpoint`` callbacks. They used to carry weights
and optimizer state but no ``args``, ``model_config`` or ``model_name``, so ``RFDETR.from_checkpoint("last.ckpt")``
raised ``KeyError: 'args'`` and ``RFDETRNano(pretrain_weights="last.ckpt")`` came back with the COCO class names. The
end-to-end tests read from one real one-epoch CPU run of ``RFDETRNano`` on a tiny synthetic dataset;
``TestOnSaveCheckpoint`` calls the hook on a stand-in holding only the ``trainer``, ``model_config`` and
``train_config`` it reads.

Which epochs write ``last.ckpt`` and ``checkpoint_<epoch>.ckpt`` when ``eval_interval > 1``.

``build_trainer`` registers two plain ``ModelCheckpoint`` callbacks next to ``BestModelCallback``. With
``check_val_every_n_epoch != 1`` Lightning's ``ModelCheckpoint`` saves from ``on_validation_end`` unless told to save on
train epoch end, so without ``save_on_train_epoch_end=True`` both callbacks only fired on the epochs ``eval_interval``
validated: ``last.ckpt`` went stale between validations and interval archives skipped every epoch that was not also an
evaluation epoch. Runs a real ``Trainer.fit()`` with mocked model internals (no dataset, no GPU) and inspects the files.
"""

from __future__ import annotations

import datetime
import io
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import Callback

from rfdetr import RFDETR, RFDETRNano
from rfdetr.config import RFDETRBaseConfig, RFDETRLargeDeprecatedConfig, RFDETRNanoConfig, TrainConfig
from rfdetr.training import build_trainer
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.reproducibility import seed_all
from tests.conftest import build_synthetic_dataset
from tests.training.helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts, _TinyModel

#: Not ``RFDETRNano``'s default of 384, so a reload that falls back to class defaults shows up.
_RESOLUTION = 224


#: Seed for the untrained model; module-scoped fixtures run before the autouse per-test reseed.
_TRAIN_SEED = 1552


#: Enough for a train and a valid split; the checkpoint contents do not depend on how long training ran.
_NUM_IMAGES = 16


#: A note a weights-only ``torch.load`` cannot read, so the run covers the ``repr`` fallback every writer applies.
#: ``PurePosixPath`` rather than ``Path`` so its ``repr`` is the same string on every OS.
_NOTES = PurePosixPath("runs/exp7")


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny synthetic COCO detection dataset.

    Examples:
        dataset_dir(tmp_path_factory)
        # A pytest fixture; it cannot run standalone.
    """
    path = tmp_path_factory.mktemp("lightning_checkpoint_reload_dataset")
    build_synthetic_dataset(path, task="detection", num_images=_NUM_IMAGES)
    return path


@pytest.fixture(scope="module")
def training_output_dir(dataset_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train ``RFDETRNano`` on the CPU for one epoch and return its output directory.

    Trains with a ``Path``-valued ``notes`` so the run covers a config value no weights-only ``torch.load`` can read.
    Every checkpoint writer must then agree on the ``repr`` fallback; when only one of them applied it, ``last.ckpt``
    and ``last_ema.pth`` described the same epoch differently.

    Examples:
        training_output_dir(dataset_dir, tmp_path_factory)
        # A pytest fixture that trains a model; it cannot run standalone.
    """
    seed_all(_TRAIN_SEED)
    output_dir = tmp_path_factory.mktemp("lightning_checkpoint_reload")
    RFDETRNano(pretrain_weights=None, resolution=_RESOLUTION, device="cpu").train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=4,
        num_workers=0,
        multi_scale=False,
        augmentation_backend="torchvision",
        tensorboard=False,
        run_test=False,
        device="cpu",
        notes=_NOTES,
    )
    return output_dir


@pytest.fixture(scope="module")
def last_ckpt(training_output_dir: Path) -> dict[str, Any]:
    """The ``last.ckpt`` of the training run, fully unpickled.

    Examples:
        sorted(last_ckpt(training_output_dir))
        # A pytest fixture over the training run; it cannot run standalone.
    """
    return torch.load(training_output_dir / "last.ckpt", map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def last_ema_pth(training_output_dir: Path) -> dict[str, Any]:
    """The ``last_ema.pth`` of the same run, whose description ``last.ckpt`` must match.

    Examples:
        last_ema_pth(training_output_dir)["model_name"]
        # A pytest fixture over the training run; it cannot run standalone.
    """
    return torch.load(training_output_dir / "last_ema.pth", map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def reloaded(training_output_dir: Path) -> RFDETR:
    """``last.ckpt`` reloaded on the CPU through ``RFDETR.from_checkpoint``.

    Examples:
        reloaded(training_output_dir).model_config.resolution
        # A pytest fixture over the training run; it cannot run standalone.
    """
    return RFDETR.from_checkpoint(training_output_dir / "last.ckpt", device="cpu")


@pytest.fixture(scope="module")
def constructed(training_output_dir: Path) -> RFDETRNano:
    """``last.ckpt`` passed as ``pretrain_weights`` to the class it was trained with.

    Examples:
        constructed(training_output_dir).class_names
        # A pytest fixture over the training run; it cannot run standalone.
    """
    return RFDETRNano(pretrain_weights=str(training_output_dir / "last.ckpt"), resolution=_RESOLUTION, device="cpu")


class TestLastCkptDescribesModel:
    """``last.ckpt`` carries the same model description as the best ``.pth`` files of its run."""

    @pytest.mark.parametrize("key", ["args", "model_name", "model_config", "rfdetr_version"])
    def test_matches_best_pth(self, last_ckpt: dict[str, Any], last_ema_pth: dict[str, Any], key: str) -> None:
        """Each description key equals the one ``BestModelCallback`` wrote for the same epoch."""
        assert last_ckpt[key] == last_ema_pth[key]

    def test_description_survives_weights_only_load(self, training_output_dir: Path) -> None:
        """Lightning resumes ``ckpt_path`` with ``torch.load(weights_only=True)``; the description must load there."""
        checkpoint = torch.load(training_output_dir / "last.ckpt", map_location="cpu", weights_only=True)
        assert checkpoint["model_config"]["resolution"] == _RESOLUTION

    @pytest.mark.parametrize("filename", ["last.ckpt", "last_ema.pth"])
    def test_unreadable_note_is_stored_as_repr(self, training_output_dir: Path, filename: str) -> None:
        """A ``Path`` note reaches every file of the run as its ``repr``, so a weights-only load reads them all.

        ``.ckpt`` and ``.pth`` are written by different callers of the same description helper. While only the ``.ckpt``
        caller replaced what a weights-only load rejects, ``last_ema.pth`` kept the raw ``Path`` and could only be
        reloaded by falling back to full pickling.
        """
        checkpoint = torch.load(training_output_dir / filename, map_location="cpu", weights_only=True)
        assert checkpoint["args"]["notes"] == repr(_NOTES)


class TestFromCheckpointOnLastCkpt:
    """``RFDETR.from_checkpoint("last.ckpt")`` rebuilds the model that was trained."""

    def test_restores_model_class(self, reloaded: RFDETR) -> None:
        """The class comes from the checkpoint, not from its file name, which names no model."""
        assert type(reloaded) is RFDETRNano

    def test_restores_resolution(self, reloaded: RFDETR) -> None:
        """The trained resolution comes back instead of the class default."""
        assert reloaded.model_config.resolution == _RESOLUTION

    def test_loads_ckpt_weights(self, reloaded: RFDETR, last_ckpt: dict[str, Any]) -> None:
        """The reloaded model holds exactly the weights stored in ``last.ckpt``."""
        state_dict = reloaded.model.model.state_dict()
        saved = {
            key.removeprefix("model."): value
            for key, value in last_ckpt["state_dict"].items()
            if key.startswith("model.")
        }
        assert [key for key, value in saved.items() if not torch.equal(state_dict[key], value)] == []


class TestLastCkptKeepsDatasetClassNames:
    """Either way of loading ``last.ckpt`` returns the dataset's class names, not COCO's (the #509 symptom)."""

    @pytest.mark.parametrize("model_fixture", ["reloaded", "constructed"])
    def test_class_names(self, request: pytest.FixtureRequest, dataset_dir: Path, model_fixture: str) -> None:
        """The class names are the dataset's categories, in id order."""
        categories = json.loads((dataset_dir / "train" / "_annotations.coco.json").read_text())["categories"]
        expected = [category["name"] for category in sorted(categories, key=lambda category: category["id"])]
        assert request.getfixturevalue(model_fixture).class_names == expected


class _RaisingRepr:
    """A note whose ``__repr__`` raises, standing in for a user object whose own debugging code is broken.

    Module-level so ``torch.save`` can look the class up by name; the checkpoint writers must survive it either way.
    """

    def __repr__(self) -> str:
        """Raise, the way a half-initialised object's ``__repr__`` does."""
        raise RuntimeError("this object cannot describe itself")


def _save_hook_module(model_config: Any = None, **train_overrides: Any) -> SimpleNamespace:
    """Stand in for the module ``on_save_checkpoint`` reads: a trainer without datamodule plus the two configs.

    Args:
        model_config: Model config to hold, or ``None`` for a CPU Nano config without pretrained weights.
        **train_overrides: Train config fields to override.

    Examples:
        >>> _save_hook_module(notes="n").train_config.notes
        'n'
    """
    if model_config is None:
        model_config = RFDETRNanoConfig(pretrain_weights=None, device="cpu")
    return SimpleNamespace(
        trainer=SimpleNamespace(datamodule=None),
        model_config=model_config,
        train_config=TrainConfig(dataset_dir="dataset", **train_overrides),
    )


class TestOnSaveCheckpoint:
    """``RFDETRModelModule.on_save_checkpoint`` on inputs the one-epoch CPU run does not produce."""

    def test_syncs_num_classes_from_compiled_weights(self) -> None:
        """The saved ``num_classes`` follows the weights ``torch.compile`` nests under ``model._orig_mod.``."""
        module = _save_hook_module(RFDETRNanoConfig(pretrain_weights=None, num_classes=5, device="cpu"))
        checkpoint = {"state_dict": {"model._orig_mod.class_embed.weight": torch.zeros(4, 8)}}

        RFDETRModelModule.on_save_checkpoint(module, checkpoint)

        assert checkpoint["model_config"]["num_classes"] == 3

    @pytest.mark.parametrize(
        "notes",
        [
            pytest.param(PurePosixPath("runs/exp7"), id="path"),
            pytest.param({"started": datetime.date(2026, 9, 29)}, id="dict-with-date"),
        ],
    )
    def test_stores_what_a_weights_only_load_rejects_as_repr(self, notes: object) -> None:
        """Lightning resumes ``ckpt_path`` weights-only, so ``notes`` it cannot read must not reach the file as is."""
        module = _save_hook_module(notes=notes)
        checkpoint: dict[str, Any] = {"state_dict": {}}
        with pytest.warns(UserWarning, match="'notes'"):
            RFDETRModelModule.on_save_checkpoint(module, checkpoint)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)

        assert torch.load(buffer, weights_only=True)["args"]["notes"] == repr(notes)

    def test_value_whose_repr_raises_is_named_by_type(self) -> None:
        """A note that cannot even describe itself is replaced, not re-raised, so the epoch's write still finishes.

        The ``repr`` fallback is the last resort for a value no checkpoint can hold; calling it unguarded made a broken
        ``__repr__`` abort every checkpoint write of the run instead of costing one field of provenance.
        """
        module = _save_hook_module(notes=_RaisingRepr())
        checkpoint: dict[str, Any] = {"state_dict": {}}

        with pytest.warns(UserWarning, match="'notes'"):
            RFDETRModelModule.on_save_checkpoint(module, checkpoint)

        assert checkpoint["args"]["notes"] == "<unrepresentable _RaisingRepr>"

    def test_names_a_deprecated_config_instead_of_refusing_to_save(self) -> None:
        """A ``Trainer`` without ``BestModelCallback`` still writes a checkpoint for a deprecated config.

        ``RFDETRLargeDeprecated`` is supported until 2.0 and ``from_checkpoint`` resolves that name, so this hook — the
        only thing between a plain ``Trainer`` and its ``.ckpt`` — records it. The callback's own ``.pth`` files keep
        refusing it; refusing here made such a run unable to save at all.
        """
        module = _save_hook_module(RFDETRLargeDeprecatedConfig(model_name=None, pretrain_weights=None, device="cpu"))
        checkpoint: dict[str, Any] = {"state_dict": {}}

        RFDETRModelModule.on_save_checkpoint(module, checkpoint)

        assert checkpoint["model_name"] == "RFDETRLargeDeprecated"


class _RecordLastCheckpointEpoch(Callback):
    """At every train epoch start, record the ``epoch`` stored in ``last.ckpt`` (``None`` if absent).

    Checkpoint callbacks run after every other callback's ``on_train_epoch_end``, so the previous epoch's write is only
    observable from the next epoch's start hook.
    """

    def __init__(self, output_dir: Path) -> None:
        self.last_path = output_dir / "last.ckpt"
        self.epochs: list[int | None] = []
        self.current_epoch = 0

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Track the active epoch and append the prior checkpoint epoch."""
        self.current_epoch = trainer.current_epoch
        if trainer.current_epoch == 0:
            return
        if not self.last_path.exists():
            self.epochs.append(None)
            return
        self.epochs.append(int(torch.load(self.last_path, map_location="cpu", weights_only=False)["epoch"]))


def _fit(
    tmp_path: Path,
    epochs: int,
    eval_interval: int,
    checkpoint_interval: int,
    improve_validation: bool = False,
) -> tuple[Path, list[int | None]]:
    """Fit ``epochs`` epochs of two batches each and return the output dir plus the recorded ``last.ckpt`` epochs.

    When ``improve_validation`` is true, predictions miss before the final epoch and match exactly in its validation.

    Examples:
        >>> import contextlib, io
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as d:
        ...     with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ...         out, epochs = _fit(Path(d), epochs=2, eval_interval=1, checkpoint_interval=2)
        ...     sorted(p.name for p in out.glob("checkpoint_*.ckpt")), epochs
        (['checkpoint_1.ckpt'], [0])
    """
    mc = RFDETRBaseConfig(pretrain_weights=None, device="cpu", num_classes=3)
    tc = TrainConfig(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=epochs,
        batch_size=2,
        num_workers=0,
        eval_interval=eval_interval,
        checkpoint_interval=checkpoint_interval,
        use_ema=False,
        tensorboard=False,
        run_test=False,
    )
    recorder = _RecordLastCheckpointEpoch(Path(tc.output_dir))
    postprocess = MagicMock(side_effect=_fake_postprocess)
    if improve_validation:
        postprocess = MagicMock(
            side_effect=lambda outputs, orig_sizes: [
                {
                    "boxes": torch.tensor(
                        [[14.4, 14.4, 17.6, 17.6]] if recorder.current_epoch == epochs - 1 else [[5.0, 5.0, 20.0, 20.0]]
                    ),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                }
                for _ in range(orig_sizes.shape[0])
            ]
        )
    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), postprocess),
        ),
        patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=20)),
        patch(
            "rfdetr.training.module_model.get_param_dict",
            side_effect=lambda args, model: _make_param_dicts(model),
        ),
    ):
        module = RFDETRModelModule(mc, tc)
        datamodule = RFDETRDataModule(mc, tc)
        trainer = build_trainer(
            tc,
            mc,
            accelerator="cpu",
            limit_train_batches=2,
            limit_val_batches=2,
            log_every_n_steps=1,
        )
        trainer.callbacks.append(recorder)
        trainer.fit(module, datamodule=datamodule)
    return Path(tc.output_dir), recorder.epochs


class TestCheckpointCadenceWithEvalInterval:
    """``eval_interval`` must not change which epochs write the resume and archive checkpoints."""

    def test_last_ckpt_is_rewritten_every_epoch(self, tmp_path):
        """With eval_interval=2, last.ckpt at the start of epoch N must hold epoch N-1 for every N."""
        _, epochs = _fit(tmp_path, epochs=6, eval_interval=2, checkpoint_interval=3)

        assert epochs == [0, 1, 2, 3, 4]

    def test_interval_archives_land_on_every_interval_epoch(self, tmp_path):
        """With checkpoint_interval=3 and eval_interval=2, epoch 2 (not an eval epoch) must still be archived."""
        out, _ = _fit(tmp_path, epochs=6, eval_interval=2, checkpoint_interval=3)

        assert sorted(p.name for p in out.glob("checkpoint_*.ckpt")) == ["checkpoint_2.ckpt", "checkpoint_5.ckpt"]

    def test_last_ckpt_carries_that_epochs_validation_state(self, tmp_path):
        """Saving on train epoch end must still capture the validation that ran inside the same epoch."""
        out, _ = _fit(tmp_path, epochs=2, eval_interval=1, checkpoint_interval=5, improve_validation=True)

        state = torch.load(out / "last.ckpt", map_location="cpu", weights_only=False)
        best_states = [value for key, value in state["callbacks"].items() if key.startswith("BestModelCallback")]
        assert best_states
        assert state["epoch"] == 1
        torch.testing.assert_close(best_states[0]["best_model_score"], torch.tensor(1.0), rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("eval_interval", [1, 2])
    def test_final_epoch_is_saved(self, tmp_path, eval_interval):
        """The last epoch is always archived, whether or not eval_interval divides the epoch count."""
        out, _ = _fit(tmp_path, epochs=3, eval_interval=eval_interval, checkpoint_interval=3)

        assert (out / "checkpoint_2.ckpt").exists()
        assert int(torch.load(out / "last.ckpt", map_location="cpu", weights_only=False)["epoch"]) == 2
