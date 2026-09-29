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
"""

import datetime
import io
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rfdetr import RFDETR, RFDETRNano
from rfdetr.config import RFDETRLargeDeprecatedConfig, RFDETRNanoConfig, TrainConfig
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.reproducibility import seed_all
from tests.conftest import build_synthetic_dataset

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
        >>> dataset_dir(tmp_path_factory)  # doctest: +SKIP
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
        >>> training_output_dir(dataset_dir, tmp_path_factory)  # doctest: +SKIP
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
        >>> sorted(last_ckpt(training_output_dir))  # doctest: +SKIP
        # A pytest fixture over the training run; it cannot run standalone.
    """
    return torch.load(training_output_dir / "last.ckpt", map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def last_ema_pth(training_output_dir: Path) -> dict[str, Any]:
    """The ``last_ema.pth`` of the same run, whose description ``last.ckpt`` must match.

    Examples:
        >>> last_ema_pth(training_output_dir)["model_name"]  # doctest: +SKIP
        # A pytest fixture over the training run; it cannot run standalone.
    """
    return torch.load(training_output_dir / "last_ema.pth", map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def reloaded(training_output_dir: Path) -> RFDETR:
    """``last.ckpt`` reloaded on the CPU through ``RFDETR.from_checkpoint``.

    Examples:
        >>> reloaded(training_output_dir).model_config.resolution  # doctest: +SKIP
        # A pytest fixture over the training run; it cannot run standalone.
    """
    return RFDETR.from_checkpoint(training_output_dir / "last.ckpt", device="cpu")


@pytest.fixture(scope="module")
def constructed(training_output_dir: Path) -> RFDETRNano:
    """``last.ckpt`` passed as ``pretrain_weights`` to the class it was trained with.

    Examples:
        >>> constructed(training_output_dir).class_names  # doctest: +SKIP
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


class TestOnSaveCheckpoint:
    """``RFDETRModelModule.on_save_checkpoint`` on inputs the one-epoch CPU run does not produce."""

    def test_syncs_num_classes_from_compiled_weights(self) -> None:
        """The saved ``num_classes`` follows the weights ``torch.compile`` nests under ``model._orig_mod.``."""
        module = SimpleNamespace(
            trainer=SimpleNamespace(datamodule=None),
            model_config=RFDETRNanoConfig(pretrain_weights=None, num_classes=5, device="cpu"),
            train_config=TrainConfig(dataset_dir="dataset"),
        )
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
        module = SimpleNamespace(
            trainer=SimpleNamespace(datamodule=None),
            model_config=RFDETRNanoConfig(pretrain_weights=None, device="cpu"),
            train_config=TrainConfig(dataset_dir="dataset", notes=notes),
        )
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
        module = SimpleNamespace(
            trainer=SimpleNamespace(datamodule=None),
            model_config=RFDETRNanoConfig(pretrain_weights=None, device="cpu"),
            train_config=TrainConfig(dataset_dir="dataset", notes=_RaisingRepr()),
        )
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
        module = SimpleNamespace(
            trainer=SimpleNamespace(datamodule=None),
            model_config=RFDETRLargeDeprecatedConfig(model_name=None, pretrain_weights=None, device="cpu"),
            train_config=TrainConfig(dataset_dir="dataset"),
        )
        checkpoint: dict[str, Any] = {"state_dict": {}}

        RFDETRModelModule.on_save_checkpoint(module, checkpoint)

        assert checkpoint["model_name"] == "RFDETRLargeDeprecated"
