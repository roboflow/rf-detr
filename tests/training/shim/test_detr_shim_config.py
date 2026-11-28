# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the training_config.json RFDETR.train() writes: ``_save_training_config`` and start-of-run class names."""

import json
import logging
import os
import stat
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pytest_mock import MockerFixture

from rfdetr.config import RFDETRKeypointPreviewConfig
from rfdetr.datasets.webdataset.index import ShardIndex, index_name
from rfdetr.datasets.webdataset.load import WebDatasetDetection
from rfdetr.detr import RFDETR, _save_training_config
from rfdetr.detr import logger as detr_logger
from tests._markers import _requires_posix_permissions
from tests.training.shim.helpers import (
    _DATASET_READ_ERRORS,
    _make_model_config,
    _make_rfdetr_self,
    _make_train_config,
    _read_training_config,
)


def _count_config_write_warnings(records: list[logging.LogRecord]) -> int:
    """Count the distinct ``training_config.json`` warnings among *records*.

    pytest installs its capture handler on the non-propagating ``rf-detr`` logger *and* on the root logger, so
    once a test forces ``propagate = True`` every emission is appended to ``caplog.records`` twice. Both entries
    are the same ``LogRecord`` object, so counting distinct objects recovers how many writes actually failed.

    Examples:
        >>> def _record(message):
        ...     record = logging.LogRecord("rf-detr", logging.WARNING, __file__, 0, message, (), None)
        ...     record.message = record.getMessage()
        ...     return record
        >>> warned, unrelated = _record("Could not save training_config.json to /out."), _record("something else")
        >>> _count_config_write_warnings([warned, warned, unrelated])
        1
        >>> _count_config_write_warnings([])
        0
    """
    return len(
        {id(record) for record in records if record.levelname == "WARNING" and "training_config.json" in record.message}
    )


class _UnevaluableValue:
    """A value that raises when its truthiness is tested, as the payload's ``num_classes`` branch does.

    Examples:
        >>> bool(_UnevaluableValue())
        Traceback (most recent call last):
        ...
        RuntimeError: cannot evaluate truthiness
    """

    def __bool__(self) -> bool:
        raise RuntimeError("cannot evaluate truthiness")


class _UnserializableValue:
    """A value ``json.dumps`` cannot write even with ``default=str``, because coercing it raises.

    Examples:
        >>> str(_UnserializableValue())
        Traceback (most recent call last):
        ...
        RuntimeError: cannot stringify
    """

    def __str__(self) -> str:
        raise RuntimeError("cannot stringify")


_TRAINING_CONFIG_KEYS = {"train_config", "model_config", "model_config_type", "class_names", "num_classes"}


# ---------------------------------------------------------------------------
# TestSaveTrainingConfig
# ---------------------------------------------------------------------------


class TestSaveTrainingConfig:
    """RFDETR.train() writes training_config.json to output_dir when training starts and again when it ends."""

    def _run_train(self, tmp_path, patch_lit, class_names=None, **train_overrides):
        """Run RFDETR.train() with patched PTL; return (mock_self, output_dir path).

        class_names is injected via the datamodule mock (the path RFDETR.train uses to sync self.model.class_names after
        trainer.fit completes).
        """
        if class_names is None:
            class_names = ["cat", "dog", "bird"]
        mock_self = _make_rfdetr_self(tmp_path, **train_overrides)
        _, dmcls, _ = patch_lit
        dmcls.return_value.class_names = class_names
        RFDETR.train(mock_self)
        config = mock_self.get_train_config.return_value
        return mock_self, config.output_dir

    def test_training_config_json_created(self, tmp_path, patch_lit):
        """training_config.json is written to output_dir after train() completes."""
        _, output_dir = self._run_train(tmp_path, patch_lit)
        assert os.path.exists(os.path.join(output_dir, "training_config.json"))

    def test_training_config_json_has_required_keys(self, tmp_path, patch_lit):
        """Saved JSON contains all expected top-level keys."""
        _, output_dir = self._run_train(tmp_path, patch_lit)
        with open(os.path.join(output_dir, "training_config.json")) as f:
            saved = json.load(f)
        assert set(saved.keys()) == _TRAINING_CONFIG_KEYS

    def test_training_config_json_class_names_and_num_classes(self, tmp_path, patch_lit):
        """class_names and num_classes in saved JSON match model state after training."""
        _, output_dir = self._run_train(tmp_path, patch_lit)
        with open(os.path.join(output_dir, "training_config.json")) as f:
            saved = json.load(f)
        assert saved["class_names"] == ["cat", "dog", "bird"]
        assert saved["num_classes"] == 3

    def test_model_config_type_reflects_class_name(self, tmp_path, patch_lit):
        """model_config_type field matches the actual model config class name."""
        _, output_dir = self._run_train(tmp_path, patch_lit)
        with open(os.path.join(output_dir, "training_config.json")) as f:
            saved = json.load(f)
        assert saved["model_config_type"] == "RFDETRBaseConfig"

    def test_non_serializable_value_coerced_to_str(self, tmp_path, patch_lit):
        """Non-JSON-serializable values are coerced via default=str, not raising TypeError."""
        mock_self = _make_rfdetr_self(tmp_path)
        _, dmcls, _ = patch_lit
        dmcls.return_value.class_names = [Path("/some/class"), None]
        RFDETR.train(mock_self)  # must not raise TypeError
        config = mock_self.get_train_config.return_value
        assert os.path.exists(os.path.join(config.output_dir, "training_config.json"))

    def test_output_dir_created_when_missing(self, tmp_path, patch_lit):
        """output_dir is created by makedirs if it does not exist before training."""
        nested_dir = str(tmp_path / "new" / "nested" / "output")
        _, output_dir = self._run_train(tmp_path, patch_lit, output_dir=nested_dir)
        assert os.path.exists(os.path.join(output_dir, "training_config.json"))

    def _run_train_capturing_pre_fit(
        self,
        tmp_path: Path,
        patch_lit: tuple[Any, ...],
        dataset_class_names: list[str] | None = None,
        fit_exception: BaseException | None = None,
        post_fit_exception: BaseException | None = None,
        load_classes_patch: Any = None,
        **train_overrides: Any,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Run RFDETR.train() and read training_config.json both at fit() time and after train() returns.

        trainer.fit is a MagicMock, so its side_effect is the only code that runs at the exact moment the start-of-run
        write must already have happened. Returns (pre_fit_payload, final_payload), each None when the file did not
        exist at that point. load_classes_patch defaults to an unreadable dataset so the start-of-run class-name lookup
        is deterministic instead of depending on what tmp_path happens to contain. post_fit_exception, when set, makes
        remove_optimized_model() raise after a successful fit() but before the post-fit write, simulating a
        housekeeping step (distinct from fit() itself) failing on the way to the second write.

        Examples:
            Needs the ``tmp_path`` and ``patch_lit`` fixtures, so it cannot run standalone:

            >>> self._run_train_capturing_pre_fit(tmp_path, patch_lit)  # doctest: +SKIP
        """
        if dataset_class_names is None:
            dataset_class_names = ["cat", "dog", "bird"]
        if load_classes_patch is None:
            load_classes_patch = patch.object(RFDETR, "_load_classes", side_effect=FileNotFoundError("no dataset"))
        mock_self = _make_rfdetr_self(tmp_path, **train_overrides)
        if post_fit_exception is not None:
            mock_self.remove_optimized_model.side_effect = post_fit_exception
        _, dmcls, mock_bt = patch_lit
        dmcls.return_value.class_names = dataset_class_names
        config_path = os.path.join(mock_self.get_train_config.return_value.output_dir, "training_config.json")
        captured = {}

        def _capture(*args: Any, **kwargs: Any) -> None:
            captured["pre_fit"] = _read_training_config(config_path)
            if fit_exception is not None:
                raise fit_exception

        mock_bt.return_value.fit.side_effect = _capture
        expected_exception = fit_exception or post_fit_exception
        with load_classes_patch:
            if expected_exception is None:
                RFDETR.train(mock_self)
            else:
                with pytest.raises(type(expected_exception)):
                    RFDETR.train(mock_self)
        return captured.get("pre_fit"), _read_training_config(config_path)

    def test_training_config_json_written_before_fit(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """The file already exists by the time trainer.fit() is entered."""
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit)
        assert pre_fit is not None

    def test_pre_fit_payload_carries_the_full_schema(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """The start-of-run copy carries the same five keys as the final one, not a reduced subset."""
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit)
        assert set(pre_fit.keys()) == _TRAINING_CONFIG_KEYS

    def test_post_fit_write_replaces_pre_fit_class_names(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """The second write wins: the dataset's class names overwrite the start-of-run ones."""
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["placeholder"])
        _, final = self._run_train_capturing_pre_fit(tmp_path, patch_lit, load_classes_patch=load_classes_patch)
        assert final["class_names"] == ["cat", "dog", "bird"]

    @pytest.mark.parametrize(
        "fit_exception",
        [
            pytest.param(RuntimeError("boom"), id="crash"),
            pytest.param(KeyboardInterrupt(), id="interrupt"),
        ],
    )
    def test_training_config_json_survives_interrupted_fit(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], fit_exception: BaseException
    ) -> None:
        """A run killed inside fit() still leaves a record of how it was configured (#1493)."""
        _, final = self._run_train_capturing_pre_fit(tmp_path, patch_lit, fit_exception=fit_exception)
        assert final is not None

    def test_post_fit_step_failure_keeps_pre_fit_training_config_intact(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """A post-fit step that raises before the final write must not lose the start-of-run copy.

        Unlike test_training_config_json_survives_interrupted_fit, trainer.fit() itself succeeds here --
        remove_optimized_model(), one of the housekeeping steps between fit() returning and the final
        _save_training_config() call, raises instead. The code never reaches the post-fit write, so the on-disk
        file must still hold exactly the pre-fit payload, untouched.
        """
        post_fit_exception = RuntimeError("optimized model teardown failed")
        pre_fit, final = self._run_train_capturing_pre_fit(tmp_path, patch_lit, post_fit_exception=post_fit_exception)
        assert final == pre_fit

    def test_pre_fit_class_names_taken_from_config_when_set(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """An explicit TrainConfig.class_names is recorded in preference to the dataset's."""
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["from-dataset"])
        pre_fit, _ = self._run_train_capturing_pre_fit(
            tmp_path,
            patch_lit,
            load_classes_patch=load_classes_patch,
            class_names=["from-config"],
        )
        assert pre_fit["class_names"] == ["from-config"]

    def test_pre_fit_class_names_read_from_dataset_when_config_has_none(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """Without an explicit setting the label space is read straight off the dataset directory."""
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["cat", "dog"])
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit, load_classes_patch=load_classes_patch)
        assert pre_fit["class_names"] == ["cat", "dog"]

    @pytest.mark.parametrize("exc", _DATASET_READ_ERRORS)
    def test_pre_fit_class_names_null_when_dataset_unreadable(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], exc: Exception
    ) -> None:
        """An unreadable or unsupported dataset layout records null rather than blocking training."""
        load_classes_patch = patch.object(RFDETR, "_load_classes", side_effect=exc)
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit, load_classes_patch=load_classes_patch)
        assert pre_fit["class_names"] is None

    def test_pre_fit_num_classes_zero_when_dataset_unreadable(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """num_classes stays the count of resolved names, so it is 0 when there are none."""
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit)
        assert pre_fit["num_classes"] == 0

    def test_pre_fit_payload_keeps_empty_class_names_list_not_null(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """An explicit empty class_names list is recorded as [], not coerced to null like an unset one.

        TrainConfig(class_names=[]) is not None, so the pre-fit write must record it verbatim rather than falling
        through to the dataset-lookup branch that only fires when class_names is unset -- locking in the [] vs None
        distinction the payload relies on to tell "explicitly no classes" apart from "not yet resolved".
        """
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit, class_names=[])
        assert pre_fit["class_names"] == []
        assert pre_fit["num_classes"] == 0

    def test_pre_fit_write_skipped_off_rank_zero(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], mocker: MockerFixture
    ) -> None:
        """Distributed workers must not race on the file before fit() initializes torch.distributed."""
        mocker.patch("rfdetr.detr._is_launcher_main_process", return_value=False)

        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit)
        assert pre_fit is None

    def test_post_fit_write_keeps_its_own_rank_guard(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], mocker: MockerFixture
    ) -> None:
        """The post-fit write guards on is_main_process() instead, so the launcher guard does not suppress it."""
        mocker.patch("rfdetr.detr._is_launcher_main_process", return_value=False)

        _, final = self._run_train_capturing_pre_fit(tmp_path, patch_lit)
        assert final is not None

    def test_write_failure_does_not_abort_training(
        self,
        tmp_path: Path,
        patch_lit: tuple[Any, ...],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An output_dir that cannot be created warns from both writes and still lets training run."""
        blocked = tmp_path / "blocked"
        blocked.write_text("a regular file where output_dir should be")
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level("WARNING", logger="rf-detr"):
            self._run_train_capturing_pre_fit(tmp_path, patch_lit, output_dir=str(blocked))
        assert _count_config_write_warnings(caplog.records) == 2

    def test_pre_fit_serialization_failure_does_not_abort_training(
        self,
        tmp_path: Path,
        patch_lit: tuple[Any, ...],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A value json refuses to serialize is a warning, not an exception out of train()."""
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=[_UnserializableValue()])
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level("WARNING", logger="rf-detr"):
            self._run_train_capturing_pre_fit(tmp_path, patch_lit, load_classes_patch=load_classes_patch)
        assert _count_config_write_warnings(caplog.records) == 1

    def test_payload_assembly_failure_does_not_abort_training(
        self,
        tmp_path: Path,
        patch_lit: tuple[Any, ...],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Assembling the payload is guarded too, not just writing it: both steps run inside the same try.

        The value is synthetic -- `_load_classes` is annotated `-> list[str]`, so nothing real reaches the assembly step
        this way -- but it is the only handle a test has on that step in isolation.
        """
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=_UnevaluableValue())
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level("WARNING", logger="rf-detr"):
            self._run_train_capturing_pre_fit(tmp_path, patch_lit, load_classes_patch=load_classes_patch)
        assert _count_config_write_warnings(caplog.records) == 1

    def test_post_fit_serialization_failure_does_not_abort_training(
        self,
        tmp_path: Path,
        patch_lit: tuple[Any, ...],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The post-fit write tolerates it too: before this it only caught OSError, and a completed run died here."""
        monkeypatch.setattr(detr_logger, "propagate", True)
        with caplog.at_level("WARNING", logger="rf-detr"):
            self._run_train_capturing_pre_fit(tmp_path, patch_lit, dataset_class_names=[_UnserializableValue()])
        assert _count_config_write_warnings(caplog.records) == 1

    @_requires_posix_permissions
    @pytest.mark.parametrize("process_umask", [0o022, 0o027], indirect=True)
    def test_training_config_json_mode_follows_umask(self, tmp_path: Path, process_umask: int) -> None:
        """training_config.json gets the mode ``open()`` would give it, not the temp file's owner-only ``0o600``."""
        output_dir = tmp_path / "out"
        _save_training_config(_make_train_config(tmp_path, output_dir=str(output_dir)), _make_model_config(), ["cat"])
        assert stat.S_IMODE((output_dir / "training_config.json").stat().st_mode) == 0o666 & ~process_umask

    @_requires_posix_permissions
    @pytest.mark.usefixtures("process_umask")  # the default 0o022, so both modes differ from a fresh file's 0o644
    @pytest.mark.parametrize("mode", [0o600, 0o640])
    def test_rewritten_training_config_json_keeps_existing_mode(self, tmp_path: Path, mode: int) -> None:
        """Rewriting training_config.json keeps the earlier copy's permission bits, as ``open()`` would."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        config_path = output_dir / "training_config.json"
        config_path.write_text("{}")
        config_path.chmod(mode)
        _save_training_config(_make_train_config(tmp_path, output_dir=str(output_dir)), _make_model_config(), ["cat"])
        # The content check guards against a swallowed write failure leaving the old file, and its mode, untouched.
        rewritten = _read_training_config(str(config_path))
        assert (rewritten["class_names"], stat.S_IMODE(config_path.stat().st_mode)) == (["cat"], mode)

    def test_torn_write_keeps_prior_training_config_intact(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """A write that fails partway through must not corrupt the previously saved good copy.

        Pre-seeds output_dir with a valid training_config.json from an earlier run, then makes the new write raise
        OSError right after committing half its bytes to disk -- the shape of a disk-full or killed-process failure mid-
        write. ``_save_training_config``'s ``except Exception`` path must swallow the error without letting it escape,
        and the atomic-write contract (tempfile + os.replace) requires the on-disk file to still parse as the untouched
        prior payload afterward, with the abandoned temp file cleaned up.
        """
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        config_path = output_dir / "training_config.json"
        prior_payload = {"marker": "prior-good-copy", "run": 1}
        config_path.write_text(json.dumps(prior_payload))
        real_fdopen = os.fdopen
        test_thread = threading.current_thread()

        def _torn_temporary_file(*args: Any, **kwargs: Any) -> Any:
            handle = real_fdopen(*args, **kwargs)
            if threading.current_thread() is not test_thread:
                return handle  # The patch is process-wide; only the save under test gets the torn write.
            real_write = handle.write

            def _torn_write(data: str) -> int:
                real_write(data[: len(data) // 2])
                handle.flush()
                raise OSError("disk full mid-write")

            handle.write = _torn_write
            return handle

        mocker.patch("os.fdopen", side_effect=_torn_temporary_file)

        _save_training_config(_make_train_config(tmp_path, output_dir=str(output_dir)), _make_model_config(), ["cat"])

        assert _read_training_config(str(config_path)) == prior_payload
        assert [p.name for p in output_dir.iterdir()] == ["training_config.json"]

    def test_interleaved_writers_leave_valid_json_not_a_byte_level_merge(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Two writers both cleared past the launcher-rank guard must never leave a byte-interleaved file.

        Deterministically reproduces the race the guard exists to prevent, without relying on real OS-thread scheduling
        (rejected in an earlier review round as non-deterministic): writer "a" starts writing, pauses once its first
        half is flushed to disk, writer "b" then opens and fully writes a different, longer payload to the same path,
        and only then does "a" resume and flush its second half. Ordering is pinned with ``threading.Event``s with
        bounded waits, never a bare block. The atomic tempfile+os.replace write makes each writer's file appear whole or
        not at all, so the final file is exactly one writer's payload.
        """
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        config_path = output_dir / "training_config.json"
        payload_a = ["a"]
        payload_b = ["bb"] * 100  # much longer serialized payload than payload_a
        a_paused = threading.Event()
        b_done = threading.Event()
        real_fdopen = os.fdopen
        errors: list[BaseException] = []

        def _paced_temporary_file(*args: Any, **kwargs: Any) -> Any:
            handle = real_fdopen(*args, **kwargs)
            if threading.current_thread().name != "writer-a":
                return handle
            real_write = handle.write

            def _paced_write(data: str) -> int:
                half = len(data) // 2
                written = real_write(data[:half])
                handle.flush()
                a_paused.set()
                assert b_done.wait(timeout=5), "writer b did not finish in time"
                written += real_write(data[half:])
                handle.flush()
                return written

            handle.write = _paced_write
            return handle

        def _writer_a() -> None:
            try:
                _save_training_config(
                    _make_train_config(tmp_path, output_dir=str(output_dir)), _make_model_config(), payload_a
                )
            except BaseException as exc:
                errors.append(exc)

        def _writer_b() -> None:
            try:
                assert a_paused.wait(timeout=5), "writer a did not pause in time"
                _save_training_config(
                    _make_train_config(tmp_path, output_dir=str(output_dir)), _make_model_config(), payload_b
                )
            except BaseException as exc:
                errors.append(exc)
            finally:
                b_done.set()

        thread_a = threading.Thread(target=_writer_a, name="writer-a")
        thread_b = threading.Thread(target=_writer_b, name="writer-b")
        # One patch around both writers' whole lifetime (patching per writer would nest across threads and restore
        # the wrong os.fdopen); the thread-name check in _paced_temporary_file keeps the pacing to writer "a".
        mocker.patch("os.fdopen", side_effect=_paced_temporary_file)
        thread_a.start()
        thread_b.start()
        thread_a.join(timeout=10)
        thread_b.join(timeout=10)

        assert not thread_a.is_alive(), "writer a did not finish in time"
        assert not thread_b.is_alive(), "writer b did not finish in time"
        assert not errors
        on_disk = _read_training_config(str(config_path))
        assert on_disk["class_names"] in (payload_a, payload_b)

    def test_training_config_json_written_before_the_model_is_built(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """The write precedes module construction, so a run that dies loading weights still leaves a record."""
        mock_self = _make_rfdetr_self(tmp_path)
        modcls, _, _ = patch_lit
        modcls.side_effect = RuntimeError("checkpoint is corrupt")
        load_classes_patch = patch.object(RFDETR, "_load_classes", return_value=["cat"])
        with load_classes_patch, pytest.raises(RuntimeError):
            RFDETR.train(mock_self)
        output_dir = mock_self.get_train_config.return_value.output_dir
        assert _read_training_config(os.path.join(output_dir, "training_config.json")) is not None


# ---------------------------------------------------------------------------
# Start-of-run class-name resolution (PR #1496 review follow-ups)
# ---------------------------------------------------------------------------


class TestPreFitClassNamesResolution:
    """The start-of-run training_config.json resolves class names from every layout the datamodule will read.

    ``_run_train_capturing_pre_fit`` is shared with ``TestSaveTrainingConfig`` — it never touches ``self``, so it is
    aliased here rather than inherited, which would re-collect that class's tests under this one.
    """

    _run_train_capturing_pre_fit = TestSaveTrainingConfig._run_train_capturing_pre_fit

    @staticmethod
    def _write_train_shard_index(dataset_dir: Path, category_ids: str, *, version: int | None = None) -> Path:
        """Write a packed ``train`` shard index whose grouping root carries no annotation.

        Only the index is written -- the pre-fit read and ``WebDatasetDetection.__init__`` both stop at the JSON,
        so no shard tar and no ``webdataset`` package is needed. ``version`` overrides the schema stamp to produce
        an index the current reader rejects.

        Examples:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     written = TestPreFitClassNamesResolution._write_train_shard_index(Path(directory), "remap")
            ...     json.loads((written / "train-index.json").read_text())["category_ids"]
            'remap'
        """
        categories = (
            {"id": 0, "name": "root", "supercategory": "none"},
            {"id": 3, "name": "cat", "supercategory": "root"},
            {"id": 9, "name": "dog", "supercategory": "root"},
        )
        index = ShardIndex("train", ("train-000000.tar",), 4, categories, (3, 9), category_ids, (4,))
        payload = index.to_json()
        if version is not None:
            payload["version"] = version
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / index_name("train")).write_text(json.dumps(payload), encoding="utf-8")
        return dataset_dir

    @pytest.mark.parametrize(
        ("category_ids", "expected"),
        [
            pytest.param("remap", ["cat", "dog"], id="remap-drops-the-unannotated-parent"),
            pytest.param("raw", ["root", "", "", "cat", "", "", "", "", "", "dog"], id="raw-keeps-every-id-slot"),
        ],
    )
    def test_pre_fit_class_names_match_the_webdataset_datamodule(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], category_ids: str, expected: list[str]
    ) -> None:
        """A packed directory records the same names the streaming dataset will report after fit()."""
        dataset_dir = self._write_train_shard_index(tmp_path / "ds", category_ids)
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit, dataset_file="webdataset")
        dataset = WebDatasetDetection(dataset_dir, "train", transforms=None)
        assert pre_fit["class_names"] == dataset.class_names == expected

    def test_pre_fit_num_classes_counts_the_shard_index_names(self, tmp_path: Path, patch_lit: tuple[Any, ...]) -> None:
        """num_classes follows the recorded list instead of staying 0 for a packed directory."""
        self._write_train_shard_index(tmp_path / "ds", "remap")
        pre_fit, _ = self._run_train_capturing_pre_fit(tmp_path, patch_lit, dataset_file="webdataset")
        assert pre_fit["num_classes"] == 2

    def test_unreadable_shard_index_records_null_and_does_not_block_training(
        self, tmp_path: Path, patch_lit: tuple[Any, ...]
    ) -> None:
        """An index the reader rejects (ValueError from ShardIndex.from_json) degrades to null, never an exception."""
        self._write_train_shard_index(tmp_path / "ds", "remap", version=99)
        pre_fit, final = self._run_train_capturing_pre_fit(tmp_path, patch_lit, dataset_file="webdataset")
        assert (pre_fit["class_names"], final is not None) == (None, True)

    @staticmethod
    def _train_interrupted_keypoint_run(
        tmp_path: Path, patch_lit: tuple[Any, ...], mocker: MockerFixture, **train_overrides: Any
    ) -> dict:
        """Run train() on a background-first keypoint model whose fit() dies; return the file it left behind.

        The dataset reader is patched to hand back the detection basis ``['person']``, so a recorded null proves the
        keypoint gate bypassed the reader rather than the dataset merely being unreadable.

        Examples:
            Needs the ``tmp_path``, ``patch_lit`` and ``mocker`` fixtures, so it cannot run standalone:

            >>> run = TestPreFitClassNamesResolution._train_interrupted_keypoint_run  # doctest: +SKIP
            >>> run(tmp_path, patch_lit, mocker)  # doctest: +SKIP
        """
        mock_self = _make_rfdetr_self(tmp_path, **train_overrides)
        mock_self.model_config = RFDETRKeypointPreviewConfig(
            pretrain_weights=None, device="cpu", num_keypoints_per_class=[0, 17]
        )
        _, _, mock_bt = patch_lit
        mock_bt.return_value.fit.side_effect = RuntimeError("interrupted")
        mocker.patch.object(RFDETR, "_load_classes", return_value=["person"])
        with pytest.raises(RuntimeError):
            RFDETR.train(mock_self)
        output_dir = mock_self.get_train_config.return_value.output_dir
        return _read_training_config(os.path.join(output_dir, "training_config.json")) or {}

    def test_interrupted_keypoint_run_records_null_class_names(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], mocker: MockerFixture
    ) -> None:
        """A bg-first keypoint run killed in fit() leaves class_names null, not the detection-basis ['person']."""
        final = self._train_interrupted_keypoint_run(tmp_path, patch_lit, mocker)
        assert (final["class_names"], final["num_classes"]) == (None, 0)

    def test_keypoint_run_still_records_an_explicit_config_class_names(
        self, tmp_path: Path, patch_lit: tuple[Any, ...], mocker: MockerFixture
    ) -> None:
        """Only the dataset read is gated: a TrainConfig.class_names the user set is recorded as before."""
        final = self._train_interrupted_keypoint_run(tmp_path, patch_lit, mocker, class_names=["", "person"])
        assert final["class_names"] == ["", "person"]
