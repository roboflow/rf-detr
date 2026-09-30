# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the lazy device move running under ``torch.inference_mode()``.

``predict()`` stacks ``@torch.inference_mode()`` on top of ``@_ensure_model_on_device``, so the deferred CPU-to-
accelerator move happens while inference mode is active.  Tensors materialised under inference mode are *inference
tensors*: they can never require gradients, so a later ``train()`` / auto-batch probe silently produces no gradients.
The move itself must therefore always run with inference mode disabled.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import torch
from torch import nn

from rfdetr.detr import _device_move_lock, _locked_move, _move_model_context_to_device


class _RecordingModule(nn.Module):
    """Module whose ``to()`` records whether inference mode was active at move time."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.inference_mode_at_move: bool | None = None

    def to(self, *args: Any, **kwargs: Any) -> "_RecordingModule":
        """Record the inference-mode state instead of performing a real device move."""
        self.inference_mode_at_move = torch.is_inference_mode_enabled()
        return self


class TestMoveModelContextUnderInferenceMode:
    """The deferred device move must never materialise parameters as inference tensors."""

    def test_moved_params_are_not_inference_tensors(self) -> None:
        """A real ``.to()`` move inside ``torch.inference_mode()`` must not create inference-tensor parameters."""
        ctx = SimpleNamespace(device=torch.device("meta"), model=nn.Linear(2, 2))

        with torch.inference_mode():
            _move_model_context_to_device(ctx)

        assert not any(p.is_inference() for p in ctx.model.parameters())

    def test_move_still_materializes_on_target_device(self) -> None:
        """The inference-mode guard must not suppress the device move itself."""
        ctx = SimpleNamespace(device=torch.device("meta"), model=nn.Linear(2, 2))

        with torch.inference_mode():
            _move_model_context_to_device(ctx)

        assert all(p.device.type == "meta" for p in ctx.model.parameters())

    def test_move_runs_with_inference_mode_disabled(self) -> None:
        """The ``.to()`` call itself must observe inference mode as disabled."""
        module = _RecordingModule()
        ctx = SimpleNamespace(device=torch.device("meta"), model=module)

        with torch.inference_mode():
            _move_model_context_to_device(ctx)

        assert module.inference_mode_at_move is False


class _LockObservingModule(nn.Module):
    """Module whose ``to()`` records the lock and inference-mode state at move time instead of moving anything."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.lock_held_at_move: bool | None = None
        self.inference_mode_at_move: bool | None = None

    def to(self, *args: Any, **kwargs: Any) -> "_LockObservingModule":
        """Record whether the device-move lock was held and inference mode active, without a real device move."""
        self.lock_held_at_move = _device_move_lock(self).locked()
        self.inference_mode_at_move = torch.is_inference_mode_enabled()
        return self


class TestLockedMove:
    """``_locked_move`` serialises a caller-driven move of the live module against the deferred first-use move.

    ``export()`` and ``evaluate()`` move the live module to CPU and back around their own work. Those moves rewrite the
    same parameter storage that a first ``predict()`` on another thread may be moving to the accelerator, so they have
    to take the same lock as the deferred move rather than running unsynchronised beside it.
    """

    def test_move_holds_the_device_move_lock(self) -> None:
        """The ``.to()`` call itself must observe the module's device-move lock as held."""
        module = _LockObservingModule()

        _locked_move(module, torch.device("meta"))

        assert module.lock_held_at_move is True

    def test_lock_is_released_after_the_move(self) -> None:
        """The lock must not be left held once the move returns, or every later caller would deadlock."""
        module = _LockObservingModule()

        _locked_move(module, torch.device("meta"))

        assert not _device_move_lock(module).locked()

    def test_move_runs_with_inference_mode_disabled(self) -> None:
        """A move requested under ``torch.inference_mode()`` must still run with inference mode disabled."""
        module = _LockObservingModule()

        with torch.inference_mode():
            _locked_move(module, torch.device("meta"))

        assert module.inference_mode_at_move is False


class _FakeParam:
    """Duck-typed stand-in for a parameter: only ``.device`` is read by the guard."""

    def __init__(self, device: torch.device) -> None:
        self.device = device


class _CountingDeviceModule:
    """Duck-typed module stand-in that counts real ``.to()`` calls without touching any accelerator.

    ``_move_model_context_to_device`` only calls ``next(inner.parameters(), None)`` and ``inner.to(target)`` on the
    model context's inner module, so a minimal stand-in exercises the guard logic without requiring a CUDA device to be
    present (this repo's CPU-only CI has none).
    """

    def __init__(self, initial_device: torch.device) -> None:
        self._device = initial_device
        self.to_call_count = 0

    def parameters(self) -> Any:
        """Yield the single fake parameter tracking the module's current device.

        Examples:
            >>> module = _CountingDeviceModule(torch.device("cpu"))
            >>> next(module.parameters()).device
            device(type='cpu')
        """
        yield _FakeParam(self._device)

    def to(self, device: torch.device) -> "_CountingDeviceModule":
        """Record the call and move the fake parameter to *device*."""
        self.to_call_count += 1
        self._device = device
        return self


class TestMoveModelContextIndexNormalization:
    """``torch.device('cuda')`` (no index) must compare equal to the indexed device it resolves to.

    ``model_ctx.device`` is built from a plain ``"cuda"`` string (see ``_build_model_context``), which
    ``torch.device()`` converts to an index-less device. A real parameter's ``.device`` always carries an explicit index
    once placed. Comparing the two with ``!=`` is ``True`` even when they name the same physical GPU, so without index
    normalization the guard below would move the whole model on every single call instead of only the first one that
    actually changes device.
    """

    def test_second_call_skips_redundant_move_when_target_has_no_index(self) -> None:
        """A second call with an index-less target must not re-trigger ``.to()`` once already placed."""
        module = _CountingDeviceModule(initial_device=torch.device("cpu"))
        ctx = SimpleNamespace(device=torch.device("cuda"), model=module)

        with mock.patch("torch.cuda.current_device", return_value=0):
            _move_model_context_to_device(ctx)
            assert module.to_call_count == 1
            assert next(module.parameters()).device == torch.device("cuda", 0)

            _move_model_context_to_device(ctx)

        assert module.to_call_count == 1

    def test_explicit_index_mismatch_still_triggers_move(self) -> None:
        """An explicit different cuda index (multi-GPU) must still trigger a real move."""
        module = _CountingDeviceModule(initial_device=torch.device("cuda", 0))
        ctx = SimpleNamespace(device=torch.device("cuda", 1), model=module)

        with mock.patch("torch.cuda.current_device", return_value=0):
            _move_model_context_to_device(ctx)

        assert module.to_call_count == 1
        assert next(module.parameters()).device == torch.device("cuda", 1)


class TestMoveModelContextCudaIndexMemoisation:
    """The concrete CUDA device is resolved once and recorded, not recomputed per caller.

    ``torch.cuda.current_device()`` is thread-local. Two threads that called ``torch.cuda.set_device()`` differently
    would each resolve an index-less ``"cuda"`` target to their own GPU, each find the weights on the other's, and move
    the whole model back and forth on every ``predict()`` — no corruption, just a silent throughput cliff.
    """

    def test_resolved_index_is_recorded_on_the_context(self) -> None:
        """The index-less target must be replaced by the concrete device the first move resolved."""
        ctx = SimpleNamespace(device=torch.device("cuda"), model=_CountingDeviceModule(torch.device("cpu")))

        with mock.patch("torch.cuda.current_device", return_value=0):
            _move_model_context_to_device(ctx)

        assert ctx.device == torch.device("cuda", 0)

    def test_current_device_is_read_only_once_across_calls(self) -> None:
        """A later call must reuse the recorded device instead of consulting its own thread's selection again."""
        ctx = SimpleNamespace(device=torch.device("cuda"), model=_CountingDeviceModule(torch.device("cpu")))

        with mock.patch("torch.cuda.current_device", side_effect=[0, 1]) as mock_current_device:
            _move_model_context_to_device(ctx)
            _move_model_context_to_device(ctx)

        assert mock_current_device.call_count == 1

    def test_second_call_with_another_thread_selection_does_not_move_again(self) -> None:
        """A caller whose own ``current_device()`` is a different GPU must not bounce the weights onto it."""
        module = _CountingDeviceModule(initial_device=torch.device("cpu"))
        ctx = SimpleNamespace(device=torch.device("cuda"), model=module)

        with mock.patch("torch.cuda.current_device", side_effect=[0, 1]):
            _move_model_context_to_device(ctx)
            _move_model_context_to_device(ctx)

        assert module.to_call_count == 1


class _PartiallyMovingDeviceModule:
    """Duck-typed module whose first ``to()`` moves one of two parameters and then raises, as a mid-move OOM would.

    ``nn.Module.to()`` rewrites parameters one at a time and has no rollback, so a failure part-way through leaves the
    module split across two devices. The retry completes it, mirroring the real per-parameter ``.to()``, which is a no-
    op for a parameter already sitting on the target device.
    """

    def __init__(self, initial_device: torch.device) -> None:
        self._devices = [initial_device, initial_device]
        self.to_call_count = 0

    def parameters(self) -> Any:
        """Yield one fake parameter per tracked device.

        Examples:
            >>> module = _PartiallyMovingDeviceModule(torch.device("cpu"))
            >>> [param.device for param in module.parameters()]
            [device(type='cpu'), device(type='cpu')]
        """
        for device in self._devices:
            yield _FakeParam(device)

    def to(self, device: torch.device) -> "_PartiallyMovingDeviceModule":
        """Move the first parameter, raise on the first call only, and finish the move on the retry."""
        self.to_call_count += 1
        self._devices[0] = device
        if self.to_call_count == 1:
            raise RuntimeError("simulated out-of-memory part-way through the move")
        self._devices[1] = device
        return self


class TestMoveModelContextAfterFailedMove:
    """A move that raised part-way must not leave the module reported as moved.

    The guard decides whether the deferred move still has work to do. If it consults only the first parameter, a half-
    moved module — the state ``nn.Module.to()`` leaves behind when it raises, since it has no rollback — reads as fully
    moved, and the next ``predict()`` runs across two devices instead of finishing the move.
    """

    def test_retry_completes_a_move_that_failed_part_way(self) -> None:
        """After a failed move, the next call must move the parameters that were left behind."""
        module = _PartiallyMovingDeviceModule(initial_device=torch.device("cpu"))
        ctx = SimpleNamespace(device=torch.device("meta"), model=module)
        # Arrange: the failed first move leaves one parameter on the target device and one on CPU.
        with pytest.raises(RuntimeError, match="simulated out-of-memory"):
            _move_model_context_to_device(ctx)

        _move_model_context_to_device(ctx)

        assert all(param.device == torch.device("meta") for param in module.parameters()), (
            "the retry left a parameter behind: the guard treated the half-moved module as already moved"
        )


class _SlowCountingDeviceModule(_CountingDeviceModule):
    """Counting stand-in whose ``.to()`` mimics a real move: the first parameter lands first, the rest follow.

    ``nn.Module.to()`` rewrites parameters one after another, so once the first one reports the target device another
    thread can see "already moved" while later parameters are still in flight. ``completed`` flips only when the whole
    move is done, and the sleep holds the window open long enough for that overlap on a CPU-only runner.
    """

    move_delay_s = 0.2

    def __init__(self, initial_device: torch.device) -> None:
        super().__init__(initial_device)
        self.completed = False

    def to(self, device: torch.device) -> "_SlowCountingDeviceModule":
        """Record the call, expose the target device at once, and finish the move after ``move_delay_s``.

        Examples:
            >>> module = _SlowCountingDeviceModule(torch.device("cpu"))
            >>> module.move_delay_s = 0.0
            >>> module.to(torch.device("meta")) is module
            True
            >>> next(module.parameters()).device, module.completed, module.to_call_count
            (device(type='meta'), True, 1)
        """
        self.to_call_count += 1
        self._device = device
        time.sleep(self.move_delay_s)
        self.completed = True
        return self


class _RaiseOnceDeviceModule(_CountingDeviceModule):
    """Counting stand-in whose first ``.to()`` call raises; every later call performs the move normally.

    Mimics a transient device-move failure (e.g. a CUDA OOM, which subclasses ``RuntimeError``) so the guard's ``with
    _DEVICE_MOVE_LOCK:`` block can be checked for exception safety: the lock must release even when the protected
    ``.to()`` call raises, or every later caller would hang forever waiting to acquire it.
    """

    def __init__(self, initial_device: torch.device) -> None:
        super().__init__(initial_device)
        self._raised = False

    def to(self, device: torch.device) -> "_RaiseOnceDeviceModule":
        """Raise ``RuntimeError`` on the first call; delegate to the counting parent on every later call.

        Examples:
            >>> module = _RaiseOnceDeviceModule(torch.device("cpu"))
            >>> module.to(torch.device("meta"))
            Traceback (most recent call last):
                ...
            RuntimeError: simulated move failure
            >>> next(module.parameters()).device, module.to_call_count
            (device(type='cpu'), 1)
            >>> module.to(torch.device("meta")) is module
            True
            >>> next(module.parameters()).device, module.to_call_count
            (device(type='meta'), 2)
        """
        self.to_call_count += 1
        if not self._raised:
            self._raised = True
            raise RuntimeError("simulated move failure")
        self._device = device
        return self


class TestMoveModelContextConcurrency:
    """Concurrent first calls must move the weights exactly once, and later callers must never deadlock.

    Several channel threads sharing one model can hit the deferred move at the same time. Two overlapping in-place
    ``.to()`` calls on the same module race on the parameter storage and can leave corrupted weights behind without
    raising, so the first move has to be serialised and the later callers must observe it as done — whether they arrive
    after a successful move or after one that failed.
    """

    def test_concurrent_first_calls_move_exactly_once_and_wait_for_completion(self) -> None:
        """Four threads racing on a cold model trigger a single ``.to()``, and none returns before it has finished."""
        module = _SlowCountingDeviceModule(initial_device=torch.device("cpu"))
        ctx = SimpleNamespace(device=torch.device("meta"), model=module)
        barrier = threading.Barrier(4)
        returned_early: list[bool] = []
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait()
                _move_model_context_to_device(ctx)
                returned_early.append(not module.completed)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert len(returned_early) == 4
        assert not any(returned_early)
        assert module.to_call_count == 1
        assert next(module.parameters()).device == torch.device("meta")

    def test_late_thread_after_batch_completes_fast_returns_without_moving_again(self) -> None:
        """A 5th thread joining after the 4-thread batch has finished must not trigger another ``.to()``.

        Arranges a warm model via the same 4-thread batch as above, then starts one more thread once that batch has
        fully joined. The lock must let this late arrival see "already moved" and return immediately instead of blocking
        on a real move — a bounded ``join`` proves that fast return rather than merely hoping for it.
        """
        module = _SlowCountingDeviceModule(initial_device=torch.device("cpu"))
        ctx = SimpleNamespace(device=torch.device("meta"), model=module)
        barrier = threading.Barrier(4)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait()
                _move_model_context_to_device(ctx)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert module.to_call_count == 1

        late_thread = threading.Thread(target=_move_model_context_to_device, args=(ctx,))
        late_thread.start()
        late_thread.join(timeout=5)

        assert not late_thread.is_alive(), "late caller fast-return took longer than the bounded join"
        assert module.to_call_count == 1

    def test_move_failure_propagates_and_next_call_retries_without_deadlock(self) -> None:
        """A failed ``.to()`` must propagate undiminished, and the lock it held must release for the next caller.

        If the guard's lock leaked when the protected ``.to()`` raised, the retry below would hang forever instead of
        acquiring it and completing — a bounded ``join`` turns that hang into a fast, deterministic assertion failure
        rather than a 60s CI timeout.
        """
        module = _RaiseOnceDeviceModule(initial_device=torch.device("cpu"))
        ctx = SimpleNamespace(device=torch.device("meta"), model=module)

        with pytest.raises(RuntimeError, match="simulated move failure"):
            _move_model_context_to_device(ctx)

        assert module.to_call_count == 1
        assert next(module.parameters()).device == torch.device("cpu")

        retry_thread = threading.Thread(target=_move_model_context_to_device, args=(ctx,))
        retry_thread.start()
        retry_thread.join(timeout=5)

        assert not retry_thread.is_alive(), "retry after a failed move deadlocked instead of completing"
        assert module.to_call_count == 2
        assert next(module.parameters()).device == torch.device("meta")


class _BlockingDeviceModule(_CountingDeviceModule):
    """Counting stand-in whose ``.to()`` parks until the test releases it, holding one move open deterministically.

    Lets a test keep one model's move in flight while a second, unrelated model is moved, with no timing guesses: the
    ``started`` event says the move is under way, ``release`` decides when it ends, and ``completed`` records that it
    did. The wait is bounded so a regression stalls the test instead of hanging the suite.
    """

    def __init__(self, initial_device: torch.device) -> None:
        super().__init__(initial_device)
        self.started = threading.Event()
        self.release = threading.Event()
        self.completed = False

    def to(self, device: torch.device) -> "_BlockingDeviceModule":
        """Announce that the move started, wait to be released, then perform it.

        Examples:
            >>> module = _BlockingDeviceModule(torch.device("cpu"))
            >>> module.release.set()
            >>> module.to(torch.device("meta")) is module
            True
            >>> next(module.parameters()).device, module.completed
            (device(type='meta'), True)
        """
        self.started.set()
        self.release.wait(timeout=5)
        super().to(device)
        self.completed = True
        return self


class TestMoveModelContextPerModuleLock:
    """The move lock belongs to the module, so unrelated models never queue behind each other.

    One process-wide lock made a cold move of one model stall every other model's already-warm guard for the whole
    transfer. The lock is keyed on the module because the module owns the raced parameter storage — keying it on the
    ``ModelContext`` instead would hand two wrappers of one module two different locks over the same tensors.
    """

    def test_a_move_in_flight_does_not_block_another_model(self) -> None:
        """A second, unrelated model must move while the first model's move is still in flight."""
        blocked = _BlockingDeviceModule(initial_device=torch.device("cpu"))
        other = _CountingDeviceModule(initial_device=torch.device("cpu"))
        blocked_ctx = SimpleNamespace(device=torch.device("meta"), model=blocked)
        other_ctx = SimpleNamespace(device=torch.device("meta"), model=other)
        mover = threading.Thread(target=_move_model_context_to_device, args=(blocked_ctx,))
        mover.start()
        assert blocked.started.wait(timeout=5), "precondition: the first model's move never started"

        _move_model_context_to_device(other_ctx)
        moved_while_first_in_flight = not blocked.completed

        blocked.release.set()
        mover.join(timeout=5)
        assert moved_while_first_in_flight, (
            "the second model's move only completed after the first model's — unrelated models are sharing one lock"
        )

    def test_two_contexts_wrapping_one_module_still_move_it_once(self) -> None:
        """Two contexts around the same module must serialise: they share one module, hence one lock."""
        module = _SlowCountingDeviceModule(initial_device=torch.device("cpu"))
        contexts = [SimpleNamespace(device=torch.device("meta"), model=module) for _ in range(2)]
        barrier = threading.Barrier(len(contexts))

        def worker(ctx: SimpleNamespace) -> None:
            barrier.wait()
            _move_model_context_to_device(ctx)

        threads = [threading.Thread(target=worker, args=(ctx,)) for ctx in contexts]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert module.to_call_count == 1, (
            f"the shared module was moved {module.to_call_count} times: each context took its own lock"
        )
