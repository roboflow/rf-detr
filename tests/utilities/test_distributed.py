# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for distributed utility helpers."""

from unittest.mock import patch

import pytest
import torch

from rfdetr.utilities.distributed import all_gather


def _fake_all_gather(output_tensors, input_tensor) -> None:
    """Stand-in for dist.all_gather: broadcast the local tensor to every output slot."""
    for out in output_tensors:
        out.copy_(input_tensor)


def test_all_gather_supports_cpu_without_tensor_truthiness_error() -> None:
    """all_gather derives a cpu device and works when the backend is not nccl (e.g. gloo)."""
    with (
        patch("rfdetr.utilities.distributed.get_world_size", return_value=2),
        patch("rfdetr.utilities.distributed.dist.all_gather", side_effect=_fake_all_gather),
        patch("rfdetr.utilities.distributed.is_dist_avail_and_initialized", return_value=True),
        patch("rfdetr.utilities.distributed.dist.get_backend", return_value="gloo"),
    ):
        result = all_gather({"value": 7})

    assert result == [{"value": 7}, {"value": 7}]


def test_all_gather_explicit_device_bypasses_backend_probe() -> None:
    """A caller-supplied device (e.g. an XLA device) is used as-is; no backend derivation runs."""
    with (
        patch("rfdetr.utilities.distributed.get_world_size", return_value=2),
        patch("rfdetr.utilities.distributed.dist.all_gather", side_effect=_fake_all_gather) as mock_all_gather,
        patch("rfdetr.utilities.distributed.dist.get_backend") as mock_get_backend,
    ):
        result = all_gather({"value": 7}, device=torch.device("cpu"))

    mock_get_backend.assert_not_called()
    input_tensor = mock_all_gather.call_args.args[1]
    assert input_tensor.device == torch.device("cpu")
    assert result == [{"value": 7}, {"value": 7}]


def test_all_gather_explicit_device_wins_over_cuda_heuristic() -> None:
    """Explicit device= is still honored when the cuda-available heuristic would otherwise pick cuda.

    Guards against a regression that silently ignores ``device=`` and falls back to the cuda-if-available-else-cpu
    heuristic -- such a bug would coincidentally pass on CPU-only CI (where cuda is never available) without this
    explicit ``is_available=True`` override.
    """
    with (
        patch("rfdetr.utilities.distributed.get_world_size", return_value=2),
        patch("rfdetr.utilities.distributed.dist.all_gather", side_effect=_fake_all_gather) as mock_all_gather,
        patch("rfdetr.utilities.distributed.torch.cuda.is_available", return_value=True),
    ):
        all_gather({"value": 7}, device=torch.device("cpu"))

    input_tensor = mock_all_gather.call_args.args[1]
    assert input_tensor.device == torch.device("cpu")


def _xla_all_gather_worker(_local_index: int) -> None:
    """Per-process body for ``xmp.spawn``/``torch_xla.launch`` -- must stay module-level (picklable) not a closure.

    PJRT's multi-process spawn dispatches through ``concurrent.futures.ProcessPoolExecutor``, which pickles the target
    with stdlib ``pickle``; a nested function fails with ``AttributeError: Can't pickle local object``.
    """
    import torch.distributed as dist
    import torch_xla
    import torch_xla.runtime as xr

    dist.init_process_group("xla", init_method="xla://")
    device = torch_xla.device()
    world_size = xr.world_size()
    result = all_gather({"rank": xr.global_ordinal()}, device=device)
    assert len(result) == world_size
    assert {item["rank"] for item in result} == set(range(world_size))


@pytest.mark.xla
def test_all_gather_multiprocess_xla_collective_routing() -> None:
    """all_gather(device=<xla device>) round-trips per-rank data through ProcessGroupXla under real multiprocess XLA.

    T1-mp lane (plan Sec 1.3): validates Task 1.3's fix -- routing all_gather's intermediate byte tensors through an
    explicit device instead of the cuda-if-available-else-cpu heuristic -- against a real ``torch_xla`` multi-process
    collective, not a mock. ``dist.init_process_group("xla", init_method="xla://")`` collectives are TPU/NEURON-only
    in torch_xla r2.9 -- confirmed against ``torch_xla/_internal/pjrt.py``'s ``run_multiprocess`` (CPU falls through
    to ``num_processes = 1``, thread-fanout not real multiprocess) and torch_xla's own
    ``test/torch_distributed/test_torch_distributed_all_gather_xla_backend.py``, which guards the identical
    collective with ``xm.xla_device_hw(device) in ("TPU", "NEURON")`` and no-ops otherwise. No CPU-PJRT path exists,
    so this skips off real TPU/NEURON hardware; real multi-replica proof is Phase 2b's Kaggle TPU smoke test
    (``notebooks/tpu_phase2b_kaggle_smoke.py``).
    """
    pytest.importorskip("torch_xla")

    from torch_xla import runtime as xr

    if xr.device_type() not in ("TPU", "NEURON"):
        pytest.skip(
            "ProcessGroupXla xla:// collectives are TPU/NEURON-only (torch_xla r2.9); "
            "no CPU-PJRT path exists. Real proof: Phase 2b Kaggle TPU smoke test."
        )

    import torch_xla

    torch_xla.launch(_xla_all_gather_worker)
