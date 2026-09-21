# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for distributed utility helpers."""

import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from rfdetr.utilities.distributed import _is_launcher_main_process, all_gather

_RANK_ENV_VARS = (
    "RANK",
    "LOCAL_RANK",
    "NODE_RANK",
    "SLURM_PROCID",
    "JSM_NAMESPACE_RANK",
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
)


def _minimal_subprocess_env() -> dict[str, str]:
    """Build a child interpreter environment with every launcher rank variable removed.

    Preserves the parent runtime environment so fresh Windows interpreters retain dependency-specific configuration,
    then strips every rank/launcher variable ``_is_launcher_main_process`` or Lightning's own rank resolution reads.
    Each subprocess probe therefore starts from the same non-rank environment as the test process before its case
    applies its own launcher variables on top.

    Returns:
        A fresh environment mapping, safe for a caller to mutate per test case.

    Examples:
        >>> env = _minimal_subprocess_env()
        >>> "PATH" in env
        True
        >>> "RANK" in env
        False
    """
    env = os.environ.copy()
    for rank_var in _RANK_ENV_VARS:
        env.pop(rank_var, None)
    return env


def test_minimal_subprocess_env_preserves_non_rank_runtime_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """The subprocess environment retains inherited runtime configuration but removes launcher ranks."""
    monkeypatch.setenv("APPDATA", "C:\\Users\\runneradmin\\AppData\\Roaming")
    for rank_var in _RANK_ENV_VARS:
        monkeypatch.setenv(rank_var, "1")

    env = _minimal_subprocess_env()

    assert env["APPDATA"] == "C:\\Users\\runneradmin\\AppData\\Roaming"
    assert all(rank_var not in env for rank_var in _RANK_ENV_VARS)


def _fake_all_gather(output_tensors, input_tensor) -> None:
    """Stand-in for dist.all_gather: broadcast the local tensor to every output slot.

    Examples:
        >>> outs = [torch.zeros(2), torch.zeros(2)]
        >>> _fake_all_gather(outs, torch.tensor([1.0, 2.0]))
        >>> outs
        [tensor([1., 2.]), tensor([1., 2.])]
    """
    for out in output_tensors:
        out.copy_(input_tensor)


class TestIsLauncherMainProcess:
    """_is_launcher_main_process() answers from the launcher's environment, before torch.distributed exists."""

    @pytest.mark.parametrize(
        ("rank", "node_rank", "expected"),
        [
            pytest.param(0, None, True, id="rank-zero-node-zero"),
            pytest.param(1, None, False, id="off-rank-zero"),
            pytest.param(0, "1", False, id="rank-zero-secondary-node"),
        ],
    )
    def test_combines_resolved_rank_with_node_rank(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rank: int,
        node_rank: str | None,
        expected: bool,
    ) -> None:
        """The guard is True only when both the resolved rank and NODE_RANK say "main".

        An ordinary single-process run (rank 0, no NODE_RANK) is the one that writes. Any process off rank 0 does not,
        regardless of which launcher variable Lightning folded into rank_zero_only.rank. NODE_RANK is consulted
        separately from that resolved rank because Lightning's own subprocess launcher leaves LOCAL_RANK at 0 on every
        node of a multi-node run, so without the separate check one process per node would pass.
        """
        monkeypatch.setattr(rank_zero_only, "rank", rank)
        if node_rank is None:
            monkeypatch.delenv("NODE_RANK", raising=False)
        else:
            monkeypatch.setenv("NODE_RANK", node_rank)

        assert _is_launcher_main_process() is expected


class TestIsLauncherMainProcessEnvPrecedence:
    """_is_launcher_main_process(), probed via real subprocesses, honors Lightning's env-var precedence.

    ``TestIsLauncherMainProcess`` monkeypatches ``rank_zero_only.rank`` directly, so it never exercises the
    ``RANK`` > ``LOCAL_RANK`` > ``SLURM_PROCID`` > ``JSM_NAMESPACE_RANK`` resolution order Lightning applies once,
    at first import of ``pytorch_lightning.utilities.rank_zero`` -- nor the guard's own handling of
    ``LOCAL_RANK``/``OMPI_COMM_WORLD_RANK``/``PMI_RANK``. A fresh subprocess per case is the only way to observe
    that resolution honestly, since a single process can only resolve it once.
    """

    @pytest.mark.parametrize(
        ("env_vars", "expected"),
        [
            pytest.param({}, True, id="no-launcher-vars-present"),
            pytest.param({"LOCAL_RANK": "1"}, False, id="local-rank-nonzero"),
            pytest.param({"SLURM_PROCID": "1"}, False, id="slurm-procid-nonzero"),
            pytest.param({"RANK": "0", "NODE_RANK": "1"}, False, id="rank-zero-secondary-node"),
            pytest.param({"SLURM_PROCID": "0", "LOCAL_RANK": "1"}, False, id="local-rank-precedes-slurm-procid"),
            pytest.param({"RANK": "0", "LOCAL_RANK": "1"}, False, id="rank-zero-but-local-rank-nonzero"),
            pytest.param({"RANK": "1"}, False, id="rank-nonzero"),
            pytest.param({"JSM_NAMESPACE_RANK": "1"}, False, id="jsm-namespace-rank-nonzero"),
            pytest.param({"OMPI_COMM_WORLD_RANK": "1"}, False, id="ompi-comm-world-rank-nonzero"),
            pytest.param({"PMI_RANK": "1"}, False, id="pmi-rank-nonzero"),
            pytest.param({"NODE_RANK": "0", "LOCAL_RANK": "0"}, True, id="node-rank-zero-local-rank-zero"),
        ],
    )
    def test_answers_from_a_fresh_process_env(self, env_vars: dict[str, str], expected: bool) -> None:
        """A fresh interpreter resolves the launcher's rank the same way a real launched process would.

        Each case starts a subprocess from a minimal, rank-var-stripped environment (see ``_minimal_subprocess_env``),
        applies only its own launcher variables, imports ``rfdetr.utilities.distributed`` fresh so
        ``rank_zero_only.rank`` is resolved from exactly that environment rather than carried over from this test
        process or an earlier case, and prints ``_is_launcher_main_process()``. The ``rank-zero-but-local-rank-
        nonzero``, ``ompi-comm-world-rank-nonzero``, and ``pmi-rank-nonzero`` cases pin the guard's target contract --
        requiring ``LOCAL_RANK`` in ``{unset, "0"}`` and rejecting a nonzero ``OMPI_COMM_WORLD_RANK``/``PMI_RANK`` --
        and fail until that guard change lands in ``_is_launcher_main_process``.
        """
        pytest.importorskip("pytorch_lightning")
        env = _minimal_subprocess_env()
        env.update(env_vars)
        code = "import rfdetr.utilities.distributed as d; print(d._is_launcher_main_process())"

        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.strip() == str(expected), result.stderr


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

    Examples:
        Not directly callable in a doctest -- it dispatches ``dist.init_process_group("xla", ...)`` and requires
        a real TPU/NEURON multi-process runtime supplied by ``torch_xla.launch``. See
        ``test_all_gather_multiprocess_xla_collective_routing`` for the real invocation.

        >>> callable(_xla_all_gather_worker)  # doctest: +SKIP
        True
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

    if torch_xla._XLAC._xla_runtime_is_initialized():
        pytest.skip(
            "the XLA runtime is already live in this process, so torch_xla.launch cannot spawn replicas -- PJRT "
            "hands the chips to the first process that claims them. Any sibling xla-marked test that touched the "
            "device leaves it live, and a subprocess does not help because the parent still owns the chips. Run "
            "this test alone, e.g. 'pytest tests/utilities/test_distributed.py::test_all_gather_multiprocess_"
            "xla_collective_routing', to exercise it on real TPU/NEURON hardware."
        )

    torch_xla.launch(_xla_all_gather_worker)
