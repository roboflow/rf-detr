# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Distributed-training helpers (world-size, rank, all_gather, reduce_dict)."""

from __future__ import annotations

import os
import pickle
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


def is_dist_avail_and_initialized() -> bool:
    """Return True if torch.distributed is available and has been initialised."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    """Return the number of processes in the current distributed group."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Return the rank of the current process in the distributed group."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Return True if the current process is rank 0.

    See Also:
        Before the process group is up, use :func:`_is_launcher_main_process`; :func:`get_rank` reports 0 in every
        process until then.
    """
    return get_rank() == 0


def _is_launcher_main_process() -> bool:
    """Return True if the process launcher designates this process to write files shared across a run.

    The counterpart to :func:`is_main_process` for code that runs *before* ``torch.distributed`` is initialized,
    where :func:`get_rank` reports 0 in every process and so cannot tell them apart. Prefer
    :func:`is_main_process` wherever the process group is already up: it reads the real global rank instead of
    inferring one from the environment.

    The answer is an environment-based heuristic. It starts from Lightning's own ``rank_zero_only.rank`` -- resolved
    once at import from ``RANK``, ``LOCAL_RANK``, ``SLURM_PROCID`` or ``JSM_NAMESPACE_RANK`` -- and adds checks that
    resolution does not make, each of which marks the process as not the main one when set to anything but ``"0"``:

    * ``NODE_RANK``, because the Lightning rank is node-local when it comes from ``LOCAL_RANK``, which Lightning's
      own subprocess launcher sets to 0 for the process it launches from on every node of a multi-node run, so
      without it one process per node would pass.
    * ``LOCAL_RANK``, because a ``RANK=0`` inherited from the parent environment wins Lightning's resolution in
      every subprocess child (the launcher overrides only ``LOCAL_RANK``), so without it every local rank would
      pass; no supported launcher gives global rank 0 a ``LOCAL_RANK`` other than ``"0"``.
    * ``OMPI_COMM_WORLD_RANK`` and ``PMI_RANK``, the global rank under ``mpirun``/PMI launchers. Lightning reads it
      only through ``mpi4py`` (its ``MPIEnvironment``), never at import, so without them every MPI worker would pass.

    The answer is per launch, not per training: because ``SLURM_PROCID`` feeds the Lightning rank, an ``srun`` sweep
    of independent ``devices=1`` trainings, one per task, sees ``SLURM_PROCID`` other than 0 on tasks 1..N-1 and the
    guard returns False there, so those tasks skip the pre-fit writes it protects. The post-fit rewrite guarded by
    :func:`is_main_process` is unaffected, as each such task has no process group and reports rank 0.

    Returns:
        Whether the launcher's environment identifies this process as rank 0 of node 0.

    Examples:
        The answer is read from the launcher environment this process was started in, so it is only meaningful
        against a known one; ``tests/utilities/test_distributed.py`` pins each case.

        >>> isinstance(_is_launcher_main_process(), bool)
        True
    """
    # pytorch_lightning ships in the optional `train` extra; a module-scope import would make `import rfdetr`
    # require it. Every caller reaches this only after the training stack has already been imported. Lightning
    # resolves `.rank` from RANK / LOCAL_RANK / SLURM_PROCID / JSM_NAMESPACE_RANK when this module is first
    # imported, which a launcher does before the subprocess it starts ever calls into rfdetr.
    # PTL re-exports this without `__all__` (an implicit re-export mypy --strict rejects; the `rank_zero_warn` import
    # in `training/callbacks/gpu_memory_progress_bar.py` carries the same ignore) and types it as an overload, which
    # has no `.rank` attribute -- Lightning attaches that at runtime, hence the second ignore on the access below.
    from pytorch_lightning.utilities.rank_zero import rank_zero_only  # type: ignore[attr-defined]

    return (
        rank_zero_only.rank == 0  # type: ignore[attr-defined]
        and os.environ.get("NODE_RANK", "0") == "0"
        and os.environ.get("LOCAL_RANK", "0") == "0"
        and os.environ.get("OMPI_COMM_WORLD_RANK", "0") == "0"
        and os.environ.get("PMI_RANK", "0") == "0"
    )


def save_on_master(obj: Any, f: Any, *args: Any, **kwargs: Any) -> None:
    """Save *obj* to *f* only on the main process (rank 0).

    Args:
        obj: Object to save.
        f: File path or file-like object passed to ``torch.save``.
        *args: Additional positional arguments forwarded to ``torch.save``. **kwargs: Additional keyword arguments
        forwarded to ``torch.save``.
    """
    if is_main_process():
        torch.save(obj, f, *args, **kwargs)


def all_gather(data: Any, device: torch.device | None = None) -> list[Any]:
    """Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: Any picklable object.
        device: Device for the intermediate byte tensors. If ``None``, derived from the process group
            backend (``cuda`` for NCCL, ``cpu`` otherwise). XLA callers must pass their local XLA device
            explicitly — XLA has no dedicated ``dist`` backend name to probe, and defaulting to CPU would
            place gather buffers on the wrong device.

    Returns:
        List of data gathered from each rank.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # Serialize to a byte tensor on the active device.
    if device is None:
        backend = dist.get_backend() if is_dist_avail_and_initialized() else "cpu"
        device = torch.device("cuda" if backend == "nccl" else "cpu")
    buffer = pickle.dumps(data)
    tensor = torch.tensor(bytearray(buffer), dtype=torch.uint8, device=device)

    # obtain Tensor size of each rank
    local_size = tensor.numel()
    local_size_tensor = torch.tensor([local_size], device=device)
    size_tensor_list = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(size_tensor_list, local_size_tensor)
    size_list = [int(size.item()) for size in size_tensor_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict: dict[str, Tensor], average: bool = True) -> dict[str, Tensor]:
    """Reduce values in *input_dict* across all processes.

    Args:
        input_dict: Dict whose values will be reduced.
        average: If True, compute the mean across ranks; otherwise compute the sum.

    Returns:
        Dict with the same keys as *input_dict*, with values averaged/summed across ranks.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        value_list = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            value_list.append(input_dict[k])
        values = torch.stack(value_list, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
