# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Sequential-I/O dataset pipeline backed by WebDataset tar shards.

A loose-file COCO split makes one ``open()`` per image per epoch. Raising ``num_workers`` to keep a GPU fed turns that
into thousands of concurrent random opens, and the cost lands in the kernel rather than in decode work. Packing the
same split into a few hundred ``.tar`` shards converts those random opens into sequential reads of large files: each
worker walks its own shards front to back and pays one ``open()`` per shard instead of one per image.

The module is deliberately split in two halves with different dependencies:

- **Packing** (:func:`pack_coco_to_shards`) uses only the standard library's :mod:`tarfile`. A WebDataset shard is an
  ordinary POSIX tar whose members are grouped by basename, so nothing about writing one needs the ``webdataset``
  package. Keeping the packer dependency-free means it is exercised by the default CI job rather than only by an
  extra-gated one.
- **Reading** (:class:`WebDatasetDetection`, :func:`build_webdataset_loader`) imports ``webdataset`` lazily, so the
  optional ``rfdetr[webdataset]`` extra is only required by a run that actually streams shards.

:func:`build_webdataset_loader` returns a stock :class:`~torch.utils.data.DataLoader` rather than
``webdataset.WebLoader``; see that function for why the wrapper is dropped while the streaming pipeline is kept.

Image bytes are copied into the shard verbatim — no re-encode — so a packed split decodes to the same pixels as the
loose files it came from. Decoded samples are handed to :class:`~rfdetr.datasets.coco.ConvertCoco` and then to the
transform pipeline built by :func:`~rfdetr.datasets.coco.make_coco_transforms`, which is the same CPU-side
Albumentations/torchvision stack the loose-file path uses; batches leave the workers through the DataModule's own
collate function, so the ``pin_memory`` hand-off and the Kornia GPU stage in
:meth:`~rfdetr.training.module_data.RFDETRDataModule.on_after_batch_transfer` are untouched.

Scope: detection and segmentation splits. Keypoint training is rejected explicitly — its label space is derived from a
whole parsed COCO file (:func:`~rfdetr.datasets._keypoint_schema.infer_coco_keypoint_schema`), which a shard index
does not carry.

See https://github.com/roboflow/rf-detr/issues/1392.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Literal

import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader

from rfdetr.datasets.coco import (
    ConvertCoco,
    filter_parent_categories,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)
from rfdetr.datasets.kornia_transforms import is_gpu_postprocess, resolve_backend_for_build
from rfdetr.utilities.logger import get_logger

logger = get_logger()

INDEX_VERSION = 1
"""Schema version stamped into every shard index, so a future format change can be rejected with a clear message."""

DEFAULT_MAX_SHARD_BYTES = 100 * 1024 * 1024
"""Default shard size target in bytes (~100 MB), the size range WebDataset is tuned for."""

DEFAULT_SHUFFLE_BUFFER = 1000
"""Samples held in the training reservoir shuffle buffer, on top of shard-order shuffling."""

SHARD_SKEW_WARN_FRACTION = 0.05
"""Warn once per loader when the smallest worker's shard share falls this far below the average.

Shards split across workers by count, so an uneven split leaves the smallest worker short of the samples a fixed epoch
asks of it: it wraps and repeats its own while better-supplied workers leave some unseen. Measured on a 2,000-image
3-epoch fine-tune, 39 shards over 8 workers (18% short) cost about 0.011 mAP@50:95 against the map-style loader, while
re-packing the same split into 290 shards (0.6% short) closed it.
"""

DEFAULT_SHARD_SHUFFLE = 100
"""Shards held in the training shard-order shuffle buffer.

WebDataset rejects ``True`` here with a warning and silently substitutes this same value, so it is passed as an integer.
"""

CategoryIdPolicy = Literal["remap", "raw"]

_IMAGE_EXTENSIONS: tuple[str, ...] = ("jpg", "jpeg", "png", "webp", "bmp")
_TAR_MEMBER_MODE = 0o644


def _validate_split_name(split: str) -> str:
    """Reject a split name that could escape the shard directory as a path component.

    *split* becomes part of a shard or index file name with no other sanitisation, so a value containing a path
    separator or a ``..`` segment would let :func:`_shard_name`/:func:`index_name` resolve outside the shard
    directory — writing shards outside ``output_dir`` when packing, or reading an index from outside ``shard_dir``
    when loading.

    Args:
        split: Split name supplied by a caller (CLI argument or library call).

    Returns:
        *split*, unchanged, once validated.

    Raises:
        ValueError: If *split* is empty, or contains a path separator or a ``.``/``..`` segment.

    Examples:
        >>> _validate_split_name("train")
        'train'
        >>> _validate_split_name("../escape")
        Traceback (most recent call last):
            ...
        ValueError: split '../escape' must not contain a path separator or '..'.
    """
    if not split or "/" in split or "\\" in split or split in (".", ".."):
        raise ValueError(f"split {split!r} must not contain a path separator or '..'.")
    return split


def _shard_name(split: str, index: int) -> str:
    """Return the file name of shard *index* of *split*.

    Args:
        split: Split name the shard belongs to.
        index: Zero-based shard number.

    Returns:
        Shard file name.

    Raises:
        ValueError: If *split* would escape the shard directory (see :func:`_validate_split_name`).

    Examples:
        >>> _shard_name("train", 7)
        'train-000007.tar'
    """
    _validate_split_name(split)
    return f"{split}-{index:06d}.tar"


def index_name(split: str) -> str:
    """Return the file name of the shard index of *split*.

    Args:
        split: Split name the index describes.

    Returns:
        Index file name, relative to the shard directory.

    Raises:
        ValueError: If *split* would escape the shard directory (see :func:`_validate_split_name`).

    Examples:
        >>> index_name("val")
        'val-index.json'
    """
    _validate_split_name(split)
    return f"{split}-index.json"


@dataclass(frozen=True)
class ShardIndex:
    """Everything a reader needs about one packed split without opening a shard.

    Args:
        split: Split name, matching the shard and index file names.
        shards: Shard file names in packing order, relative to the index's directory.
        num_samples: Total samples across all shards.
        categories: COCO ``categories`` entries copied verbatim from the source annotation file.
        annotated_category_ids: Category ids carrying at least one annotation in this split.
        category_ids: ``"remap"`` when labels are contiguous 0-based indices, ``"raw"`` when they are the source
            ``category_id`` values.
        samples_per_shard: Sample count of each shard, aligned by position with *shards*. Shards are cut by byte
            size, not sample count, so this can vary widely between shards of the same split; an empty tuple means
            the index predates this field (or was built without it), and readers fall back to assuming a uniform
            split.
    """

    split: str
    shards: tuple[str, ...]
    num_samples: int
    categories: tuple[dict[str, Any], ...]
    annotated_category_ids: tuple[int, ...]
    category_ids: CategoryIdPolicy
    samples_per_shard: tuple[int, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-serialisable form written to disk.

        Returns:
            Mapping stamped with :data:`INDEX_VERSION`.

        Examples:
            >>> index = ShardIndex("val", ("val-000000.tar",), 1, ({"id": 1, "name": "a"},), (1,), "remap", (1,))
            >>> index.to_json()["num_samples"]
            1
        """
        return {
            "version": INDEX_VERSION,
            "split": self.split,
            "shards": list(self.shards),
            "num_samples": self.num_samples,
            "categories": list(self.categories),
            "annotated_category_ids": list(self.annotated_category_ids),
            "category_ids": self.category_ids,
            "samples_per_shard": list(self.samples_per_shard),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ShardIndex:
        """Rebuild an index from its on-disk mapping.

        Args:
            payload: Parsed index JSON.

        Returns:
            The reconstructed index.

        Raises:
            ValueError: If the index was written by an incompatible schema version.

        Examples:
            >>> payload = {
            ...     "version": 1, "split": "val", "shards": ["val-000000.tar"], "num_samples": 1,
            ...     "categories": [{"id": 1, "name": "a"}], "annotated_category_ids": [1], "category_ids": "remap",
            ... }
            >>> ShardIndex.from_json(payload).split
            'val'
        """
        version = int(payload.get("version", 0))
        if version != INDEX_VERSION:
            raise ValueError(
                f"Shard index schema version {version} is not supported by this RF-DETR release "
                f"(expected {INDEX_VERSION}); re-pack the dataset with the current packer."
            )
        policy = payload.get("category_ids", "remap")
        if policy not in ("remap", "raw"):
            raise ValueError(f"Shard index declares unknown category_ids policy {policy!r}; expected 'remap' or 'raw'.")
        return cls(
            split=str(payload["split"]),
            shards=tuple(str(shard) for shard in payload["shards"]),
            num_samples=int(payload["num_samples"]),
            categories=tuple(payload.get("categories", ())),
            annotated_category_ids=tuple(int(cid) for cid in payload.get("annotated_category_ids", ())),
            category_ids=policy,
            samples_per_shard=tuple(int(count) for count in payload.get("samples_per_shard", ())),
        )

    def cat2label(self) -> dict[int, int] | None:
        """Return the ``category_id`` to label-index mapping implied by this index.

        Mirrors :class:`~rfdetr.datasets.coco.CocoDetection`'s detection branch: unannotated grouping categories are
        dropped by :func:`~rfdetr.datasets.coco.filter_parent_categories` before indices are assigned, so a synthetic
        Roboflow root category consumes no output slot.

        Returns:
            The mapping, or ``None`` when the index declares the ``"raw"`` policy and source ids are used as labels.

        Examples:
            >>> categories = ({"id": 3, "name": "a"}, {"id": 9, "name": "b"})
            >>> ShardIndex("t", (), 0, categories, (3, 9), "remap").cat2label()
            {3: 0, 9: 1}
            >>> ShardIndex("t", (), 0, categories, (3, 9), "raw").cat2label() is None
            True
        """
        if self.category_ids == "raw":
            return None
        kept = filter_parent_categories(list(self.categories), set(self.annotated_category_ids))
        return {int(category["id"]): label for label, category in enumerate(kept)}


class WebDatasetSplitUnavailableError(FileNotFoundError):
    """Raised when a shard directory carries no index for the requested split.

    A subclass of :class:`FileNotFoundError`, so callers that only care that the split is missing keep working, while
    :meth:`~rfdetr.training.module_data.RFDETRDataModule._build_test_dataset` can tell "this split was never packed"
    apart from any other missing file and fall back to ``val`` rather than aborting the run.
    """


def read_shard_index(shard_dir: str | Path, split: str) -> ShardIndex:
    """Load the shard index of *split* from *shard_dir*.

    Args:
        shard_dir: Directory holding the shards and their index files.
        split: Split name to read.

    Returns:
        The parsed index.

    Raises:
        WebDatasetSplitUnavailableError: If the split has no index file in *shard_dir*.
    """
    path = Path(shard_dir) / index_name(split)
    if not path.exists():
        raise WebDatasetSplitUnavailableError(
            f"No WebDataset index for split {split!r} at {path}. "
            f"Pack the split first: python -m rfdetr.datasets.webdataset_io --split {split} ..."
        )
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return ShardIndex.from_json(payload)


# ----------------------------------------------------------------------------------------------------------------
# Packing
# ----------------------------------------------------------------------------------------------------------------


def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    """Append *payload* to *tar* as a regular member named *name*.

    Ownership and timestamps are zeroed so that packing the same split twice produces byte-identical shards.

    Args:
        tar: Open archive to append to.
        name: Member name, whose extension becomes the sample's WebDataset field.
        payload: Member contents.
    """
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = _TAR_MEMBER_MODE
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tar.addfile(info, io.BytesIO(payload))


def _annotations_by_image(coco_data: dict[str, Any]) -> dict[Any, list[dict[str, Any]]]:
    """Group a parsed COCO file's annotations by ``image_id``.

    Args:
        coco_data: Parsed COCO JSON.

    Returns:
        Mapping from image id to its annotation list; images without annotations are absent.

    Examples:
        >>> _annotations_by_image({"annotations": [{"image_id": 5, "category_id": 1}]})[5][0]["category_id"]
        1
    """
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco_data.get("annotations", []):
        grouped[annotation["image_id"]].append(annotation)
    return grouped


def pack_coco_to_shards(
    image_dir: str | Path,
    annotations_file: str | Path,
    output_dir: str | Path,
    *,
    split: str = "train",
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    category_ids: CategoryIdPolicy = "remap",
) -> ShardIndex:
    """Pack a COCO-format split into WebDataset tar shards plus a JSON index.

    Every image becomes two adjacent tar members sharing one basename: the original image file copied byte for byte,
    and a ``.json`` sidecar holding that image's ``image_id``, ``file_name`` and annotation list. A new shard is
    started once the current one reaches *max_shard_bytes*, so shards land near that size rather than exactly on it —
    a sample is never split across two shards.

    Args:
        image_dir: Directory holding the split's image files, named as the annotation file's ``file_name`` entries.
        annotations_file: COCO-format JSON annotation file for the split.
        output_dir: Directory to write shards and the index into; created if absent.
        split: Split name used for the shard and index file names.
        max_shard_bytes: Size at which the current shard is closed and the next one opened.
        category_ids: ``"remap"`` assigns contiguous 0-based labels, matching
            :func:`~rfdetr.datasets.coco.build_roboflow_from_coco`. ``"raw"`` keeps the source ``category_id`` values
            as labels, matching :func:`~rfdetr.datasets.coco.build_coco`'s convention for evaluating a model trained
            on the COCO-2017 label space.

    Returns:
        The index describing what was written.

    Raises:
        FileNotFoundError: If *annotations_file* or a listed image file does not exist. A missing image is fatal
            rather than skipped: dropping training images silently is worse than failing on the file that is absent.
        ValueError: If *max_shard_bytes* is not positive, if the annotation file lists no images, if an image's
            ``file_name`` resolves outside *image_dir*, if an image has an extension the reader cannot decode, or
            if any annotation's ``image_id`` matches no image — the shape an ``id``/``image_id`` type mismatch
            takes, which would otherwise pack a split that silently drops or mislabels samples. Packing stops
            before writing anything to *output_dir* on any of these failures, so a re-pack that fails partway
            through never leaves a previously-packed, valid split there in a corrupted, half-overwritten state.
    """
    if max_shard_bytes <= 0:
        raise ValueError(f"max_shard_bytes must be > 0, got {max_shard_bytes}.")
    _validate_split_name(split)

    image_root = Path(image_dir)
    image_root_resolved = image_root.resolve()
    annotations_path = Path(annotations_file)
    if not annotations_path.exists():
        raise FileNotFoundError(f"COCO annotation file {annotations_path} does not exist.")

    with annotations_path.open(encoding="utf-8") as handle:
        coco_data: dict[str, Any] = json.load(handle)

    images: list[dict[str, Any]] = list(coco_data.get("images", []))
    if not images:
        raise ValueError(f"COCO annotation file {annotations_path} lists no images; nothing to pack.")

    grouped = _annotations_by_image(coco_data)
    image_ids = {image_entry["id"] for image_entry in images}
    orphaned_ids = sorted((set(grouped) - image_ids), key=str)
    if orphaned_ids:
        raise ValueError(
            f"{annotations_path} has {len(orphaned_ids)} annotation image_id value(s) matching no image "
            f"(e.g. {orphaned_ids[0]!r}): those annotations would be dropped silently. The usual cause is a type "
            "mismatch between images[].id and annotations[].image_id (one string, one int)."
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".{split}-pack-", dir=destination))

    shard_names: list[str] = []
    shard_sample_counts: list[int] = []
    annotated_ids: set[int] = set()
    shard_index = 0
    written = 0
    shard_samples = 0
    tar: tarfile.TarFile | None = None
    shard_bytes = 0

    try:
        for position, image_entry in enumerate(images):
            source = image_root / str(image_entry["file_name"])
            resolved_source = source.resolve()
            if not resolved_source.is_relative_to(image_root_resolved):
                raise ValueError(
                    f"Image file_name {image_entry['file_name']!r} in {annotations_path} resolves outside "
                    f"{image_root}; refusing to read a file the split's image directory does not own."
                )
            if not source.exists():
                raise FileNotFoundError(
                    f"Image {source} listed in {annotations_path} does not exist; "
                    "packing stops rather than silently dropping it from the training set."
                )
            extension = source.suffix.lower().lstrip(".")
            if extension not in _IMAGE_EXTENSIONS:
                raise ValueError(
                    f"Image {source} has extension {extension or '(none)'}, which the shard reader cannot decode "
                    f"(supported: {', '.join(_IMAGE_EXTENSIONS)}). Convert the split before packing, so this fails "
                    "here rather than on the first training batch."
                )
            payload = source.read_bytes()
            annotations = grouped.get(image_entry["id"], [])
            annotated_ids.update(int(annotation["category_id"]) for annotation in annotations)
            sidecar = json.dumps(
                {
                    "image_id": image_entry["id"],
                    "file_name": image_entry["file_name"],
                    "annotations": annotations,
                }
            ).encode("utf-8")

            if tar is None:
                name = _shard_name(split, shard_index)
                tar = tarfile.open(work_dir / name, "w")
                shard_names.append(name)
                shard_bytes = 0
                shard_samples = 0

            key = f"{position:08d}"
            _add_bytes(tar, f"{key}.{extension}", payload)
            _add_bytes(tar, f"{key}.json", sidecar)
            shard_bytes += len(payload) + len(sidecar)
            written += 1
            shard_samples += 1

            if shard_bytes >= max_shard_bytes:
                tar.close()
                tar = None
                shard_sample_counts.append(shard_samples)
                shard_index += 1

        if tar is not None:
            tar.close()
            tar = None
            shard_sample_counts.append(shard_samples)

        index = ShardIndex(
            split=split,
            shards=tuple(shard_names),
            num_samples=written,
            categories=tuple(coco_data.get("categories", ())),
            annotated_category_ids=tuple(sorted(annotated_ids)),
            category_ids=category_ids,
            samples_per_shard=tuple(shard_sample_counts),
        )
        index_bytes = json.dumps(index.to_json()).encode("utf-8")

        # Every shard packed successfully: publish shards and the index together. A previous pack's shards that
        # this run does not reproduce (e.g. it produced fewer of them) are removed too, so no stale, unindexed
        # shard is left behind. The index is written last, so a reader can never observe an index whose shard
        # list is only partially on disk.
        stale_shards = {path.name for path in destination.glob(f"{split}-*.tar")} - set(shard_names)
        for name in shard_names:
            (work_dir / name).replace(destination / name)
        for name in stale_shards:
            (destination / name).unlink()
        (destination / index_name(split)).write_bytes(index_bytes)
    finally:
        if tar is not None:
            tar.close()
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info("Packed %d %s samples into %d shard(s) at %s", written, split, len(shard_names), destination)
    return index


# ----------------------------------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------------------------------


def _require_webdataset() -> Any:
    """Import and return the ``webdataset`` module, or raise an actionable ``ImportError``."""
    try:
        import webdataset
    except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
        raise ImportError(
            "Streaming WebDataset shards requires the webdataset package. "
            "Install with: pip install 'rfdetr[webdataset]'"
        ) from exc
    return webdataset


def _distributed_world_size() -> int:
    """Return the distributed world size, or ``1`` outside an initialised process group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def _shard_url(path: PurePath) -> str:
    """Return the ``file:`` URL naming *path* to ``webdataset``'s shard opener.

    ``webdataset`` calls ``urlparse`` on every shard string and, for the empty and ``file`` schemes, opens
    ``urlparse(url).path`` verbatim — no percent-decoding, and no conversion from URL syntax back to a native path.
    Two spellings that look right therefore fail on Windows: a bare ``str(path)`` parses ``C:\\shards\\train.tar``
    as scheme ``c``, which has no handler, and :meth:`pathlib.Path.as_uri` yields ``file:///C:/shards/train.tar``,
    whose path keeps a leading slash that ``open`` rejects with ``[Errno 22] Invalid argument``. An authority-less
    ``file:`` URL over the forward-slash form of the path leaves ``C:/shards/train.tar`` on Windows and
    ``/shards/train.tar`` elsewhere, both of which ``open`` takes as-is. Percent-encoding is deliberately omitted
    for the same reason: nothing downstream reverses it, so a directory containing spaces has to pass through
    literally.

    Args:
        path: Path to one shard, absolute or relative to the working directory.

    Returns:
        The URL to hand to ``webdataset``.

    Examples:
        >>> from pathlib import PurePosixPath, PureWindowsPath
        >>> _shard_url(PureWindowsPath(r"C:\\data\\shards\\train-000000.tar"))
        'file:C:/data/shards/train-000000.tar'
        >>> _shard_url(PurePosixPath("/data/shards/train-000000.tar"))
        'file:/data/shards/train-000000.tar'
    """
    return f"file:{path.as_posix()}"


class WebDatasetDetection(torch.utils.data.IterableDataset[tuple[Any, Any]]):
    """Streaming COCO-style detection dataset reading WebDataset tar shards.

    Yields the same ``(image, target)`` pairs as :class:`~rfdetr.datasets.coco.CocoDetection`: the sidecar JSON goes
    through :class:`~rfdetr.datasets.coco.ConvertCoco` and the result through *transforms*, both inside the DataLoader
    worker that read the shard.

    Sizing follows WebDataset's own convention and has two modes, chosen by :meth:`configure_epoch`:

    - **Unplanned** (the default, and what evaluation uses): every worker drains its own shards exactly once, so the
      split is seen once per epoch with no sample repeated or dropped. The dataset is unsized — ``len()`` raises
      :class:`TypeError`, as it does for any un-lengthed iterable — because the per-worker batch tails make the batch
      count depend on how shards happen to divide across workers.
    - **Planned** (what training uses): every worker yields exactly ``samples_per_worker`` samples, wrapping around
      its own shards if that subset is shorter. The epoch length is then fixed and ``len()`` is exact, which is what
      ``trainer.estimated_stepping_batches`` — and through it the LR schedule — needs. The trade-off is that a
      worker holding fewer shards than average repeats some of its samples within the epoch.

    Args:
        shard_dir: Directory holding the shards and the split's index file.
        split: Split name to read.
        transforms: Transform pipeline applied to ``(image, target)`` after annotation conversion, or ``None``.
        include_masks: Decode polygon/RLE segmentation into binary mask tensors.
        cat2label: ``category_id`` to label-index mapping. ``None`` uses the mapping implied by the shard index.
        shuffle_buffer: Reservoir size for within-shard shuffling; ``0`` disables it.
        shard_shuffle: Shards held in the shard-order shuffle buffer; ``0`` visits shards in packing order.
            Together with *shuffle_buffer* this is the streaming counterpart of ``shuffle=True`` on a map-style
            loader — a local shuffle, not a global permutation.
        seed: Base seed, used directly only when the dataset iterates in the main process. Under DataLoader workers
            the per-epoch seed comes from PyTorch's own per-epoch worker seeding; see :meth:`_epoch_seeds`.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        split: str,
        transforms: Any | None,
        *,
        include_masks: bool = False,
        cat2label: dict[int, int] | None = None,
        shuffle_buffer: int = 0,
        shard_shuffle: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._shard_dir = Path(shard_dir)
        self._split = split
        self._transforms = transforms
        self._shuffle_buffer = shuffle_buffer
        self._shard_shuffle = shard_shuffle
        self._seed = seed
        self.index = read_shard_index(self._shard_dir, split)
        self.cat2label = self.index.cat2label() if cat2label is None else dict(cat2label)
        self.label2cat = None if self.cat2label is None else {label: cat_id for cat_id, label in self.cat2label.items()}
        self.prepare = ConvertCoco(include_masks=include_masks, cat2label=self.cat2label)
        self._samples_per_worker: int | None = None
        self._planned_workers = 1
        self._epoch_counter = -1

    @property
    def total_samples(self) -> int:
        """Samples the shard index reports for this split, independent of any epoch plan."""
        return self.index.num_samples

    @property
    def class_names(self) -> list[str]:
        """Category names in the order this split's label space puts them.

        The map-style datasets expose the same thing through their ``coco`` object, which a shard stream has no
        equivalent of; the packed index carries the category list instead. Every entry sits at its own label
        index, so ``class_names[label]`` is always the emitted label's name: under ``"remap"`` that is the
        contiguous 0-based index, and under ``"raw"`` it is the source ``category_id`` itself — raw labels skip
        whatever gaps the id range has, so the list carries an empty string at every skipped index rather than
        shifting later names down to fill the gap.

        Returns:
            The category names, indexed by label, with an empty string at every label with no category.
        """
        categories = {int(category["id"]): str(category["name"]) for category in self.index.categories}
        if self.label2cat is None:
            if not categories:
                return []
            names = [""] * (max(categories) + 1)
            for category_id, name in categories.items():
                names[category_id] = name
            return names
        names = [""] * (max(self.label2cat) + 1)
        for label, category_id in sorted(self.label2cat.items()):
            if category_id in categories:
                names[label] = categories[category_id]
        return names

    def configure_epoch(self, *, samples_per_worker: int, num_workers: int) -> None:
        """Fix the epoch length so ``len()`` is exact.

        Args:
            samples_per_worker: Samples each DataLoader worker yields per epoch.
            num_workers: Workers the loader will run; ``0`` and ``1`` both mean one iterating process.

        Raises:
            ValueError: If either argument is below one.
        """
        if samples_per_worker < 1:
            raise ValueError(f"samples_per_worker must be >= 1, got {samples_per_worker}.")
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}.")
        self._samples_per_worker = samples_per_worker
        self._planned_workers = num_workers

    def __len__(self) -> int:
        """Return the planned per-epoch sample count for this rank.

        Raises:
            TypeError: If no epoch was planned, matching how ``len()`` behaves on any un-lengthed iterable.
        """
        if self._samples_per_worker is None:
            raise TypeError(
                f"WebDatasetDetection({self._split!r}) has no planned epoch length. "
                "Call configure_epoch() — build_webdataset_loader() does it for the training loader — "
                "or treat the dataset as unsized."
            )
        return self._samples_per_worker * self._planned_workers

    def _epoch_seeds(self) -> tuple[int, int]:
        """Return this epoch's ``(shard_order_seed, buffer_seed)``.

        The shard-order shuffle runs *before* the split by node and worker, so every worker of one epoch has to draw
        the same permutation or the split stops being a partition — while a seed that never changes would replay one
        sample order every epoch. Under DataLoader workers, ``torch.initial_seed()`` only changes epoch to epoch
        when the loader restarts its worker processes: with ``persistent_workers=False`` PyTorch draws a fresh base
        seed and hands worker *i* ``base_seed + i`` every time it spawns them, but with ``persistent_workers=True``
        — the DataModule's own default whenever ``num_workers > 0`` — the same worker process stays alive across
        every epoch and keeps the seed it was started with. ``__iter__`` still runs once per worker per epoch either
        way (PyTorch calls it fresh at every epoch boundary, even for a persistent worker), so ``self`` survives
        inside that worker across epochs and a plain counter incremented there is exact regardless of restart
        policy: added to the worker's base seed, it changes the draw every epoch even when the seed itself does not.
        Iterating in the main process has no ``torch.initial_seed()`` per-epoch signal at all, so the same counter
        stands in for the whole seed there too.

        Returns:
            Seed shared by every worker this epoch, and a seed unique to this worker.
        """
        self._epoch_counter += 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            shared = (self._seed + self._epoch_counter) % (2**31)
            return shared, shared
        base_seed = (torch.initial_seed() - worker_info.id) % (2**31)
        shared = (base_seed + self._epoch_counter) % (2**31)
        return shared, (shared + worker_info.id) % (2**31)

    def _decode(self, sample: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Turn one raw WebDataset sample into the ``(image, target)`` pair the transform pipeline expects.

        Args:
            sample: Raw sample dict keyed by member extension.

        Returns:
            Converted and transformed ``(image, target)``.

        Raises:
            KeyError: If the sample carries no recognised image member or no ``json`` sidecar.
        """
        extension = next((candidate for candidate in _IMAGE_EXTENSIONS if candidate in sample), None)
        if extension is None:
            present = sorted(key for key in sample if not key.startswith("__"))
            raise KeyError(
                f"WebDataset sample {sample.get('__key__', '?')} has no image member "
                f"(looked for {', '.join(_IMAGE_EXTENSIONS)}); found {present}."
            )
        if "json" not in sample:
            raise KeyError(f"WebDataset sample {sample.get('__key__', '?')} has no 'json' annotation sidecar.")

        with Image.open(io.BytesIO(sample[extension])) as handle:
            image = handle.convert("RGB")
        metadata = json.loads(sample["json"])
        target: dict[str, Any] = {"image_id": metadata["image_id"], "annotations": metadata["annotations"]}
        image, target = self.prepare(image, target)
        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target

    def _shard_urls(self) -> list[str]:
        """Return this split's shards as ``file:`` URLs.

        See :func:`_shard_url` for why a shard is named as a ``file:`` URL rather than by its plain path.

        Returns:
            One ``file:`` URL per shard, in index order.
        """
        return [_shard_url(self._shard_dir / shard) for shard in self.index.shards]

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        """Iterate this worker's share of the split.

        Returns:
            Iterator over ``(image, target)`` pairs.
        """
        wds = _require_webdataset()
        shard_seed, buffer_seed = self._epoch_seeds()
        urls = self._shard_urls()
        pipeline = wds.WebDataset(
            urls,
            # A split with fewer shards than workers legitimately leaves some workers with nothing to read;
            # empty_check=True would turn that into an exception instead of an empty share. An empty share ends
            # the worker's epoch immediately rather than spinning, because DataPipeline.iterator() breaks out of
            # its repetition loop as soon as one pass yields nothing.
            empty_check=False,
            shardshuffle=self._shard_shuffle,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            seed=shard_seed,
        )
        if self._shuffle_buffer > 0:
            pipeline = pipeline.shuffle(self._shuffle_buffer, seed=buffer_seed)
        pipeline = pipeline.map(self._decode)
        if self._samples_per_worker is not None:
            pipeline = pipeline.with_epoch(self._samples_per_worker)
        iterator: Iterator[tuple[Any, Any]] = iter(pipeline)
        return iterator


def plan_samples_per_worker(
    total_samples: int, *, batch_size: int, num_workers: int, world_size: int = 1, grad_accum_steps: int = 1
) -> int:
    """Return the per-worker epoch length that makes every emitted batch full.

    Flooring to a multiple of ``batch_size * grad_accum_steps`` is what keeps the reported length exact — and, with
    ``grad_accum_steps > 1``, what keeps every accumulation window complete: a worker that would end its epoch
    mid-batch or mid-window is asked for fewer samples instead, so ``drop_last=True`` never actually drops anything
    and PTL never fires the optimizer on a partial accumulation window
    (https://github.com/Lightning-AI/pytorch-lightning/issues/19987) — the streaming counterpart of what
    :class:`~rfdetr.training.module_data.GradAccumAlignedDataset` pads the map-style loader to.

    Args:
        total_samples: Samples in the split, across all ranks.
        batch_size: Per-rank micro-batch size.
        num_workers: DataLoader workers per rank; ``0`` counts as one iterating process.
        world_size: Number of distributed ranks sharing the split.
        grad_accum_steps: Micro-batches accumulated per optimizer step.

    Returns:
        Samples each worker yields per epoch.

    Raises:
        ValueError: If the split cannot fill one accumulation window per worker.

    Examples:
        >>> plan_samples_per_worker(1000, batch_size=4, num_workers=2)
        500
        >>> plan_samples_per_worker(1000, batch_size=16, num_workers=3)
        320
        >>> plan_samples_per_worker(1000, batch_size=4, num_workers=2, grad_accum_steps=8)
        480
    """
    workers = max(1, num_workers)
    window = batch_size * max(1, grad_accum_steps)
    per_worker = total_samples // (max(1, world_size) * workers)
    per_worker -= per_worker % window
    if per_worker < window:
        raise ValueError(
            f"A split of {total_samples} samples cannot fill one accumulation window of {window} samples "
            f"(batch_size={batch_size} x grad_accum_steps={grad_accum_steps}) per worker across {world_size} "
            f"rank(s) x {workers} worker(s). Lower num_workers, batch_size or grad_accum_steps, or pack more "
            "samples."
        )
    return per_worker


def build_webdataset_loader(
    dataset: WebDatasetDetection,
    *,
    batch_size: int,
    collate_fn: Callable[[list[tuple[Any, Any]]], tuple[Any, ...]],
    num_workers: int,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
    fixed_epoch: bool = True,
    world_size: int | None = None,
    grad_accum_steps: int = 1,
) -> DataLoader[Any]:
    """Build the loader that streams *dataset*.

    This returns a stock :class:`~torch.utils.data.DataLoader` rather than ``webdataset.WebLoader``. ``WebLoader`` is
    a fluid-interface wrapper that *constructs* the very same ``DataLoader`` and then hides it behind a
    ``DataPipeline``, which drops both ``__len__`` and the ``DataLoader`` type. RF-DETR needs the length:
    ``trainer.estimated_stepping_batches`` derives the LR schedule and the drop schedule from it, and ``WebLoader``
    only offers ``with_length()``, which fakes a length without guaranteeing it. The value ``WebLoader`` adds on top —
    post-loader ``unbatched().shuffle().batched()`` rebatching across workers — would break exactly that length
    guarantee. Streaming behaviour is identical either way: it comes from the pipeline inside *dataset*, not from the
    loader class.

    ``webdataset`` is imported here even though the loader itself does not use it, so a missing optional extra fails
    in the main process with an actionable message instead of inside a worker on the first batch.

    Args:
        dataset: The streaming dataset to wrap.
        batch_size: Per-rank micro-batch size.
        collate_fn: Batch collation callable, normally the DataModule's block-size-aware one.
        num_workers: DataLoader worker processes.
        pin_memory: Stage batches in pinned host memory before the device copy.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Batches prefetched per worker, or ``None`` for the DataLoader default.
        worker_init_fn: Per-worker initialisation hook. The CPU augmentation stack draws from NumPy and ``random``,
            which PyTorch does not seed per worker, so this needs the same seeding hook the map-style loaders use.
        fixed_epoch: Plan a fixed-length epoch and drop partial batches. ``True`` for training, where a known length
            drives the LR schedule; ``False`` for evaluation, where every sample must be seen exactly once.
        world_size: Distributed ranks sharing the split. ``None`` reads it from the active process group.
        grad_accum_steps: Micro-batches accumulated per optimizer step. Only used when *fixed_epoch* is ``True``;
            see :func:`plan_samples_per_worker`.

    Returns:
        A ``DataLoader`` over *dataset*.

    Raises:
        ValueError: If a planned epoch would leave a worker with no shard to read.
    """
    _require_webdataset()
    ranks = _distributed_world_size() if world_size is None else world_size
    if fixed_epoch:
        shards = len(dataset.index.shards)
        slots = ranks * max(1, num_workers)
        if slots > shards:
            raise ValueError(
                f"Split {dataset.index.split!r} has {shards} shard(s), which cannot cover "
                f"{ranks} rank(s) x {max(1, num_workers)} worker(s): a worker left with no shard would yield "
                "nothing and silently shorten the epoch. Lower num_workers, or re-pack with a smaller "
                "--max-shard-mb so the split has more shards."
            )
        per_shard = dataset.index.samples_per_shard
        if len(per_shard) == shards:
            # Real per-shard sample counts are available: measure the worst slot's actual share instead of
            # assuming every shard carries the same number of samples. Shards are cut by byte size, not sample
            # count, so that assumption can be badly wrong when image sizes vary — a byte-balanced split can still
            # leave one worker with far fewer samples than the count-based approximation below would suggest.
            slot_totals = [sum(per_shard[position::slots]) for position in range(slots)]
            worst = min(slot_totals)
            average = dataset.total_samples / slots
            deficit = 1.0 - (worst / average if average > 0 else 1.0)
            measured = True
        else:
            # No per-shard counts on this index (e.g. one built by hand rather than by the packer): fall back to
            # assuming every shard carries the same number of samples, which is only an approximation.
            deficit = 1.0 - (shards // slots) / (shards / slots)
            measured = False
        if deficit > SHARD_SKEW_WARN_FRACTION:
            logger.warning(
                "Split %r has %d shards for %d rank(s) x %d worker(s), so the worst-served worker holds "
                "%s%.0f%% fewer samples than the epoch asks of it: it repeats some of its own while "
                "better-supplied workers leave some unseen, which measurably costs accuracy. Re-pack with a "
                "smaller --max-shard-mb (aim for a shard count that divides %d, or simply many more shards than "
                "workers).",
                dataset.index.split,
                shards,
                ranks,
                max(1, num_workers),
                "" if measured else "an estimated ",
                deficit * 100,
                slots,
            )
        dataset.configure_epoch(
            samples_per_worker=plan_samples_per_worker(
                dataset.total_samples,
                batch_size=batch_size,
                num_workers=num_workers,
                world_size=ranks,
                grad_accum_steps=grad_accum_steps,
            ),
            num_workers=max(1, num_workers),
        )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "drop_last": fixed_epoch,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": worker_init_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def build_webdataset(image_set: str, args: Any, resolution: int) -> WebDatasetDetection:
    """Build the WebDataset-backed dataset for *image_set*.

    Reuses the loose-file transform pipeline unchanged, so a packed split trains through exactly the same CPU
    augmentation stack — including the optional Albumentations backend — as the directory it was packed from.

    Non-train splits adopt the train split's label mapping, for the same reason
    :func:`~rfdetr.datasets.coco.build_roboflow_from_coco` does: deriving indices per split shifts them whenever a
    split's annotation coverage of a grouping category differs from the train split's.

    Args:
        image_set: Split identifier, optionally suffixed (``"val_speed"`` reads the ``val`` shards).
        args: Merged model/train namespace, as built by :func:`rfdetr._namespace._namespace_from_configs`.
        resolution: Target square resolution in pixels.

    Returns:
        The streaming dataset for that split.

    Raises:
        FileNotFoundError: If ``args.dataset_dir`` does not exist.
        NotImplementedError: If keypoint training is requested.
    """
    root = Path(args.dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"WebDataset shard directory {root} does not exist")
    if getattr(args, "use_grouppose_keypoints", False):
        raise NotImplementedError(
            "dataset_file='webdataset' does not support keypoint training: the keypoint label space is inferred from "
            "a whole parsed COCO annotation file, which a shard index does not carry. Use dataset_file='coco', "
            "'roboflow' or 'yolo' for keypoints."
        )

    split = image_set.split("_", maxsplit=1)[0]
    is_train = split == "train"
    include_masks = getattr(args, "segmentation_head", False)
    aug_config = getattr(args, "aug_config", None)
    scale_jitter = getattr(args, "scale_jitter", True)
    gpu_postprocess = is_gpu_postprocess(resolve_backend_for_build(getattr(args, "augmentation_backend", "cpu")))
    transform_factory = (
        make_coco_transforms_square_div_64 if getattr(args, "square_resize_div_64", False) else make_coco_transforms
    )
    transforms = transform_factory(
        image_set,
        resolution,
        multi_scale=getattr(args, "multi_scale", False),
        expanded_scales=getattr(args, "expanded_scales", False),
        skip_random_resize=not getattr(args, "do_random_resize_via_padding", False),
        patch_size=getattr(args, "patch_size", 16),
        num_windows=getattr(args, "num_windows", 4),
        aug_config=aug_config,
        scale_jitter=scale_jitter,
        gpu_postprocess=gpu_postprocess,
        keypoint_flip_pairs=None,
    )

    cat2label = None if is_train else read_shard_index(root, "train").cat2label()
    logger.info("Building WebDataset %s dataset at resolution %d from %s", image_set, resolution, root)
    return WebDatasetDetection(
        root,
        split,
        transforms=transforms,
        include_masks=include_masks,
        cat2label=cat2label,
        shuffle_buffer=DEFAULT_SHUFFLE_BUFFER if is_train else 0,
        shard_shuffle=DEFAULT_SHARD_SHUFFLE if is_train else 0,
        seed=int(getattr(args, "seed", 0) or 0),
    )


# ----------------------------------------------------------------------------------------------------------------
# Packing CLI
# ----------------------------------------------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the packing entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m rfdetr.datasets.webdataset_io",
        description="Pack a COCO-format split into WebDataset tar shards for sequential-I/O training.",
    )
    parser.add_argument("--image-dir", required=True, help="Directory holding the split's image files.")
    parser.add_argument("--annotations", required=True, help="COCO-format JSON annotation file for the split.")
    parser.add_argument("--output-dir", required=True, help="Directory to write shards and the index into.")
    parser.add_argument("--split", default="train", help="Split name used for shard and index file names.")
    parser.add_argument(
        "--max-shard-mb",
        type=float,
        default=DEFAULT_MAX_SHARD_BYTES / (1024 * 1024),
        help="Approximate shard size in MB.",
    )
    parser.add_argument(
        "--category-ids",
        choices=("remap", "raw"),
        default="remap",
        help="'remap' assigns contiguous 0-based labels; 'raw' keeps the source category_id values.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Pack one COCO split into shards from the command line.

    Args:
        argv: Argument list, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = _build_arg_parser().parse_args(argv)
    index = pack_coco_to_shards(
        args.image_dir,
        args.annotations,
        args.output_dir,
        split=args.split,
        max_shard_bytes=int(args.max_shard_mb * 1024 * 1024),
        category_ids=args.category_ids,
    )
    print(
        f"{index.num_samples} samples -> {len(index.shards)} shard(s) in "
        f"{os.fspath(Path(args.output_dir))} ({index_name(index.split)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
