# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""WebDataset shard-index schema, validation and on-disk access.

Purpose: Define the stable contract shared by packing and streaming WebDataset splits. Scope: shard names, split
validation, index serialization and loading. Usage: import ShardIndex and read_shard_index from this module. Outputs:
parsed or serialized JSON index data and verified shard-relative paths. Failure: rejects malformed names, incompatible
schemas and unavailable split indexes. Used by: webdataset pack, load, RF-DETR dataset construction and the CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rfdetr.datasets.coco import filter_parent_categories

#: Schema version stamped into every shard index, so a future format change can be rejected with a clear message.
INDEX_VERSION = 1

#: Default shard size target in bytes (~100 MB), the size range WebDataset is tuned for.
DEFAULT_MAX_SHARD_BYTES = 100 * 1024 * 1024

#: Supported image extensions written to and decoded from shards.
IMAGE_EXTENSIONS: tuple[str, ...] = ("jpg", "jpeg", "png", "webp", "bmp")

CategoryIdPolicy = Literal["remap", "raw"]


def resolve_within(base: Path, name: str) -> Path:
    """Resolve *name* under *base*, rejecting any path that escapes it.

    Shard file names come from ordinary on-disk JSON. Both shard reading and
    stale-shard cleanup trust those entries, so traversal must fail before an
    open or unlink reaches a path outside the shard directory.

    Args:
        base: Directory *name* must resolve inside.
        name: Untrusted path-like string taken from a shard index.

    Returns:
        The resolved absolute path to *name* under *base*.

    Raises:
        ValueError: If *name* is absolute or resolves outside *base*.

    Examples:
        >>> resolve_within(Path("/data/shards"), "train-000000.tar").name
        'train-000000.tar'
        >>> resolve_within(Path("/data/shards"), "../../etc/passwd")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: shard entry '../../etc/passwd' resolves outside ...shards.
    """
    base_resolved = base.resolve()
    candidate = (base / name).resolve()
    if not candidate.is_relative_to(base_resolved):
        raise ValueError(f"shard entry {name!r} resolves outside {base}.")
    return candidate


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
        ValueError: If *split* is empty, contains a path separator or a ``.``/``..`` segment, or carries a
            glob metacharacter. The last one matters because the name is also matched against existing shard
            files: a split called ``*`` would match — and then delete — the shards of every other split.

    Examples:
        >>> _validate_split_name("train")
        'train'
        >>> _validate_split_name("../escape")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: split '../escape' must not contain a path separator, drive prefix, or '..'.
    """
    if not split or "/" in split or "\\" in split or ":" in split or split in (".", ".."):
        raise ValueError(f"split {split!r} must not contain a path separator, drive prefix, or '..'.")
    forbidden = sorted(set("*?[]") & set(split))
    if forbidden:
        raise ValueError(
            f"split {split!r} must not contain {''.join(forbidden)}: the name is matched against existing shard "
            "files, so a glob metacharacter would reach other splits' shards."
        )
    return split


def _shard_name(split: str, index: int, generation: str | None = None) -> str:
    """Return the file name of shard *index* of *split*.

    Args:
        split: Split name the shard belongs to.
        index: Zero-based shard number.
        generation: Token identifying one pack run. Included in the name so that re-packing a split writes new
            files instead of overwriting the ones the published index still points at.

    Returns:
        Shard file name.

    Raises:
        ValueError: If *split* would escape the shard directory (see :func:`_validate_split_name`).

    Examples:
        >>> _shard_name("train", 7)
        'train-000007.tar'
        >>> _shard_name("train", 7, "a1b2c3d4")
        'train-a1b2c3d4-000007.tar'
    """
    _validate_split_name(split)
    if generation is None:
        return f"{split}-{index:06d}.tar"
    return f"{split}-{generation}-{index:06d}.tar"


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
            f"Pack the split first: python -m rfdetr.cli.webdataset --split {split} ..."
        )
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    index = ShardIndex.from_json(payload)
    if index.split != split:
        raise ValueError(
            f"{path} is named for split {split!r} but its own recorded split is {index.split!r} — "
            "the index file was likely copied or renamed rather than produced by packing this split; "
            "repack it instead of renaming an existing index."
        )
    return index
