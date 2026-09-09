# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pack COCO-format datasets into deterministic WebDataset tar shards.

Purpose: Convert one COCO split into tar shards and its JSON index. Scope: standard-library archive writing,
deterministic generation names and atomic index publication. Usage: call pack_coco_to_shards or the dedicated CLI.
Outputs: tar shards and an index under the requested output directory. Failure: rejects malformed annotations and paths
before publishing a partial index. Used by: rfdetr.cli.webdataset and library callers.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path, PurePath
from typing import Any

from rfdetr.datasets.webdataset.index import (
    DEFAULT_MAX_SHARD_BYTES,
    IMAGE_EXTENSIONS,
    CategoryIdPolicy,
    ShardIndex,
    WebDatasetSplitUnavailableError,
    _shard_name,
    _validate_split_name,
    index_name,
    read_shard_index,
    resolve_within,
)
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_TAR_MEMBER_MODE = 0o644
_TAR_BLOCK_BYTES = 512


def tar_member_bytes(payload_len: int) -> int:
    """Return the on-disk bytes occupied by a POSIX tar member payload.

    Each member adds one header block and rounds its content up to the next
    block. Packing uses this rather than raw payload lengths to respect the
    requested shard-size boundary.

    Args:
        payload_len: Content length of the member, in bytes.

    Returns:
        Total bytes occupied in the tar archive, including header and padding.

    Examples:
        >>> tar_member_bytes(0)
        512
        >>> tar_member_bytes(513)
        1536
    """
    return _TAR_BLOCK_BYTES + -(-payload_len // _TAR_BLOCK_BYTES) * _TAR_BLOCK_BYTES


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


def _shard_content_digest(path: Path) -> str:
    """Return a streamed SHA-256 hex digest of *path*'s bytes.

    Args:
        path: Shard file to digest.

    Returns:
        Full 64-character hex digest.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _pack_generation(payload: dict[str, Any], shard_digests: list[str]) -> str:
    """Derive a short generation token from what was packed.

    Deterministic on purpose: the same split packed twice produces the same token, so re-packing unchanged data
    reproduces the same file names rather than churning the directory. Any change to the sample count, the
    category set, the per-shard sample counts or a shard's actual bytes changes the token, which is what keeps
    a re-pack from writing over the shards a currently published index still references — including the case a
    byte size alone would miss: a changed image or annotation whose re-encoded shard happens to land on the same
    padded tar size as the one it replaces.

    Args:
        payload: The index mapping for this pack, excluding the shard names it is about to name.
        shard_digests: SHA-256 hex digest of each staged shard's content, in order (see
            :func:`_shard_content_digest`).

    Returns:
        Eight hex characters.

    Examples:
        >>> a = _pack_generation({"split": "train", "num_samples": 2}, ["aa", "bb"])
        >>> a == _pack_generation({"split": "train", "num_samples": 2}, ["aa", "bb"])
        True
        >>> a == _pack_generation({"split": "train", "num_samples": 2}, ["aa", "cc"])
        False
    """
    material = json.dumps({"index": payload, "digests": shard_digests}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:8]


def _published_shard_names(destination: Path, split: str) -> set[str]:
    """Return the shard names the currently published index of *split* references.

    Reading the index rather than globbing keeps cleanup scoped to what this split actually published: the
    directory legitimately holds other splits, and a half-written generation from a crashed run must not be
    mistaken for a live one.

    Args:
        destination: Directory holding the pack.
        split: Split whose published index to read.

    Returns:
        Shard file names from the published index, or an empty set when the split has no readable index yet.

    Examples:
        >>> _published_shard_names(Path("/nonexistent"), "train")
        set()
    """
    try:
        return set(read_shard_index(destination, split).shards)
    except (WebDatasetSplitUnavailableError, ValueError, KeyError, json.JSONDecodeError):
        return set()


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
    a sample is never split across two shards. Size accounting counts each member's actual tar footprint (its
    512-byte header plus content padded to the next 512-byte boundary, see :func:`tar_member_bytes`), not just raw
    payload length, so shards land close to *max_shard_bytes* even with many small samples.

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
    if category_ids not in ("remap", "raw"):
        # The CLI constrains this with `choices`, but a library caller can pass anything. Without this the pack
        # succeeds and only fails later, at read time, when ShardIndex.from_json rejects the policy it wrote.
        raise ValueError(f"category_ids must be 'remap' or 'raw', got {category_ids!r}.")
    _validate_split_name(split)

    image_root = Path(image_dir)
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
    unpublished: list[Path] = []

    try:
        for position, image_entry in enumerate(images):
            file_name = str(image_entry["file_name"])
            candidate = PurePath(file_name)
            if candidate.anchor or ".." in candidate.parts:
                # Lexical check only, deliberately not `.resolve()` + `is_relative_to`: `.resolve()` follows
                # symlinks, and a legitimately symlinked image tree (the standard way a large COCO-scale split is
                # shared on a cluster without duplicating storage) would then resolve outside `image_root` and be
                # rejected even though `file_name` itself never leaves it. The actual threat is a crafted
                # `file_name` string, which is lexical by construction.
                raise ValueError(
                    f"Image file_name {file_name!r} in {annotations_path} is absolute or escapes {image_root} "
                    "via '..'; refusing to read a file the split's image directory does not own."
                )
            source = image_root / file_name
            if not source.exists():
                raise FileNotFoundError(
                    f"Image {source} listed in {annotations_path} does not exist; "
                    "packing stops rather than silently dropping it from the training set."
                )
            extension = source.suffix.lower().lstrip(".")
            if extension not in IMAGE_EXTENSIONS:
                raise ValueError(
                    f"Image {source} has extension {extension or '(none)'}, which the shard reader cannot decode "
                    f"(supported: {', '.join(IMAGE_EXTENSIONS)}). Convert the split before packing, so this fails "
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
            shard_bytes += tar_member_bytes(len(payload)) + tar_member_bytes(len(sidecar))
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

        provisional = list(shard_names)
        shard_digests = [_shard_content_digest(work_dir / name) for name in provisional]
        generation = _pack_generation(
            {
                "split": split,
                "num_samples": written,
                "categories": list(coco_data.get("categories", ())),
                "annotated_category_ids": sorted(annotated_ids),
                "category_ids": category_ids,
                "samples_per_shard": shard_sample_counts,
            },
            shard_digests,
        )
        shard_names = [_shard_name(split, position, generation) for position in range(len(provisional))]
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
        # Shard names carry this run's generation token (content-derived, see _pack_generation), so moving them
        # in cannot overwrite a shard the published index still points at. Publication is therefore: move this
        # generation's shards in, swap the index with os.replace (atomic on POSIX), then drop the shards only
        # the previous index referenced. A reader that opened the old index keeps a complete pack until that
        # last step; one that opens the new index sees this generation in full.
        # This narrows, but does not close, the window where a concurrent reader loses files mid-epoch: a
        # reader that opened the old index and has not yet finished reading every shard it names can still hit
        # a missing file once the cleanup step below runs. Closing that fully needs reference-counted or
        # TTL-based garbage collection of old generations across active readers, which is a design decision
        # beyond what this module tracks today (no reader registry exists to know when it is safe to delete).
        previous = _published_shard_names(destination, split)
        for staged, published in zip(provisional, shard_names):
            target = destination / published
            (work_dir / staged).replace(target)
            if published not in previous:
                unpublished.append(target)
        # Staged inside work_dir, not destination: work_dir was created with dir=destination (same filesystem,
        # so the replace() below stays an atomic rename), and this way a failure between write_bytes and
        # replace() leaves the partial file inside work_dir, where the finally block's rmtree cleans it up
        # instead of leaving a stray dotfile behind in destination.
        staged_index = work_dir / f".{index_name(split)}.{generation}"
        staged_index.write_bytes(index_bytes)
        staged_index.replace(destination / index_name(split))
        # The index now owns these shards; a later stale-generation cleanup failure must not remove them.
        unpublished.clear()
        for name in previous - set(shard_names):
            resolve_within(destination, name).unlink(missing_ok=True)
    finally:
        # Roll back only files introduced by this attempt, never a generation the previous index references.
        if unpublished:
            logger.warning("Rolling back %d unpublished shard(s) for split %r.", len(unpublished), split)
        for path in unpublished:
            path.unlink(missing_ok=True)
        if tar is not None:
            tar.close()
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info("Packed %d %s samples into %d shard(s) at %s", written, split, len(shard_names), destination)
    return index
