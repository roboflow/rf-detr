# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared test helpers for the dataset tests.

Plain classes and functions shared by ``test_unreadable_images*.py`` and ``tests/training/test_module_data.py`` so the
fake reader and the on-disk dataset writers exist once.  Import with an absolute import::

    from tests.datasets.helpers import _Reader, _path_of, _write_coco_dataset, _write_yolo_dataset

The retry pause is patched by :func:`_patch_retry_sleep`.  A module that needs it applies it explicitly, as
``sleep = pytest.fixture(autouse=True)(_patch_retry_sleep)``; it is never made autouse for all dataset tests.

The Kornia augmentation tests (``test_kornia_*.py``) share :func:`_only_child`, which picks one child module of a
built pipeline, and :func:`_logged_messages`, which renders the calls recorded on a mocked logger method::

    from tests.datasets.helpers import _logged_messages, _only_child
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from rfdetr.datasets import io_utils


def _patch_retry_sleep(mocker: Any) -> Any:
    """Replace the retry pause so the tests do not wait, and return the mock for tests that check the pauses.

    Apply it per module with ``sleep = pytest.fixture(autouse=True)(_patch_retry_sleep)``.

    Args:
        mocker: The ``pytest-mock`` fixture, or any object whose ``patch.object`` patches an attribute.

    Examples:
        >>> from unittest import mock
        >>> started = []
        >>> class _Patch:
        ...     def object(self, target, attribute):
        ...         patcher = mock.patch.object(target, attribute)
        ...         started.append(patcher)
        ...         return patcher.start()
        >>> class _Mocker:
        ...     patch = _Patch()
        >>> sleep = _patch_retry_sleep(_Mocker())
        >>> _ = io_utils.time.sleep(5)
        >>> sleep.call_args_list
        [call(5)]
        >>> _ = [patcher.stop() for patcher in started]
    """
    return mocker.patch.object(io_utils.time, "sleep")


class _Reader:
    """Stands in for a dataset's image read: fails for chosen indices and records every index it is asked for.

    An index in ``failing`` always raises, unless ``failures`` limits it to that many leading reads.  An index in
    ``flaky`` raises ``exc`` for that many leading reads, then reads fine.

    Args:
        failing: Indices whose read raises, either a set (each raises ``exc``) or a mapping of index to the error
            raised for it.
        exc: Exception raised for an index of a ``failing`` set or of ``flaky``; defaults to
            ``OSError("image file is truncated")``.
        flaky: Number of leading reads that fail, per index, before the read starts to succeed.
        failures: Number of leading reads of an index of ``failing`` that raise; an index left out always raises.

    Examples:
        >>> reader = _Reader(failing={1}, flaky={2: 1})
        >>> reader(0)
        'image-0'
        >>> reader(1)
        Traceback (most recent call last):
            ...
        OSError: image file is truncated
        >>> try:
        ...     reader(2)
        ... except OSError:
        ...     pass
        >>> reader(2), reader.calls
        ('image-2', [0, 1, 2, 2])
        >>> reader = _Reader({1: OSError("short read")}, failures={1: 1})
        >>> try:
        ...     reader(1)
        ... except OSError as exc:
        ...     print(exc)
        short read
        >>> reader(1), reader(2), reader.calls
        ('image-1', 'image-2', [1, 1, 2])
    """

    def __init__(
        self,
        failing: set[int] | Mapping[int, Exception] | None = None,
        exc: Exception | None = None,
        flaky: Mapping[int, int] | None = None,
        failures: Mapping[int, int] | None = None,
    ) -> None:
        default_error = exc or OSError("image file is truncated")
        if isinstance(failing, Mapping):
            self._errors: dict[int, Exception] = dict(failing)
        else:
            self._errors = {index: default_error for index in failing or ()}
        self._leading_failures = dict(failures or {})
        for index, count in (flaky or {}).items():
            if index not in self._errors:
                self._errors[index] = default_error
                self._leading_failures[index] = count
        self.calls: list[int] = []

    def __call__(self, index: int) -> str:
        """Return a stand-in image for ``index``, or raise its error while it still has failures left."""
        self.calls.append(index)
        error = self._errors.get(index)
        if error is not None:
            remaining = self._leading_failures.get(index)
            if remaining is None:
                raise error
            if remaining > 0:
                self._leading_failures[index] = remaining - 1
                raise error
        return f"image-{index}"


def _path_of(index: int) -> str:
    """Return the made-up image path the log and error messages of sample ``index`` name.

    Examples:
        >>> _path_of(3)
        '/data/frame000003.png'
    """
    return f"/data/frame{index:06d}.png"


#: ``(width, height)`` of COCO image ``i`` in :func:`_write_coco_dataset`.  A JPEG decoded with ``draft_size=16`` is
#: reduced by 1, 2, 4 and 8 respectively, each dividing both sides exactly, so every image has its own draft size.
_COCO_IMAGE_SIZES = [(24, 20), (48, 32), (128, 112), (160, 144)]


def _coco_box(index: int) -> list[int]:
    """Return the COCO ``[x, y, width, height]`` box of image ``index``, distinct per image and inside every decode.

    Examples:
        >>> _coco_box(2)
        [12, 12, 16, 20]
    """
    return [4 * (index + 1), 4 * (index + 1), 4 * (index + 2), 4 * (index + 3)]


def _write_coco_dataset(root: Path, extension: str = "jpg", num_images: int = 4) -> Path:
    """Write ``num_images`` images, one box each, and return the annotation file.

    Image ``i`` has the size ``_COCO_IMAGE_SIZES[i]``, the box :func:`_coco_box`, the gray level ``40 * (i + 1)`` and
    the COCO image id ``i + 1``, so a decoded sample and its target identify their source file, and a draft decode
    reduces each source by a different factor.

    Args:
        root: Directory that receives the images and the annotation file.
        extension: File extension, which also selects the encoding.
        num_images: Number of images, at most ``len(_COCO_IMAGE_SIZES)``.

    Raises:
        ValueError: If ``num_images`` exceeds the number of defined image sizes.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     annotation_file = _write_coco_dataset(Path(tmp), num_images=2)
        ...     content = json.loads(annotation_file.read_text(encoding="utf-8"))
        ...     print([image["id"] for image in content["images"]])
        ...     print([(image["width"], image["height"]) for image in content["images"]])
        ...     print(content["annotations"][1]["bbox"])
        ...     print(sorted(path.name for path in Path(tmp).glob("*.jpg")))
        [1, 2]
        [(24, 20), (48, 32)]
        [8, 8, 12, 16]
        ['image0.jpg', 'image1.jpg']
        >>> _write_coco_dataset(Path("unused"), num_images=5)
        Traceback (most recent call last):
            ...
        ValueError: at most 4 images are defined, got 5
    """
    if num_images > len(_COCO_IMAGE_SIZES):
        raise ValueError(f"at most {len(_COCO_IMAGE_SIZES)} images are defined, got {num_images}")
    root.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for index in range(num_images):
        file_name = f"image{index}.{extension}"
        width, height = _COCO_IMAGE_SIZES[index]
        Image.new("RGB", (width, height), color=(40 * (index + 1),) * 3).save(root / file_name)
        images.append({"id": index + 1, "file_name": file_name, "width": width, "height": height})
        box = _coco_box(index)
        annotation = {"id": index + 1, "image_id": index + 1, "category_id": 1, "bbox": box, "area": box[2] * box[3]}
        annotations.append({**annotation, "iscrowd": 0})
    annotation_file = root / "_annotations.coco.json"
    annotation_file.write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "object"}]}),
        encoding="utf-8",
    )
    return annotation_file


def _write_yolo_dataset(root: Path, num_images: int = 3) -> tuple[Path, Path, Path]:
    """Write ``num_images`` 64 x 48 PNG images, image ``i`` holding ``i + 1`` boxes, and return the three paths.

    The paths are the image directory, the label directory and the ``data.yaml`` file, in that order.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     image_dir, label_dir, data_file = _write_yolo_dataset(Path(tmp), num_images=2)
        ...     print(sorted(path.name for path in image_dir.iterdir()))
        ...     print(len((label_dir / "image1.txt").read_text(encoding="utf-8").splitlines()))
        ['image0.png', 'image1.png']
        2
    """
    image_dir, label_dir = root / "images", root / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()
    for index in range(num_images):
        Image.new("RGB", (64, 48), color=(40 * (index + 1),) * 3).save(image_dir / f"image{index}.png")
        (label_dir / f"image{index}.txt").write_text("0 0.5 0.5 0.25 0.25\n" * (index + 1), encoding="utf-8")
    data_file = root / "data.yaml"
    data_file.write_text("names:\n  - object\n", encoding="utf-8")
    return image_dir, label_dir, data_file


def _only_child(pipeline: torch.nn.Module, class_name: str) -> torch.nn.Module:
    """Return the single child module of ``pipeline`` whose class is named ``class_name``.

    Matching on the class name keeps the optional Kornia package out of the test body's imports.

    Args:
        pipeline: Module whose direct children are searched.
        class_name: Exact class name of the wanted child.

    Returns:
        The one matching child; the assertion fails if there is none or more than one.

    Examples:
        >>> _only_child(torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Tanh()), "Tanh")
        Tanh()
    """
    matches = [child for child in pipeline.children() if child.__class__.__name__ == class_name]
    assert len(matches) == 1, f"Expected exactly 1 {class_name}, found {len(matches)}"
    return matches[0]


def _logged_messages(logger_method: Any) -> list[str]:
    """Render each call recorded on a mocked logger method as its final ``%``-formatted message.

    Args:
        logger_method: A mock standing in for ``logger.warning`` or similar.

    Returns:
        One formatted string per recorded call.

    Examples:
        >>> from unittest.mock import MagicMock
        >>> method = MagicMock()
        >>> _ = method("ignores %s", "fill")
        >>> _ = method("plain")
        >>> _logged_messages(method)
        ['ignores fill', 'plain']
    """
    return [
        call.args[0] % call.args[1:] if len(call.args) > 1 else call.args[0] for call in logger_method.call_args_list
    ]
