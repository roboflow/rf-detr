# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import json
import shutil
from pathlib import Path
from typing import Any, Generator

import pytest
from fuse_augmentations.data import SplitRatios, generate_dataset

# NOTE: Model weights (rf-detr-*.pth) download to the CWD (typically tests/ when running pytest).
# This is a known limitation. Route: change pretrain_weights default to
# platformdirs.user_cache_dir("rfdetr") in a future PR.
from rfdetr.utilities.reproducibility import seed_all

#: Seed handed to ``generate_dataset``; it draws from fresh entropy when left unset, which would make
#: the session-scoped dataset fixtures differ between runs.
SYNTHETIC_DATASET_SEED = 42


@pytest.fixture(scope="session", autouse=True)
def _prewarm_dinov2_cache() -> None:
    """Download DINOv2 backbone weights once per test session.

    HuggingFace hub uses file-level locking internally, so concurrent xdist
    workers block on each other rather than issuing duplicate network requests.
    After the first worker finishes, all others read from the local disk cache.

    Examples:
        This fixture is autouse — no explicit reference needed in tests.
    """
    import os

    if os.environ.get("RFDETR_SKIP_DINOV2_PREWARM") == "1":
        return

    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            "facebook/dinov2-with-registers-base",
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
    except Exception as exc:
        pytest.skip(f"Skipping DINOv2 cache prewarm (snapshot_download failed): {exc}")


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Reset all RNG sources before every test for reproducibility."""
    seed_all()


def sparsify_category_ids(annotations_path: Path) -> None:
    """Re-encode a COCO JSON in place so category ids are sparse instead of consecutive.

    ``fuse_augmentations`` writes dense 1-based ids (1, 2, 3, …). Real COCO exports leave gaps, and
    rf-detr's ``cat2label`` remapping exists precisely to survive them, so the fixtures spread the ids
    out (``1, 3, 5, …``) to keep exercising that path. Annotation ``category_id`` values are remapped
    with the categories, so the file stays internally consistent.

    Args:
        annotations_path: COCO JSON file to rewrite.

    Examples:
        >>> import json
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "_annotations.coco.json"
        ...     _ = path.write_text(
        ...         json.dumps(
        ...             {
        ...                 "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        ...                 "annotations": [{"id": 1, "category_id": 2}],
        ...             }
        ...         )
        ...     )
        ...     sparsify_category_ids(path)
        ...     content = json.loads(path.read_text())
        >>> [category["id"] for category in content["categories"]]
        [1, 3]
        >>> content["annotations"][0]["category_id"]
        3
    """
    content = json.loads(annotations_path.read_text())
    remap = {category["id"]: index * 2 + 1 for index, category in enumerate(content["categories"])}
    for category in content["categories"]:
        category["id"] = remap[category["id"]]
    for annotation in content.get("annotations", []):
        annotation["category_id"] = remap[annotation["category_id"]]
    annotations_path.write_text(json.dumps(content))


def build_synthetic_dataset(dataset_dir: Path, task: str) -> None:
    """Generate a Roboflow-style synthetic COCO dataset in ``dataset_dir``.

    ``generate_dataset`` writes ``train``/``val`` splits; rf-detr's readers expect ``train``/``valid``/
    ``test``, so ``val`` is renamed and cloned into ``test``. Every split's category ids are sparsified
    by :func:`sparsify_category_ids`.

    Args:
        dataset_dir: Existing directory the dataset is written into.
        task: ``"detection"`` for boxes only, ``"segmentation"`` to also emit polygon annotations.

    Examples:
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmp:
        ...     build_synthetic_dataset(Path(tmp), task="detection")
        ...     sorted(p.name for p in Path(tmp).iterdir())
        ['test', 'train', 'valid']
    """
    generate_dataset(
        output_dir=dataset_dir,
        num_images=100,
        fmt="coco",
        task=task,
        class_mode="shape",
        split_ratios=SplitRatios(train=0.8, val=0.2, test=0.0),
        seed=SYNTHETIC_DATASET_SEED,
        img_size=224,
        min_objects=3,
        max_objects=7,
    )
    valid_dir = dataset_dir / "valid"
    (dataset_dir / "val").rename(valid_dir)
    test_dir = dataset_dir / "test"
    test_dir.mkdir()
    # The test split reuses the valid split's images so its annotations resolve to real files.
    for item in valid_dir.iterdir():
        shutil.copy2(item, test_dir / item.name)
    for split in ("train", "valid", "test"):
        sparsify_category_ids(dataset_dir / split / "_annotations.coco.json")


@pytest.fixture(scope="session")
def synthetic_shape_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, Any, None]:
    """Build a synthetic COCO-style dataset on disk and clean it up after tests.

    Args:
        tmp_path_factory: Pytest factory for temporary directories.

    Yields:
        Path to the synthetic dataset directory.
    """
    dataset_dir = tmp_path_factory.mktemp("synthetic_dataset")
    build_synthetic_dataset(dataset_dir, task="detection")
    yield dataset_dir
    shutil.rmtree(dataset_dir)


@pytest.fixture(scope="session")
def synthetic_shape_segmentation_dataset_dir(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[Path, Any, None]:
    """Build a synthetic COCO-style dataset with polygon segmentation annotations.

    Same layout as :func:`synthetic_shape_dataset_dir` but every annotation includes a ``segmentation`` polygon field so
    the dataset can be used to train or evaluate segmentation models.

    Args:
        tmp_path_factory: Pytest factory for temporary directories.

    Yields:
        Path to the synthetic segmentation dataset directory.
    """
    dataset_dir = tmp_path_factory.mktemp("synthetic_seg_dataset")
    build_synthetic_dataset(dataset_dir, task="segmentation")
    yield dataset_dir
    shutil.rmtree(dataset_dir)
