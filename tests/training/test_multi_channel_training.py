# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""``train()`` with ``num_channels=4`` on RGBA images, and fine-tuning from the checkpoints it writes.

One real one-epoch CPU run of ``RFDETRNano`` on a tiny synthetic dataset whose images carry a fourth channel. A model
whose data loader dropped that channel, or whose training module kept a 3-channel patch embedding, fails the run with a
channel mismatch in the backbone.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr import RFDETR, RFDETRNano
from rfdetr.utilities.reproducibility import seed_all
from tests.conftest import build_synthetic_dataset

#: Not ``RFDETRNano``'s default of 384, to keep the run short.
_RESOLUTION = 224
#: Seed for the untrained model; module-scoped fixtures run before the autouse per-test reseed.
_TRAIN_SEED = 4
#: Enough for a train and a valid split.
_NUM_IMAGES = 16
#: State-dict key suffix of the DINOv2 patch-embedding projection weight.
_PROJ_KEY_SUFFIX = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"


def _add_fourth_channel(dataset_dir: Path) -> None:
    """Rewrite every image of a Roboflow-style COCO dataset as an RGBA PNG with a random fourth channel.

    Args:
        dataset_dir: Dataset root holding ``train``, ``valid`` and ``test`` splits.

    Examples:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> build_synthetic_dataset(root, task="detection", num_images=10)
        >>> _add_fourth_channel(root)
        >>> {Image.open(path).mode for path in (root / "train").glob("*.png")}
        {'RGBA'}
    """
    rng = np.random.default_rng(0)
    for split in ("train", "valid", "test"):
        annotations_path = dataset_dir / split / "_annotations.coco.json"
        content = json.loads(annotations_path.read_text(encoding="utf-8"))
        for image_info in content["images"]:
            source = dataset_dir / split / image_info["file_name"]
            rgb = np.asarray(Image.open(source).convert("RGB"))
            fourth = rng.integers(0, 256, size=(*rgb.shape[:2], 1), dtype=np.uint8)
            target = source.with_suffix(".png")
            Image.fromarray(np.concatenate([rgb, fourth], axis=2), mode="RGBA").save(target)
            if target != source:
                source.unlink()
            image_info["file_name"] = target.name
        annotations_path.write_text(json.dumps(content), encoding="utf-8")


@pytest.fixture(scope="module")
def training_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a 4-channel ``RFDETRNano`` on the CPU for one epoch on RGBA images and return its output directory.

    Examples:
        training_output_dir(tmp_path_factory)
        # A pytest fixture that trains a model; it cannot run standalone.
    """
    seed_all(_TRAIN_SEED)
    dataset_dir = tmp_path_factory.mktemp("multi_channel_dataset")
    build_synthetic_dataset(dataset_dir, task="detection", num_images=_NUM_IMAGES)
    _add_fourth_channel(dataset_dir)
    output_dir = tmp_path_factory.mktemp("multi_channel_training")
    RFDETRNano(pretrain_weights=None, resolution=_RESOLUTION, num_channels=4, device="cpu").train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=4,
        num_workers=0,
        multi_scale=False,
        augmentation_backend="torchvision",
        tensorboard=False,
        run_test=False,
        device="cpu",
    )
    return output_dir


class TestFourChannelTraining:
    """A 4-channel model trains on RGBA images and its checkpoints load back for fine-tuning."""

    @pytest.mark.parametrize("checkpoint_name", ["checkpoint_best_total.pth", "last_ema.pth", "last.ckpt"])
    def test_checkpoint_holds_four_channel_stem(self, training_output_dir: Path, checkpoint_name: str) -> None:
        """Every checkpoint the run writes stores a 4-channel patch-embedding projection."""
        checkpoint = torch.load(training_output_dir / checkpoint_name, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint["model"]

        stems = [tensor for key, tensor in state.items() if key.endswith(_PROJ_KEY_SUFFIX)]

        assert stems
        assert all(stem.shape[1] == 4 for stem in stems)

    @pytest.mark.parametrize("checkpoint_name", ["checkpoint_best_total.pth", "last.ckpt"])
    def test_checkpoint_reloads_with_its_stem_weights(self, training_output_dir: Path, checkpoint_name: str) -> None:
        """Loading a checkpoint as ``pretrain_weights`` restores its projection instead of re-deriving it."""
        checkpoint_path = training_output_dir / checkpoint_name
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint["model"]
        saved_stem = next(tensor for key, tensor in state.items() if key.endswith(_PROJ_KEY_SUFFIX))

        reloaded = RFDETRNano(
            pretrain_weights=str(checkpoint_path), resolution=_RESOLUTION, num_channels=4, device="cpu"
        )

        assert torch.equal(reloaded.model.model.state_dict()[_PROJ_KEY_SUFFIX], saved_stem)

    def test_from_checkpoint_restores_channel_count_and_stem(self, training_output_dir: Path) -> None:
        """``RFDETR.from_checkpoint`` reads ``num_channels`` from the checkpoint and loads its projection."""
        checkpoint_path = training_output_dir / "checkpoint_best_total.pth"
        saved_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]

        reloaded = RFDETR.from_checkpoint(str(checkpoint_path), device="cpu")

        assert reloaded.model_config.num_channels == 4
        assert torch.equal(reloaded.model.model.state_dict()[_PROJ_KEY_SUFFIX], saved_state[_PROJ_KEY_SUFFIX])

    def test_reloaded_model_predicts_on_four_channel_array(self, training_output_dir: Path) -> None:
        """A reloaded checkpoint predicts on an ``(H, W, 4)`` array."""
        reloaded = RFDETRNano(
            pretrain_weights=str(training_output_dir / "checkpoint_best_total.pth"),
            resolution=_RESOLUTION,
            num_channels=4,
            device="cpu",
        )
        image = np.random.default_rng(0).integers(0, 256, size=(96, 128, 4), dtype=np.uint8)

        detections = reloaded.predict(image, threshold=0.0)

        assert detections.xyxy.shape[1] == 4

    def test_checkpoint_into_three_channel_model_raises(self, training_output_dir: Path) -> None:
        """Loading the 4-channel checkpoint into a 3-channel model names both channel counts."""
        checkpoint_path = training_output_dir / "checkpoint_best_total.pth"

        with pytest.raises(ValueError, match="num_channels=4, but this model is configured with num_channels=3"):
            RFDETRNano(pretrain_weights=str(checkpoint_path), resolution=_RESOLUTION, device="cpu")
