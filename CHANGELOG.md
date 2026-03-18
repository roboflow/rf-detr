# Changelog

All notable changes to RF-DETR are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking:** Minimum `transformers` version bumped to `>=5.0.0,<6.0.0`. The DINOv2 windowed-attention backbone now uses the transformers v5 API (`BackboneMixin._init_transformers_backbone()`, removed `head_mask` plumbing). Projects still on transformers v4 must pin `rfdetr<` this release ([#760](https://github.com/roboflow/rf-detr/pull/760)).
- **Breaking:** `draw_synthetic_shape` now returns `Tuple[np.ndarray, List[float]]` instead of `np.ndarray`. The second element is a flat COCO-style polygon list `[x1, y1, x2, y2, …]`. Any caller that previously did `img = draw_synthetic_shape(...)` must be updated to `img, polygon = draw_synthetic_shape(...)` ([#781](https://github.com/roboflow/rf-detr/pull/781)).
- Update Albumentations version requirement to support 1.x and 2.x (`albumentations>=1.4.24,<3.0.0`). `RandomSizedCrop` configs using `height`/`width` are automatically adapted to the `size=(height, width)` API ([#786](https://github.com/roboflow/rf-detr/pull/786)).

### Fixed

- Raise a descriptive `ValueError` instead of a cryptic `RuntimeError` / tensor-size mismatch when a checkpoint is incompatible with the current model architecture — covers `segmentation_head` mismatch (e.g. loading a segmentation checkpoint into `RFDETRNano`) and `patch_size` mismatch ([#810](https://github.com/roboflow/rf-detr/pull/810), closes [#806](https://github.com/roboflow/rf-detr/issues/806)).
- Fix detection head reinitialization logic when loading a fine-tuned checkpoint that has fewer classes than the model's default `num_classes`. Previously the second `reinitialize_detection_head` call would resize the head to `num_classes + 1` after `load_state_dict`, destroying the loaded fine-tuned weights. The second reinit now only fires in the backbone-pretrain scenario (checkpoint has *more* classes than configured), preserving loaded weights for fine-tuned checkpoints ([#815](https://github.com/roboflow/rf-detr/pull/815)).
- Fix `AttributeError` crash in `update_drop_path` when the DinoV2 backbone layer structure does not match any known pattern ([#750](https://github.com/roboflow/rf-detr/issues/750)). `_get_backbone_encoder_layers` now returns `None` for unrecognised architectures and `update_drop_path` exits early instead of raising.
- Add warning when `drop_path_rate > 0.0` is configured with a non-windowed DinoV2 backbone, where drop-path is silently ignored.
- Fix `ValueError: matrix entries are not finite` crash in `HungarianMatcher` when the cost matrix contains NaN or Inf values ([#784](https://github.com/roboflow/rf-detr/issues/784)). Non-finite entries are now replaced with a finite sentinel before `linear_sum_assignment` is called; the warning is emitted at most once per matcher instance to avoid log spam during long training runs.
