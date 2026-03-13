# Changelog

All notable changes to RF-DETR are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Update Albumentations version requirement to support 1.x and 2.x (`albumentations>=1.4.24,<3.0.0`). `RandomSizedCrop` configs using `height`/`width` are automatically adapted to the `size=(height, width)` API ([#786](https://github.com/roboflow/rf-detr/pull/786)).

### Fixed

- Fix `AttributeError` crash in `update_drop_path` when the DinoV2 backbone layer structure does not match any known pattern ([#750](https://github.com/roboflow/rf-detr/issues/750)). `_get_backbone_encoder_layers` now returns `None` for unrecognised architectures and `update_drop_path` exits early instead of raising.
- Add warning when `drop_path_rate > 0.0` is configured with a non-windowed DinoV2 backbone, where drop-path is silently ignored.
- Fix `ValueError: matrix entries are not finite` crash in `HungarianMatcher` when the cost matrix contains NaN or Inf values ([#784](https://github.com/roboflow/rf-detr/issues/784)). Non-finite entries are now replaced with a finite sentinel before `linear_sum_assignment` is called; the warning is emitted at most once per matcher instance to avoid log spam during long training runs.
