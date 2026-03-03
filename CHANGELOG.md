# Changelog

All notable changes to RF-DETR are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Fix `AttributeError` crash in `update_drop_path` when the DinoV2 backbone layer structure does not match any known pattern ([#750](https://github.com/roboflow/rf-detr/issues/750)). `_get_backbone_encoder_layers` now returns `None` for unrecognised architectures and `update_drop_path` exits early instead of raising.
- Add warning when `drop_path_rate > 0.0` is configured with a non-windowed DinoV2 backbone, where drop-path is silently ignored.
