# Changelog

All notable changes to RF-DETR are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
RF-DETR uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Evaluation metrics on 1-indexed Roboflow datasets** — `CocoEvaluator._should_use_raw_category_ids` previously used a per-batch label-inspection heuristic that would permanently flip the evaluator into raw-ID mode when any label value coincided with a valid COCO category ID (e.g. due to head reinitialization producing an out-of-range label). This caused incorrect category-ID resolution for all subsequent batches, silently corrupting mAP results. The heuristic is replaced with a stable structural check: the evaluator is in raw-ID mode iff `label2cat` is an identity mapping (`{0: 0, 1: 1, …}`). Datasets with 1-indexed contiguous categories (e.g. `{1, 2, 3, 4}`) are now evaluated correctly regardless of the label values seen in any individual batch. Fixes [#262](https://github.com/roboflow/rf-detr/issues/262).
