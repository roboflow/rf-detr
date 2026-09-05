# Changelog

All notable changes to RF-DETR are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `RFDETR.inference()` now accepts `compile_backend="inductor"` as an opt-in backend for long-running inference at a fixed batch size and resolution; the default remains the existing TorchScript path. On CUDA, Inductor completes its two setup invocations and synchronizes inside `inference()` before the first public `predict()` call. Runtime benefit and compatibility depend on the workload, CUDA device, operators, and installed PyTorch version.

### Changed

### Deprecated

### Fixed

- `SetCriterion.loss_masks` no longer reads its normalizing denominator back to the host on every call; `forward` calls `loss_masks` once per matched output layer (final, every aux layer, and enc), so a segmentation training step pays this several times per step (5-7, by model size), not once. `dice_loss` and `sigmoid_ce_loss` declared `num_masks: float`, so `num_boxes` — the distributed-rank all-reduced Tensor `num_boxes_for_targets` returns (segmentation's default path), or a grad-accum-aware override on the keypoint manual-optimization path — was unwrapped with `float(...)` immediately before the TorchScript call boundary; on a lazy backend such as XLA each unwrap is a device-to-host synchronization that cuts the graph. Both signatures now accept `num_masks: Union[Tensor, float, int]` — `loss_masks` always passes the Tensor straight through, removing the synchronization, while a Python `float` or `int` still divides bit-for-bit as before for `lwdetr.py`'s backward-compat re-exports of these functions; a NumPy scalar, which the old single-typed signature happened to accept too, is no longer accepted (an inherent TorchScript `Union`-binding limitation, not exercised by any in-repo caller). The in-repo comment claiming TorchScript rejects a Tensor supplied for a `float` parameter is wrong — it silently converts it inside the scripted function, which is exactly why the signatures had to change for the synchronization to actually go away. On the dtype combination `loss_masks` actually produces (`point_labels` is always `float32`), loss values are unchanged. Measured on a Cloud TPU v6e-1 against this commit, a direct call to each JIT loss with a Tensor denominator went from 1 host read to 0; `loss_masks` itself shared a single host read across both losses before this change, once per matched output layer. ([#1058](https://github.com/roboflow/rf-detr/issues/1058))

- `build_trainer` now selects `XLAStrategy` when the accelerator resolves to XLA/TPU — including default `accelerator="auto"` once it resolves to XLA — and more than one local device is requested. RF-DETR's generic `strategy="auto"` distributed branch otherwise creates `DDPStrategy` before Lightning can apply its XLA-first auto selection, producing `The XLAAccelerator can only be used with a SingleDeviceXLAStrategy or XLAStrategy, found DDPStrategy` during `Trainer` construction. Single-device and one-device-per-host XLA are not promoted; the latter needs separate runtime validation. Explicit strategies are never overridden, keypoint models remain excluded, and segmentation follows the same multi-device guard. ([#1058](https://github.com/roboflow/rf-detr/issues/1058))

- Fixed XLA-marked tests on real TPU hardware: the Trainer-refusal assertion runs only where `XLAAccelerator.is_available()` is false, and the multi-process collective runs only on TPU/NEURON with an uninitialized runtime. CPU-PJRT skips the collective because it has no multi-process path. ([#1058](https://github.com/roboflow/rf-detr/issues/1058))

### Breaking Changes

---

## [1.10.0] — 2026-09-04

### Added

- `TrainConfig.pack_targets` (default `True`) concatenates each batch's per-sample target dicts into one tensor per field before the DataLoader worker-to-main boundary, rebuilding them losslessly on the other side: a batch of 16 crosses as 9 objects, not 114, with bit-identical values. Loaders yield `PackedTargets` when packing is lossless, else the original tuple of dicts. ([#1399](https://github.com/roboflow/rf-detr/pull/1399))
- `TrainConfig.eval_batch_size` decouples the validation/test/predict dataloaders from the training `batch_size`. Default `None` inherits `batch_size`; unlike `batch_size` it accepts no `"auto"`. ([#1378](https://github.com/roboflow/rf-detr/pull/1378))
- `TrainConfig.best_model_metric` (`"map"` or `"mar"`, default `"map"`) ranks checkpoints and early-stopping by mAR instead of mAP. ([#1305](https://github.com/roboflow/rf-detr/pull/1305))
- Training progress bar restored/extended:
    - Restored peak GPU memory (`max_mem`), dropped during the PyTorch Lightning migration (#794). ([#974](https://github.com/roboflow/rf-detr/issues/974))
    - Live free/total GPU memory (`free_mem`) alongside `max_mem` (`trainer.fit()` only). ([#1314](https://github.com/roboflow/rf-detr/issues/1314))
    - Restored `train/lr`, including per-group learning rates. ([#1310](https://github.com/roboflow/rf-detr/pull/1310))
- `deploy_to_roboflow()`:
    - `version` is now optional; when omitted, the highest existing dataset version resolves automatically via the Roboflow API. ([#1116](https://github.com/roboflow/rf-detr/issues/1116))
    - Accepts `ROBOFLOW_HOME` as an alias for the `RF_HOME` weights cache directory. ([#1264](https://github.com/roboflow/rf-detr/pull/1264))
- Experimental, undocumented XLA/TPU training path, not announced in the release notes and not exercised by any 1.10.0 benchmark: `build_trainer()` routes `accelerator="xla"`/`"tpu"` through an `XLAPrecision("bf16-true")` plugin, with a new `xla` optional extra (`torch_xla==2.9.*`, Linux only, py3.10-3.13). ([#1257](https://github.com/roboflow/rf-detr/pull/1257), [#1256](https://github.com/roboflow/rf-detr/pull/1256), [#1254](https://github.com/roboflow/rf-detr/pull/1254))
- Kornia GPU augmentation backend gains seven ops: `ToGray`, `Blur`, `Sharpen`, `Equalize`, `CLAHE`, `Perspective`, `ShiftScaleRotate`. Params Kornia cannot express are warned about, not silently dropped; `HueSaturationValue` remains unsupported. ([#1249](https://github.com/roboflow/rf-detr/pull/1249), [#1277](https://github.com/roboflow/rf-detr/pull/1277), [#1330](https://github.com/roboflow/rf-detr/pull/1330), [#1370](https://github.com/roboflow/rf-detr/pull/1370))
- GPU batched linear-assignment solver (`rfdetr.models._assignment`) wraps `torch_linear_assignment` (Triton-backed), folding every decoder layer's assignment problem into one solve. SciPy's `linear_sum_assignment` remains the CPU/fallback path, and wherever the Triton backend cannot run (non-Linux, compute capability < 8.0, old torch) it falls back internally to that same SciPy solve. New `[train]`-extra dependency `torch-hungarian`, pinned to the `0.1.0rc0` pre-release on PyPI pending a stable `0.1.0`, imported lazily so inference-only installs are unaffected. ([#1368](https://github.com/roboflow/rf-detr/pull/1368))

### Changed

- `RFDETR.predict()` performance work, none of it changing detections: every entry measured byte-identical or checksum-identical against the previous path.
    - Skips the recursive `eval()` reassignment when the module tree is already in eval mode, saving ~0.4-0.5 ms/call on RTX 4060/L4 in the common repeated-inference case. ([#1419](https://github.com/roboflow/rf-detr/pull/1419))
    - Transfers PIL/uint8 NumPy inputs to device in their original byte storage and widens to float on-device, not on host, cutting host-to-device transfer size 4x. ([#1415](https://github.com/roboflow/rf-detr/pull/1415))
    - Converts PIL/uint8 NumPy inputs to contiguous CHW float storage in one fused allocation, not a separate dtype/layout pass. ([#1390](https://github.com/roboflow/rf-detr/pull/1390))
    - `include_source_image=True` converts CUDA float images to `uint8` source bytes on-device before the host transfer, not on CPU. CPU tensors and unsupported CUDA dtypes (e.g. `bfloat16`) keep the previous path. ([#1388](https://github.com/roboflow/rf-detr/pull/1388))
    - Skips the deferred `[0, 1]` pixel-range scan (from #1341) for PIL/uint8 NumPy inputs, since `to_tensor` already guarantees that range for them; tensor and non-uint8 NumPy inputs are unaffected. ([#1387](https://github.com/roboflow/rf-detr/pull/1387))
- Single-feature-level fast paths reuse tensors instead of re-materializing them (current Nano/Small/Medium/Large models; legacy `RFDETRLargeDeprecatedConfig` unaffected where noted); outputs bit-identical:
    - Eager forward pass skips rebuilding the sine position embedding, padding masks, and padded batch tensor when a batch carries no padding, tracked via `NestedTensor.no_padding`; position embeddings are served from a small cache in eval mode. Batches with real padding are unaffected. ([#1416](https://github.com/roboflow/rf-detr/pull/1416))
    - Deformable attention reuses its sampled tensor directly for single-level inputs instead of stack+flatten over a one-element list, mainly benefiting keypoint cross-attention. ([#1385](https://github.com/roboflow/rf-detr/pull/1385))
    - `Transformer.forward` reuses flattened tensors instead of `torch.cat` over a one-element list. ([#1377](https://github.com/roboflow/rf-detr/pull/1377))
    - Decoder's grouped self-attention reuses the regrouped query tensor as the key, not materializing the same grouping twice. ([#1371](https://github.com/roboflow/rf-detr/pull/1371))
- Evaluation:
    - New `TrainConfig.eval_base_model` (default `False`) restores base+EMA validation comparison when only one model is evaluated (see Breaking Changes). `TrainConfig.eval_ema_only` is deprecated, removal in v1.13. ([#1380](https://github.com/roboflow/rf-detr/pull/1380))
    - COCO mAP computation consolidated into a new `rfdetr.training.coco_map.OnePassCocoMeanAveragePrecision` adapter: base and EMA share one evaluation pass, and each image's detection scores convert once, not once per detection. Narrows the `torchmetrics[detection]` pin to `>=1.8.2,<1.9.0`, which validates a TorchMetrics-internal contract this adapter relies on. ([#1375](https://github.com/roboflow/rf-detr/pull/1375), [#1379](https://github.com/roboflow/rf-detr/pull/1379))
    - Shares bbox IoU per image with a unified tie-break contract, plus C=1/no-crowd fast paths. ([#1373](https://github.com/roboflow/rf-detr/pull/1373))
    - mAP metric state kept on CPU, restricted to consumed metrics only; the train hot path is gated on eval epochs. ([#1356](https://github.com/roboflow/rf-detr/pull/1356))
    - Detection validation converts each batch's ground-truth targets once and shares the result between base and EMA mAP accumulators. Segmentation still converts twice, because per-head mask grids can differ. ([#1381](https://github.com/roboflow/rf-detr/pull/1381))
- Segmentation postprocessing, both bit-identical to the previous output:
    - Reads each image's mask resize target once per batch, not per image, cutting CUDA syncs; same fix applied to `COCOEvalCallback._convert_targets`. ([#1369](https://github.com/roboflow/rf-detr/pull/1369))
    - Writes thresholded interpolation chunks directly into a preallocated buffer instead of `torch.cat`-ing a list. Small CUDA selections keep the prior path, for lower peak memory at shipped `num_select=100` defaults. ([#1374](https://github.com/roboflow/rf-detr/pull/1374))
- `SetCriterion.loss_masks` samples matched ground-truth mask labels via direct tensor indexing instead of `point_sample`, under size/contiguity/dtype guards; CUDA keeps the previous path. Measured 6.7-7.1x faster on a single-thread CPU microbenchmark of the full `loss_masks` call, labels bit-identical either way. ([#1367](https://github.com/roboflow/rf-detr/pull/1367))
- `HungarianMatcher` batches host transfers instead of issuing them per problem. ([#1361](https://github.com/roboflow/rf-detr/pull/1361))
- Oversized JPEGs, including 1080p sources, are draft-decoded while preserving draft geometry. ([#1389](https://github.com/roboflow/rf-detr/pull/1389))
- Torch-free NumPy export kernels: bilinear resize made separable, top-k selection partitioned. ([#1394](https://github.com/roboflow/rf-detr/pull/1394), [#1393](https://github.com/roboflow/rf-detr/pull/1393))
- Kornia `GaussianBlur.sigma` default changed `(0.1, 2.0)` → `(0.5, 3.0)` and `GaussNoise.std_range` default changed `(0.01, 0.05)` → `(0.2, 0.44)`, 4-9x stronger, matching Albumentations' defaults. **Silently changes augmentation strength** for any config that omits these params on the Kornia/GPU backend (e.g. `AUG_INDUSTRIAL` reaches the blur default); pin explicit values if you rely on the old strength. ([#1395](https://github.com/roboflow/rf-detr/pull/1395))
- Training skips PyTorch Lightning's pre-training sanity validation batches by default; `num_sanity_val_steps` restores it. Per-microbatch training-loss metrics are compacted, 17 → 9 keys on default `RFDETRSmall`, and `compact_train_metrics=False` restores per-layer keys. LR metrics emit only on optimizer updates, not every microbatch: a no-op at the new `grad_accum_steps=1` default, but ~75% fewer log calls at `grad_accum_steps=4`, the 1.9.x default. ([#1360](https://github.com/roboflow/rf-detr/pull/1360))

### Deprecated

- `rfdetr.datasets.aug_config` compatibility shim now has a concrete removal target: deprecated since 1.9.0, **removal in v1.12.0**. Use `rfdetr.datasets.aug_configs` (plural) instead; constants unchanged. ([#1103](https://github.com/roboflow/rf-detr/pull/1103), [#1037](https://github.com/roboflow/rf-detr/pull/1037))
- `TrainConfig.eval_ema_only` is deprecated, **removal in v1.13**, superseded by `eval_base_model`. Legacy `True`/`False` still migrate to the equivalent `eval_base_model` value with a `FutureWarning`; it still requires `use_ema=True` and conflicts with `eval_base_model=True`. ([#1380](https://github.com/roboflow/rf-detr/pull/1380))

### Fixed

- Packed targets materialize directly into per-sample device tensors instead of clone-after-move, removing a transient CUDA allocation equal to the mask field's size. ([#1405](https://github.com/roboflow/rf-detr/pull/1405))
- Empty COCO targets keep `iscrowd`/`area` dtypes matching populated targets, enabling lossless packed-target transport for mixed empty/populated batches. ([#1404](https://github.com/roboflow/rf-detr/pull/1404))
- Fixed `compile=True` aborting training on supported PyTorch versions, including 2.2. `spatial_shapes` is now built from Python ints under compilation instead of `torch._shape_as_tensor`, which Dynamo could not trace. Eager, `torch.jit.trace`, and the ONNX/TensorRT export path (#1155) are unaffected. ([#1411](https://github.com/roboflow/rf-detr/pull/1411))
- Kornia `CLAHE` reads a scalar `clip_limit` as a range, matching Albumentations, and rejects the same sequences Albumentations rejects. ([#1350](https://github.com/roboflow/rf-detr/pull/1350))
- Corrupt COCO zip downloads are retried, size validated against `Content-Length`, up to 3 attempts with linear backoff, instead of failing the dataset build outright. ([#1306](https://github.com/roboflow/rf-detr/pull/1306))

### Breaking Changes

- **`TrainConfig.grad_accum_steps` now defaults to `1` (was `4`)**, changing the default effective batch size from 16 to 4 — a training-semantics change, not just throughput. **Set `grad_accum_steps=4` explicitly to restore prior behavior.** `batch_size="auto"` runs are unaffected, since the auto-batch probe overwrites `grad_accum_steps`. Measured 27% faster/epoch on one L4 (`batch_size=16, grad_accum_steps=1` vs. the old `4`/`4`), mAP equal within noise. ([#1378](https://github.com/roboflow/rf-detr/pull/1378))
- **Validation now evaluates one model per epoch**, EMA when `use_ema=True` (the default) and base otherwise, instead of both, removing a full validation pass worth ~5% epoch time in one measured L4 run. **Metric keys move**: `val/mAP_*`, `val/mAR`, per-class `val/AP/<class>`, and `val/loss` report whichever model was evaluated, the EMA model by default, instead of always the base model — changing what a `ReduceLROnPlateau` scheduler, `ModelCheckpoint(monitor=...)`, or early stopping watching those keys tracks. `val/ema_*` remains available for explicit EMA consumers. `checkpoint_best_regular.pth` is no longer written when the base model is not evaluated. Set `TrainConfig.eval_base_model=True` to restore the previous base+EMA comparison; `use_ema=False` runs are unaffected. ([#1380](https://github.com/roboflow/rf-detr/pull/1380))
- **Optimizer parameter groups are now one per distinct learning-rate/weight-decay combination** instead of one per parameter (`rfdetr-nano`: 465 → 28 groups), letting fused/foreach AdamW batch properly. AdamW steps are bit-identical and old checkpoints auto-regroup on load, but an explicit `lr_scheduler_kwargs` list sized to the old per-parameter group count, e.g. `LambdaLR`'s per-group `lr_lambda`, must be resized to the new group count. ([#1409](https://github.com/roboflow/rf-detr/pull/1409))
- Dataset builders (`build_roboflow_from_coco`, `build_roboflow_from_yolo`, `build_o365_raw`) now **require** seven image-pipeline options (`square_resize_div_64`, `segmentation_head`, `multi_scale`, `expanded_scales`, `do_random_resize_via_padding`, `patch_size`, `num_windows`; `build_o365_raw` takes no `segmentation_head`) instead of silently substituting contradictory defaults when called with an incomplete config namespace, which could previously train with multi-scale off and the wrong crop scales without warning. Callers passing a complete `TrainConfig`/`ModelConfig` are unaffected; callers assembling a partial namespace by hand must supply every field. ([#1413](https://github.com/roboflow/rf-detr/pull/1413))
- `TrainConfig.log_per_class_metrics` now defaults `False` (was `True`), so per-class AP keys are no longer emitted by default. `TrainConfig.compute_val_loss` now defaults `"auto"` (was `True`), so `val/loss` is computed only when a scheduler/callback consumes it. Set either explicitly to restore the prior unconditional behavior. ([#1372](https://github.com/roboflow/rf-detr/pull/1372))

---

## [1.9.4] — 2026-08-24

### Fixed

- ONNX and TFLite reference inference helpers accept an explicit `background_class_id`: `-1` preserves the existing final-background default, `None` retains every exported logit slot for sparse-ID COCO checkpoints, and `0` supports legacy background-first keypoint checkpoints. ([#1397](https://github.com/roboflow/rf-detr/pull/1397))
- Fixed the TFLite reference inference helper assuming a lone rank-4 output is a segmentation mask. ONNX output names rarely survive the conversion — RF-DETR's own TFLite files arrive as `StatefulPartitionedCall:N` — so a keypoint export's `pred_keypoints` tensor was indistinguishable from a mask by name and was silently upsampled into `Detections.mask`. `rank4_output` now defaults to `None`, decoding only named masks; pass `"masks"` explicitly for a name-stripped segmentation export. ([#1397](https://github.com/roboflow/rf-detr/pull/1397))
- Fixed the torchvision-native non-square training pipeline resampling crop-branch outputs twice. `_build_train_resize_transforms(square=False)` resizes each crop directly to a randomly selected target scale, matching the square and Albumentations paths. This changes the augmented pixel distribution for non-square training by avoiding the fixed `384x384` intermediate and its extra resampling step. Square training, the released default for every shipped model config, is untouched, as are validation, prediction, and export preprocessing. ([#1383](https://github.com/roboflow/rf-detr/pull/1383))
- Fixed custom Albumentations configs treating `TimeReverse` as a pixel-only transform, which flipped images while leaving boxes and keypoints unchanged. `TimeReverse` now shares the geometric-transform and replay-based keypoint handling used by `HorizontalFlip`. The keypoint safety filter disables both `TimeReverse` and `SquareSymmetry` when `keypoint_flip_pairs=[]`; detection-only pipelines (`keypoint_flip_pairs=None`) retain them, and configured pairs enable their keypoint-slot swapping. `SquareSymmetry` already had geometric and replay handling as the alias of `D4`; this fix extends the no-pairs safety filter to it. The default torchvision pipeline is unchanged, as are configs already using the canonical `HorizontalFlip`/`D4` names.
- Fixed TFLite export failing when `onnx2tf` could not resolve the installed `onnxsim` console script from a non-activated virtual environment. `onnx2tf` invokes the bare `onnxsim` name; when that lookup raises `FileNotFoundError` it logs `Failed to optimize the onnx file`, a warning that also appears in working runs, and a stock `RFDETRSmall()` export then failed with `RuntimeError: onnx2tf conversion failed: Output tensors of a Functional model must be the output of a TensorFlow Layer`. RF-DETR now temporarily adds the running interpreter's script directory to `PATH` during conversion. ([#1365](https://github.com/roboflow/rf-detr/issues/1365))
- Fixed the default torchvision-native training pipeline silently corrupting keypoint annotations when `keypoint_flip_pairs` is empty on a schema with genuine left/right pairs. `RandomHorizontalFlip` on this backend always mirrored keypoint x-coordinates when a flip was drawn, but relabeled left/right joints only `if self.keypoint_flip_pairs:` — with an empty list, the pydantic default and one possible outcome when automatic flip-pair inference from dataset metadata misses an asymmetric schema, affected training samples got their keypoints mirrored in position while keeping their original left/right label, with no warning. `_build_torchvision_pipeline` now drops the flip entirely for an empty-but-not-`None` `keypoint_flip_pairs`, logging the warning the Albumentations backend already emits via `filter_keypoint_hflip_augmentations`, worded for this backend's lack of an editable `aug_config`, matching the annotation-safety behavior that backend has had since #1122. An empty list can also legitimately mean the schema has no left/right pairs at all, e.g. a single midline keypoint; the unpatched flip was harmless there since nothing needed relabeling, but this fix disables it there too, for consistency with the Albumentations backend's contract, at the cost of a now-unavailable-by-default augmentation for that narrower case. Detection-only pipelines (`keypoint_flip_pairs=None`) and keypoint pipelines with real pairs are unaffected.
- Fixed `BestModelCallback` treating PyTorch Lightning's pre-training sanity-check validation pass as a real epoch's result. Its EMA-checkpoint tracking and the `smooth_alpha` smoothing accumulator are custom bookkeeping sitting outside `ModelCheckpoint`'s own `trainer.sanity_checking` guard, which the regular-checkpoint path already inherits, so a positive sanity-check score — common when starting a new run initialized with `pretrain_weights` from a checkpoint pretrained on a different dataset — could be written out as the permanent "best" `checkpoint_best_ema.pth` before a single real epoch ran, and real training could then never surpass it. This is distinct from PTL's own `resume`/`ckpt_path` restart, which PTL itself skips the sanity check for (`not val_loop.restarting`). ([#1357](https://github.com/roboflow/rf-detr/pull/1357), fixes [#1348](https://github.com/roboflow/rf-detr/issues/1348))

## [1.9.3] — 2026-08-17

### Changed

- `HungarianMatcher`'s compact-path safety gate computes its target-side half, the box/label finiteness checks, once per training step rather than once per `matcher()` call. `SetCriterion.forward` invokes `matcher()` separately for the final layer, each auxiliary decoder layer, and the encoder layer with the same `targets`, so the target-side precheck is precomputed once and reused across all of them, keyed on `targets` object identity plus `pred_boxes` dtype/device and `num_classes`; a mismatch triggers a fresh computation. Matching results are unchanged. Callers must not mutate `targets` in place between precompute and reuse — the identity check cannot detect that. ([#1340](https://github.com/roboflow/rf-detr/pull/1340))
- Per-class confidence-threshold sweeps in evaluation are O(N log N), not O(T·N): one stable ascending sort per class plus `np.searchsorted` into precomputed suffix sums replaces a full rescan per threshold. NaN scores are explicitly masked so they never count as "above threshold". Results are unchanged. ([#1339](https://github.com/roboflow/rf-detr/pull/1339))
- `RFDETR.predict()` no longer blocks the host on a per-image CUDA sync for its `[0, 1]` pixel-range validation. The range-check tensors are collected unsynced across all images and resolved to Python booleans once, after every image's conversion, range check, and transfer have been queued, so later images' GPU work can overlap the sync. Error-message precedence per image is unchanged. A malformed-rank input combined with `include_source_image=True` now raises a public `ValueError` with a shape message, where it previously surfaced an internal `RuntimeError` from `permute()`. ([#1341](https://github.com/roboflow/rf-detr/pull/1341))
- `Transformer.forward`'s two-stage query selection gathers the `torch.topk`-selected rows *before* running the bbox-delta MLP (`enc_out_bbox_embed`), not after: the MLP is pointwise with no cross-token mixing, so it needs at most the `num_queries` rows that survive selection, not every one of the `sum(H*W)` encoder positions. ([#1334](https://github.com/roboflow/rf-detr/pull/1334))
- `PostProcess` box/mask/keypoint selection is deterministically tie-broken: `torch.argsort(..., stable=True)` plus a slice replaces `torch.topk`, so ties resolve by descending score then ascending flattened query/class index — the rule now shared with the torch-free export decoders, both sides changed together in this PR. Output *ordering* may differ from 1.9.2 when scores tie (same detections, different order; `detections[0]` may change), but ordering among equal scores was never contractual. `PostProcess(num_select=<negative>)` now raises `ValueError` at construction instead of being silently accepted. ([#1320](https://github.com/roboflow/rf-detr/pull/1320))

### Fixed

- Fixed `evaluate(split="test")` on YOLO-format datasets silently evaluating `valid/` instead of the real `test/` split. When no resolvable `test` split exists, evaluation falls back to `valid/` with a logged warning rather than failing; a new `YoloSplitUnavailableError`, a `FileNotFoundError` subclass, drives that fallback and is catchable by callers. If a `test` path is declared in `data.yaml` but unresolvable, or the images directory exists but is empty, or the labels directory is missing, evaluation raises instead of silently relabeling the split as validation. COCO-format Roboflow exports have no such fallback and still raise `FileNotFoundError`; COCO and Objects365 datasets never attempt a `test` split. ([#1329](https://github.com/roboflow/rf-detr/pull/1329), [#1343](https://github.com/roboflow/rf-detr/pull/1343))
- Fixed `metrics.csv` training history being wiped by a resumed run. `build_trainer()` reconstructs a fresh `CSVLogger(version="")` on every start, and PyTorch Lightning's `_ExperimentWriter` deletes any pre-existing `metrics.csv` the first time `.experiment` is accessed, removing every pre-resume row. The file is now snapshotted before that access and restored after, with the writer's column cache seeded so the next `save()` appends instead of overwriting. This is gated on `resume` being set, so reusing an `output_dir` for a fresh, non-resumed run still resets the file instead of appending onto an unrelated run's history. ([#1325](https://github.com/roboflow/rf-detr/pull/1325), closes [#1321](https://github.com/roboflow/rf-detr/issues/1321))
- Fixed `SegmentationHead`'s `skip_blocks` branch skipping the learned `spatial_features_proj` 1×1 convolution; it is now applied before computing mask logits, matching the non-skip branch. This affects the encoder-branch aux mask supervision during training only (`sparse_forward`, `skip_blocks=True`); the export path (`forward_export`) already applied the projection unconditionally, and the main decoder path was already projected, so `predict()` outputs and exported models are unchanged. Custom deployment decoders consuming `sparse_forward`'s `spatial_features` dict entry must not re-apply the projection themselves, since it is now applied upstream. ([#1331](https://github.com/roboflow/rf-detr/pull/1331))
- Fixed non-finite keypoint predictions poisoning the shared box head's gradients, in both the decoder and encoder branches. `compute_l1_keypoint_loss` already guarded its own inputs, but could not zero the local backward pass of a multiply feeding `ref_wh`, shared with the box head, letting a NaN delta propagate through `0.0 * nan == nan`; deltas are now sanitized at the source with `torch.nan_to_num(..., 0.0)` before the reference is composed. The keypoint loss also masks out non-finite predicted keypoints and non-finite target areas rather than letting them poison the loss. Not yet covered: the matcher's own keypoint cost (`compute_keypoint_matching_cost`) still lacks the equivalent guard. ([#1336](https://github.com/roboflow/rf-detr/pull/1336))
- Fixed `batch_size="auto"` probing ignoring AdamW's optimizer-state memory (`exp_avg`/`exp_avg_sq`); it now accounts for it via a shadow optimizer, where previously the probed batch size overshot what real training could fit, causing an out-of-memory error on the first optimizer step. A warning is logged when a non-AdamW optimizer is configured, since the estimate no longer directly applies. The search loop starts from `candidate=2`/`lower_ok=1`, not `1`/`0`. ([#1342](https://github.com/roboflow/rf-detr/pull/1342))
- Fixed the ONNX Runtime export benchmark ignoring the requested `device`: the inference session is built with `providers=[("CUDAExecutionProvider", {"device_id": device})]` instead of the bare provider name, which previously always bound to GPU 0 regardless of `--device N`. ([#1346](https://github.com/roboflow/rf-detr/pull/1346))
- Fixed training metric plots drawing a legend only on the subplot titled "Loss"; every subplot now gets one. ([#1335](https://github.com/roboflow/rf-detr/pull/1335))
- Fixed the ONNX and TFLite reference decoders taking a per-query `argmax`, which silently dropped legitimate detections whenever a query scored above threshold on more than one class; both now mirror `PostProcess`'s multi-label selection. Both paths flatten `(Q, C)` scores into `Q·C` query/class pairs and take the top-scoring pairs before thresholding, via a shared `_select_topk_multiclass` helper using the same deterministic tie rule as `PostProcess`. The selection cap defaults to the exported model's query count; custom exports can pass an explicit value. Empty, zero, negative, and NaN inputs are handled correctly during debug logging. ([#1320](https://github.com/roboflow/rf-detr/pull/1320))
- Fixed EMA training performing an extra averaged-model update at epoch boundaries after the final optimizer step, which let one update per epoch bypass `ema_update_interval` and change the EMA trajectory. ([#1319](https://github.com/roboflow/rf-detr/pull/1319))
- Fixed `model.export(format="tflite")` hanging forever at the ONNX → TFLite conversion step. `onnx`'s C extension and TensorFlow both statically link Abseil and export its symbols as *weak* definitions, which the dynamic loader coalesces onto whichever library loads first. The TFLite route runs a full ONNX export before reaching `onnx2tf`, so ONNX won that race and supplied Abseil's synchronization primitives to TensorFlow, whose executor then blocked forever in `absl::Notification::WaitForNotification()` while restoring the SavedModel bundle: no traceback, no error, 0% CPU, no `.tflite`. TensorFlow is now imported before the ONNX export (`rfdetr.export._backend.preload_tensorflow_before_onnx`), and a warning is logged when the calling process had already imported `onnx` before TensorFlow, e.g. a direct `export_tflite()` call, since that order cannot be repaired in-process. Importing `onnx` *after* TensorFlow is safe and does not warn. ([#1322](https://github.com/roboflow/rf-detr/issues/1322), [#1323](https://github.com/roboflow/rf-detr/pull/1323))

### Breaking Changes

- Exported artifact filenames encode precision or backend for variant-derived/default names: TFLite `{stem}_float32.tflite` / `{stem}_float16.tflite` → `{stem}_fp32.tflite` / `{stem}_fp16.tflite`; ExecuTorch `{variant}.pte` → `{variant}_{backend}.pte` (or `{variant}_qnn_{soc}.pte`); CoreML `{variant}.mlpackage` → `{variant}_fp32.mlpackage` / `{variant}_fp16.mlpackage`; TensorRT `{stem}.trt` → `{stem}_fp16.trt` / `{stem}_fp32.trt`. ONNX filenames are unchanged. Update scripts that hardcode or glob these artifact filenames; explicit `output_name` overrides are unchanged.

## [1.9.2] — 2026-08-11

### Changed

- `HungarianMatcher`'s detection-only cost matrix is built padded to each batch's `max(T_i)` target count and diagonal-extracted, not padded to the cross-image `sum(T_i)`, whenever the batch's targets and predictions pass a fast eligibility check; ineligible batches fall back to the previous full-cartesian computation with identical results. The matcher runs inside the training-step criterion under `torch.no_grad()`, so this is a training-time, not inference-time, saving: on real COCO batches matcher time drops ~51% and peak CUDA memory ~73-76%, and the measured end-to-end training step goes from 288.364 ms to 232.457 ms on an A100. The saving scales with target-count evenness `r = sum(T_i) / max(T_i)`, capped at the batch size, with a `1 - 1/r` ceiling, so a batch where one image holds nearly all the targets (`r` close to 1) sees little to no improvement. The compact path also copies only the diagonal cost blocks to CPU before assignment instead of the full-size matrix, and its safety gate batches its box/label finiteness sweeps into one synchronization, not one per image. ([#1297](https://github.com/roboflow/rf-detr/pull/1297), [#1281](https://github.com/roboflow/rf-detr/pull/1281), [#1312](https://github.com/roboflow/rf-detr/pull/1312))
- `seed_all()` escalates to `torch.use_deterministic_algorithms(True, warn_only=True)` after setting the cuDNN flags, so every op with a deterministic kernel uses it; ops without one, some scatter / `grid_sample` CUDA kernels, warn at execution time instead of raising, and a failure to enable determinism is caught and logged rather than propagating out of `seed_all`. This is user-visible as new runtime warnings and a possible slight performance cost. ([#1307](https://github.com/roboflow/rf-detr/pull/1307))
- `RFDETR.predict()` pins CPU image tensors before the CUDA transfer. ([#1313](https://github.com/roboflow/rf-detr/pull/1313))
- Two-stage query selection avoids materialising repeated top-k gather indices. ([#1278](https://github.com/roboflow/rf-detr/pull/1278))
- Evaluation matching counts labels on the host, not the device. ([#1276](https://github.com/roboflow/rf-detr/pull/1276))
- Keypoint decode skips redundant CUDA presence checks in postprocessing. ([#1282](https://github.com/roboflow/rf-detr/pull/1282))

### Fixed

- Fixed loading a detection checkpoint published before keypoint support warning that `_kp_active_mask` is a "model parameter not in checkpoint (left at random init)". The key is a deterministic schema buffer the model always rebuilds from the configured keypoint schema, empty for detection-only variants, not a learned parameter, so its absence never affected the loaded weights. Affects `Nano`, `Small`, `Large` (2026) and `SegSmall`. The filter matches the exact terminal key, so a similarly-named real parameter still warns, and an *unexpected* `_kp_active_mask` in a checkpoint still warns; the filtered key is recorded at debug level. ([#1302](https://github.com/roboflow/rf-detr/pull/1302))
- Fixed resuming training from one of `BestModelCallback`'s four lightweight checkpoints (`checkpoint_best_regular.pth`, `checkpoint_best_ema.pth`, `checkpoint_best_total.pth`, `last_ema.pth`) silently restarting per-callback state cold; it now restores. Those files intentionally omit optimizer/LR-scheduler state, and a warning says so explicitly, distinguishing them from checkpoints that predate callback-state persistence entirely, where best-score tracking, EMA, and early-stopping all restart cold too. Best-score restore additionally requires the original `output_dir` to match. ([#1318](https://github.com/roboflow/rf-detr/pull/1318))
- Fixed training-time log calls corrupting or duplicating the completed Rich epoch progress bar when `RichProgressBar(leave=True)` is active. A new stream handler tracks the log target by name and re-resolves `stdout`/`stderr` on every emit, following Rich's redirect proxies instead of capturing the pre-redirect stream once at import time. ([#1316](https://github.com/roboflow/rf-detr/pull/1316))
- Fixed an index-less `torch.device("cuda")` never matching an indexed device like `cuda:0` in the deferred-move guard, which re-moved every parameter on every call; it is now normalised to the current device index before the comparison. ([#1311](https://github.com/roboflow/rf-detr/pull/1311))
- Fixed the legacy query-embedding fallback warning on every load; it now warns only when it actually truncates weights. ([#1301](https://github.com/roboflow/rf-detr/pull/1301))
- Fixed `eval_ema_only` runs logging no validation output at all when the base metric was empty. EMA metrics are now computed and logged in that case (`val/ema_mAP_50_95`, `val/ema_mAP_50`, `val/ema_mAR`, per-class AP, and a `val (ema)` summary table), and `val/F1` is no longer silently dropped. The `eval_ema_only` contract is now: `val/mAP_50_95` stays unpopulated, so point `monitor_ema` at `val/ema_mAP_50_95` — a prior comment claiming otherwise has been corrected. ([#1289](https://github.com/roboflow/rf-detr/pull/1289))
- Fixed `ModelContext.reinitialize_detection_head()` raising `AttributeError: 'NoneType'` after `RFDETR.inference(inplace=True)` cleared the weights; it now raises a clear `RuntimeError`, and does so before `args.num_classes` is mutated so a rejected call cannot leave the context half-updated. ([#1283](https://github.com/roboflow/rf-detr/pull/1283))
- Fixed `evaluate()` not building its datamodule from the resolution-override config. ([#1280](https://github.com/roboflow/rf-detr/pull/1280))

### Breaking Changes

- COCO datasets containing an unannotated grouping category no longer spend a model output slot on it. Roboflow COCO exports prepend a synthetic root category (id `0`, `supercategory: "none"`, named after the project) that every real class then lists as its own `supercategory`; it carries no annotations, but previously took label index `0` and an extra class channel. `CocoDetection.cat2label`, the auto-detected `num_classes` and `RFDETR._load_classes()` now share one filter (`rfdetr.datasets.coco.filter_parent_categories`), so training such a dataset builds an *N*-class head instead of *N+1* and every real class shifts down one label index. A parent category that owns annotations keeps its slot, and flat datasets are unaffected. Checkpoints trained before this change keep their *N+1*-class head — evaluating one against the same dataset now misaligns per-class metrics, firing the existing class-count `UserWarning`; retrain. Passing `num_classes` explicitly preserves the checkpoint's *N+1*-class head width so the weights still load, but does not restore the old label indices: `CocoDetection` drops the grouping category whenever `remap_category_ids=True`, so every real class still shifts down one slot and the pretrained head is misaligned against the new labels. The keypoint remapping path (`_build_keypoint_cat2label`) is unchanged, so keypoint datasets still include the grouping category. For hierarchical datasets, the `train`/`valid`/`test` splits now share one label mapping, always derived from the `train` split, so a grouping category annotated in only some splits no longer shifts that split's label indices out from under the others. ([#1303](https://github.com/roboflow/rf-detr/pull/1303))

## [1.9.1] — 2026-08-03

### Changed

- `PostProcess` selects boxes, masks, and keypoints with `index_select`/`expand` instead of materialising a repeated `int64` gather index, an allocation reaching 21–84 MiB per image for the segmentation mask head. Mask post-processing at head resolution is 2.6–3.0× faster; the output is bit-for-bit identical. ([#1268](https://github.com/roboflow/rf-detr/pull/1268))
- `RFDETR.predict()` no longer upsamples segmentation masks whose scores fall below the caller's threshold before discarding them; on typical COCO images only a few of the `num_select` masks survive `threshold=0.5`. End-to-end `predict()` is ~20% faster at 1080p, the saving scaling with image area and neutral at 640 px; the output is unchanged. ([#1265](https://github.com/roboflow/rf-detr/pull/1265))
- ExecuTorch export lowers the `addmm` operations the XNNPACK partitioner leaves undelegated back into `aten.linear` via `AddmmToLinearTransform`, which runs ~100× faster for those shapes. RFDETRNano on XNNPACK / Apple silicon is ~2.5× faster (119.9 → 48.3 ms median); outputs match the previous lowering to ~1e-4. ([#1262](https://github.com/roboflow/rf-detr/pull/1262))

### Fixed

- Fixed `keypoint_flip_pairs` silently disabling horizontal-flip augmentations (`HorizontalFlip`, `Flip`, `D4`) on detection-only datasets when a custom `aug_config` is supplied. `AlbumentationsWrapper.from_config` treats an empty `keypoint_flip_pairs` as "keypoint pipeline with no flip pairs defined" and drops flip transforms for annotation safety; detection pipelines must pass `None` instead of `[]` to keep flips enabled. ([#1248](https://github.com/roboflow/rf-detr/pull/1248))
- Fixed export inference and INT8 calibration resizing through PIL's antialiased BILINEAR/BICUBIC filters, which diverge from `predict()` on downscale and shift exported-model confidence scores and INT8 calibration ranges. The ONNX inference, TFLite inference, INT8 TFLite calibration, and benchmark/traced-example paths now resize with `RFDETR.predict()`'s exact convention: bilinear, half-pixel centers, `antialias=False`. A shared torch-free `_bilinear_resize_half_pixel` NumPy kernel (`rfdetr/export/_resize.py`) mirrors the convention wherever torchvision is unavailable. Re-export any INT8 TFLite model to recalibrate against the corrected pixel distribution. ([#1269](https://github.com/roboflow/rf-detr/pull/1269))
- Fixed `pip install 'rfdetr[onnx]'` on Python 3.10 and `pip install 'rfdetr[executorch]'` on Python 3.14 failing during install. Each extra previously resolved to a version (`onnxruntime`, `executorch`) shipping no wheel for that interpreter and with no source distribution to fall back on; the extras are now gated to interpreters that publish wheels. ([#1267](https://github.com/roboflow/rf-detr/pull/1267))
- Fixed the Kornia augmentation builders (`GaussianBlur`, `GaussNoise`) rejecting scalars for range parameters; they now accept either a scalar or a `(min, max)` pair, matching the Albumentations path. A custom `aug_config` valid under Albumentations no longer raises a bare `TypeError` when `augmentation_backend="cpu"`/`"auto"` resolves to Kornia, i.e. Kornia installed and CUDA available. ([#1255](https://github.com/roboflow/rf-detr/pull/1255))
- Fixed `uv sync` failing to create `.venv`; an `executorch`/`tflite` extra conflict previously blocked resolution of the development environment. ([#1253](https://github.com/roboflow/rf-detr/pull/1253))

### Documentation

- Corrected RF-DETR Keypoint Preview's parameter count (126.4 M → 40.7 M), added deployment parameter-count columns to the keypoint benchmark tables, and clarified that the new SAM 3 RF100-VL result is author-reported rather than measured in SAB. ([#1258](https://github.com/roboflow/rf-detr/pull/1258), [#1261](https://github.com/roboflow/rf-detr/pull/1261))
- Documented ONNX Runtime raw-output decoding and expanded the LLM keypoint task/model/benchmark/API reference. ([#1251](https://github.com/roboflow/rf-detr/pull/1251), [#1260](https://github.com/roboflow/rf-detr/pull/1260))

## [1.9.0] — 2026-07-27

- Default dataset augmentations use torchvision-native transforms **unless Albumentations is installed**, in which case `augmentation_backend="auto"`/`"cpu"`, the default, auto-selects Albumentations instead — identical user code can therefore resolve to a different resize backend, and slightly different pixel values / mAP, purely based on whether `rfdetr[augment]` is installed. Pass `augmentation_backend="torchvision"` to pin torchvision regardless of what is installed. Non-empty custom `aug_config` dictionaries use the optional Albumentations integration and Kornia GPU backend, both via `pip install 'rfdetr[augment]'`. The `[train]` extra no longer installs Albumentations or Kornia. See the migration guide's "Upgrade 1.8 → 1.9" section for remediation steps. ([#1112](https://github.com/roboflow/rf-detr/pull/1112))

### Added

- Native CoreML export: `format="coreml"` on `RFDETR.export()` produces a `.mlpackage` (mlprogram, iOS 16+) directly from `torch.export`, with no ONNX intermediary — distinct from ExecuTorch's `format="executorch", backend="coreml"` `.pte` path. Install with `pip install 'rfdetr[coreml]'` (macOS only; `coremltools>=8.0,<10.0`). ([#1235](https://github.com/roboflow/rf-detr/pull/1235))
- Multi-GPU / multi-node **keypoint (pose) training** under `DistributedDataParallel`. Keypoint models (`RFDETRKeypointPreview`) previously raised `NotImplementedError` for any distributed strategy, `num_nodes > 1`, or `devices > 1`; they now train with `strategy="ddp"` / `strategy="auto"` on multiple GPUs and nodes, launched with `torchrun` exactly like detection models. Because keypoint models use manual optimization, gradients synchronize on every microbatch — keep `grad_accum_steps=1` on multi-GPU for best throughput (`grad_accum_steps > 1` is correct but performs redundant all-reduces). Sharded strategies (FSDP / DeepSpeed) remain unsupported for keypoint models and raise a clear error. See the "Keypoint / Pose models" note in `docs/learn/train/advanced.md`. ([#1232](https://github.com/roboflow/rf-detr/pull/1232))
- `scale_jitter: bool = True` on `TrainConfig` — independent control for the resize → crop → resize branch (Option B) in the training resize pipeline. Disabling this branch previously required passing `aug_config={}`, which also disabled the entire Albumentations augmentation stack; `aug_config` now controls only that stack. Set `scale_jitter=False` to use direct resize only, with annotations near image borders never clipped.
- `AugmentationBackend.TV` (`augmentation_backend="torchvision"`) — forces the torchvision-native default pipeline. Unlike `"cpu"`/`"auto"`, which auto-select the best *installed* backend (Albumentations > Kornia > torchvision) and can therefore resolve differently across environments, `"torchvision"` always resolves to torchvision regardless of what optional packages are installed. `AugmentationBackend` now holds only concrete, directly-usable backends (`TV`, `ALBU`, `KORNIA`); `"cpu"`/`"auto"` remain accepted `augmentation_backend` input strings, resolved lazily at dataset-build time to keep saved configs portable across environments, but are no longer enum members. `AugmentationBackend.TV`/`.ALBU` values changed from `"tv"`/`"albu"` to `"torchvision"`/`"albumentations"`; the old `"tv"`/`"albu"`/`"gpu"` strings are still accepted as legacy input aliases.
- `TrainConfig.optimizer` (`str | Callable`) and `optimizer_kwargs` — configurable training optimizer. `optimizer="adamw"`, the default, keeps RF-DETR's built-in fused `torch.optim.AdamW` path unchanged. A bare short name selects a native `torch.optim` optimizer only (e.g. `"sgd"`, `"adam"`); any other optimizer, including third-party ones such as [`pytorch-optimizer`](https://github.com/kozistr/pytorch_optimizer) (install separately), is selected by a full dotted import path (`"pytorch_optimizer.Lion"`) or a callable / `functools.partial` called with the RF-DETR parameter groups. `optimizer_kwargs` forwards constructor arguments, ignored for callables, which bake their own arguments in. ([#1006](https://github.com/roboflow/rf-detr/pull/1006))
- `TrainConfig.lr_scheduler` (`str | Callable`) plus `lr_scheduler_kwargs`, `lr_scheduler_interval`, and `lr_scheduler_monitor` — configurable LR scheduler, mirroring `optimizer`. `lr_scheduler="step"`/`"cosine"`, the managed presets, keep RF-DETR's built-in warmup-aware schedules unchanged; any other scheduler is selected by a full dotted import path (`"torch.optim.lr_scheduler.OneCycleLR"`) or a callable / `functools.partial` called with the optimizer. Explicit schedulers are built from `lr_scheduler_kwargs` only, with no `total_steps`/`T_max` injected, are auto-wrapped in a `SequentialLR` linear warmup when `warmup_epochs>0`, and step at `lr_scheduler_interval` (`"step"`/`"epoch"`). `ReduceLROnPlateau` is supported end-to-end: it steps once per epoch on the metric named by `lr_scheduler_monitor` (default `"val/loss"`), in both the automatic and manual (keypoint) optimization paths.

### Deprecated

- `TrainConfig.lr_drop` and `lr_min_factor` — pass them through `lr_scheduler_kwargs` instead (`{"lr_drop": ...}` / `{"min_factor": ...}`). Deprecated since v1.9.0, removal in v1.11.0. The fields still work meanwhile and are folded into `lr_scheduler_kwargs` for the managed presets with a `FutureWarning`; default values, e.g. on config reload, do not warn. Set with an explicit, non-managed scheduler they are inert and emit a `FutureWarning`.

### Fixed

- Fixed the keypoint L1-loss helper (`compute_l1_keypoint_loss`) returning detached `new_zeros` on its out-of-schema class-index guard; it now returns **graph-connected** zeros. A detached zero left the keypoint-head parameters without a gradient path on that batch, which desyncs `DistributedDataParallel`'s gradient reducer across ranks (hang or "parameter did not receive grad") when the guard fires on some ranks but not others. This is a prerequisite for the multi-GPU keypoint training above.
- Fixed non-square Albumentations training resize (`aug_config` set, `augmentation_backend` resolving to `"albumentations"`) silently inflating every image's longest side to `max_size`, 1333 by default. `SmallestMaxSize` → `LongestMaxSize` always forces an exact resize in Albumentations, not a conditional cap; a new `CappedLongestMaxSize` internal transform only shrinks, never upscales, matching torchvision's `RandomResize` semantics.
- Fixed explicit `augmentation_backend="albumentations"` resolving successfully without Albumentations installed and failing later, deep in dataset construction; it now raises a clear `ImportError` immediately.
- Fixed `RFDETR.from_checkpoint(..., trust_checkpoint=True)` having no effect. It previously bypassed the safe-load check only for the checkpoint's own metadata read; model construction then silently reloaded the same file through `load_pretrain_weights()` with the unsafe-load default, so the flag did nothing for checkpoints that genuinely needed it and raised the same `RuntimeError` it was supposed to bypass. ([#1239](https://github.com/roboflow/rf-detr/pull/1239))
- Fixed segmentation evaluation resizing ground-truth masks to each image's original resolution before comparison, a lossy round trip vs. the mask head's native grid; GT masks now resize directly to each prediction's own pixel grid, so segm mAP is computed on consistent pixel grids. ([#1241](https://github.com/roboflow/rf-detr/pull/1241))
- Fixed `pip install 'rfdetr[onnx]'` (and `[tflite]`) hanging while building `onnxsim` from source on CPython 3.11/3.13 and Linux aarch64. The previous `onnxsim<0.6.0` pin resolved to 0.5.0, which ships no wheels for those targets, so pip compiled onnxsim's bundled onnxruntime/onnx from source. The constraint is now `onnxsim>=0.7.0`, which publishes prebuilt wheels across CPython 3.10–3.13 on Linux x86_64/aarch64, Windows x86_64, and macOS arm64. ([#1242](https://github.com/roboflow/rf-detr/pull/1242))

### Deprecated

- `RFDETR.optimize_for_inference()` renamed to `RFDETR.inference()`, same signature. The old name is kept as a deprecated alias that forwards to `inference()` and emits a `FutureWarning`. Deprecated since v1.9.0, removal in v1.11.0.

### Changed

- Matched-pair IoU targets in the classification/matching losses compute via `elementwise_box_iou`/`elementwise_generalized_box_iou`, new public helpers in `rfdetr.utilities.box_ops`, instead of `torch.diag(box_iou(...))`. The old path built the full NxN pairwise IoU matrix just to read its diagonal; the new one computes only the N matched pairs directly, reducing peak GPU memory during loss calculation. Both new helpers raise `ValueError` on mismatched-length inputs instead of silently broadcasting. ([#1245](https://github.com/roboflow/rf-detr/pull/1245))
- The `[tensorrt]` extra no longer installs `pycuda`, needed only for `TRTInference`'s async benchmarking mode, which now requires the separate `[tensorrt-bench]` extra (`pip install 'rfdetr[tensorrt-bench]'`); the standard export→engine path (`polygraphy`, no `pycuda`) is unaffected. ([#1246](https://github.com/roboflow/rf-detr/pull/1246))

### Security

- `RFDETR.from_checkpoint()` uses safe deserialization by default (`weights_only=True`) instead of always running full pickle deserialization. Checkpoints containing custom Python objects beyond `argparse.Namespace` or `types.SimpleNamespace` need the new keyword-only `trust_checkpoint: bool = False` parameter set to `True` to opt into the old, unsafe behavior; resume-from-checkpoint during training honors the same flag. ([#1179](https://github.com/roboflow/rf-detr/pull/1179))
- TensorRT export no longer shells out to the `trtexec` CLI — engines are built in-process through the `polygraphy` Python API, removing the subprocess/shell-injection surface entirely. ([#853](https://github.com/roboflow/rf-detr/pull/853))

### Removed

- `[kornia]` extra removed — GPU-side augmentation installs via `[augment]` (`pip install 'rfdetr[augment]'`) instead. There is no `[kornia]` alias extra; `pip install 'rfdetr[kornia]'` will fail.
- `rfdetr.util.*` and `rfdetr.deploy` import paths, deprecated since v1.6.0 with `remove_in="1.9.0"`. Use `rfdetr.utilities.*`, `rfdetr.assets.coco_classes`, `rfdetr.training.drop_schedule`, `rfdetr.training.param_groups`, `rfdetr.visualize.data`, `rfdetr.models.heads.segmentation`, and `rfdetr.export` instead.
- `rfdetr._namespace.build_namespace(model_config, train_config)`, deprecated since v1.7.0 with `remove_in="1.9.0"`. Use `rfdetr.models.build_model_from_config` and `build_criterion_from_config` instead.
- The `train_config` argument to `load_pretrain_weights(nn_model, model_config, train_config)`, deprecated since v1.7.0 with `remove_in="1.9.0"`. Call it with just `(nn_model, model_config)`.
- The `start_epoch`, `do_benchmark`, and `callbacks` keyword arguments to `.train()`/`.evaluate()`, deprecated since v1.7.0 with `remove_in="1.9.0"`. PTL resumes automatically via `resume=`; use the `rfdetr.export.benchmark` module for benchmarking; pass PTL `Callback` objects directly instead of a `callbacks` dict.
- `TrainConfig.group_detr`, `TrainConfig.ia_bce_loss`, `TrainConfig.segmentation_head`, `TrainConfig.num_select`, and `ModelConfig.cls_loss_coef`, deprecated since v1.7.0 with `remove_in="1.9.0"`. `group_detr`, `ia_bce_loss`, `segmentation_head`, and `num_select` now live only on `ModelConfig`; `cls_loss_coef` now lives only on `TrainConfig`.
- `RFDETRLarge`'s automatic silent fallback to `RFDETRLargeDeprecatedConfig` on checkpoint/config incompatibility errors. Loading legacy deprecated-Large weights through `RFDETRLarge` now raises the original error instead of retrying; use `RFDETRLargeDeprecated` directly to load those checkpoints.

---

## [1.8.3] — 2026-06-27

### Added

- `optimize_for_inference(inplace=True)` — new keyword-only argument on `RFDETR.optimize_for_inference()`; skips the deep-copy of the base model for memory-constrained inference-only deployments, ~0.5× model-weight peak memory reduction. Requires `compile=False`. After inplace optimization, `export()` raises `RuntimeError` and `remove_optimized_model()` issues a `UserWarning` and returns cleanly instead of silently clearing state. New `RFDETR.is_optimized_inplace` property returns `True` after a successful inplace optimization. ([#1089](https://github.com/roboflow/rf-detr/pull/1089))
- `CocoKeypointSchema.keypoint_flip_pairs` and `YoloKeypointSchema.keypoint_flip_pairs` fields — horizontal-flip swap pairs inferred automatically from keypoint names (left/right naming convention) for COCO schemas, and from `flip_idx` permutation for YOLO schemas. Auto-populated by `infer_coco_keypoint_schema` and `infer_yolo_keypoint_schema` respectively. ([#1164](https://github.com/roboflow/rf-detr/pull/1164))
- `infer_coco_keypoint_schema` and `infer_yolo_keypoint_schema` re-exported from `rfdetr.datasets`, previously only accessible from `rfdetr.datasets._keypoint_schema`. ([#1164](https://github.com/roboflow/rf-detr/pull/1164))

### Changed

- Horizontal flip detection in `AlbumentationsWrapper` uses Albumentations `ReplayCompose` replay metadata instead of heuristic bbox-center mirroring, eliminating false positives on non-flip transforms that shift box centers. Falls back to `alb.Compose` with a `UserWarning` when `albumentations <1.3` is detected. ([#1164](https://github.com/roboflow/rf-detr/pull/1164))
- Keypoint schema inference supports native COCO format (`dataset_file="coco"`) in addition to `"roboflow"` and `"yolo"`. ([#1164](https://github.com/roboflow/rf-detr/pull/1164))
- `_keypoint_schema_cache` key changed from `dataset_dir` (string) to `(dataset_file, dataset_dir)` tuple, preventing cross-format cache collisions when the same directory is used with different dataset formats. ([#1164](https://github.com/roboflow/rf-detr/pull/1164))

### Fixed

- Fixed unbounded box regression producing negative or out-of-frame coordinates: predicted bounding boxes are clamped to image bounds `[0, width] × [0, height]` in `PostProcess._postprocess_boxes()`. `scale_fct` is also cast to `boxes.dtype` before multiplication, preventing dtype mismatch when boxes are `float16`. ([#1168](https://github.com/roboflow/rf-detr/pull/1168))
- Fixed `SegmentationTrainConfig.cls_loss_coef` default of `5.0`, corrected to `1.0` to restore the pre-v1.7 effective classification loss weight. The `5.0` value was present since v1.6 but dead code until the v1.7 TrainConfig ownership migration activated it, silently over-penalising classification relative to mask losses during segmentation fine-tuning. To reproduce pre-fix behaviour, pass `cls_loss_coef=5.0` explicitly. ([#1165](https://github.com/roboflow/rf-detr/pull/1165))
- Fixed `KeypointTrainConfig.keypoint_nll_loss_coef`, restored to `1.0` to align with the other keypoint loss terms (`keypoint_l1_loss_coef`, `keypoint_findable_loss_coef`, `keypoint_visible_loss_coef`). The previous default of `0.5` was set to dampen OKS@75 oscillation but under-weighted the NLL loss relative to other terms in practice. ([#1165](https://github.com/roboflow/rf-detr/pull/1165))

---

## [1.8.2] — 2026-06-25

### Added

- YOLO pose keypoint dataset support: load Ultralytics YOLO pose datasets (`.yaml` with `kpt_shape`) directly for keypoint fine-tuning. Schema is inferred automatically via `infer_yolo_keypoint_schema`. ([#1156](https://github.com/roboflow/rf-detr/pull/1156))
- `is_bg_first_schema`, `to_active_first`, `to_bg_first`, `schemas_semantically_equal` utilities in `rfdetr.utilities.keypoints`, re-exported from `rfdetr.utilities`, for schema-aware keypoint processing. ([#1160](https://github.com/roboflow/rf-detr/pull/1160))
- `amp_dtype` field on `TrainConfig` (`"auto"` / `"bf16"` / `"fp16"`): pin the mixed-precision autocast dtype instead of relying on device-capability auto-detection. `"auto"`, the default, preserves the historical behaviour — `bf16-mixed` on Ampere+ CUDA, `16-mixed` otherwise. Invalid values degrade gracefully to `"auto"` with a `UserWarning`. ([#1143](https://github.com/roboflow/rf-detr/pull/1143))
- Instance segmentation fine-tuning cookbook (`docs/cookbooks/fine-tune_segmentation.ipynb`) — end-to-end walkthrough using `RFDETRSegSmall` across seven diverse segmentation datasets. ([#1159](https://github.com/roboflow/rf-detr/pull/1159))
- Inference latency benchmark cookbook (`docs/cookbooks/inference-latency-benchmark.ipynb`) — benchmarks CPU/GPU throughput across model sizes with reproducible measurement methodology. ([#1152](https://github.com/roboflow/rf-detr/pull/1152))

### Changed

- Default `num_keypoints_per_class` in `RFDETRKeypointPreviewConfig` changed from `[0, 17]` (background-first) to `[17]` (active-first). Legacy bg-first checkpoints auto-align on load via `_kp_active_mask`. ([#1160](https://github.com/roboflow/rf-detr/pull/1160))

### Fixed

- Fixed `RFDETR.from_checkpoint()` misreading `num_classes` as `shape[0]`, i.e. `num_classes + 1` including the background class, causing `load_state_dict` shape mismatches or a silent extra output class on every load. It now infers `num_classes` and `num_keypoints_per_class` from checkpoint weights, `class_embed.weight.shape[0] - 1` and `_kp_active_mask` respectively. `BestModelCallback._serialize_model_config` is also fixed to persist the correct foreground-only `num_classes`. ([#1158](https://github.com/roboflow/rf-detr/pull/1158))
- Fixed `HungarianMatcher.forward()` hardcoding `0.25` in the focal classification matching cost, silently ignoring any non-default `focal_alpha` passed to the constructor or `build_matcher`; it now uses the configured value. This had misaligned the bipartite matching cost with the focal classification loss in `criterion.py`, which correctly used `self.focal_alpha`. ([#1147](https://github.com/roboflow/rf-detr/pull/1147))
- Fixed `spatial_shapes` in `Transformer.forward()` being built by `torch.empty` + in-place index assignment, which emitted a `ScatterND` feeding a shape tensor (`level_start_index`) that TensorRT rejected with "IScatterLayer cannot be used to compute a shape tensor". It now uses symbolic `Shape` ops, `torch.stack` of per-level `torch._shape_as_tensor` slices. Required to export any RF-DETR model to a TensorRT engine. ([#1155](https://github.com/roboflow/rf-detr/pull/1155))
- Fixed keypoint model inference returning the wrong `class_name` field in predictions. ([#1151](https://github.com/roboflow/rf-detr/pull/1151))
- Fixed silent train-mode inference after the first prediction: `predict()` re-asserts eval mode before each call for unoptimized models. ([#1146](https://github.com/roboflow/rf-detr/pull/1146))
- Fixed TFLite inference preprocessing and mask decoder diverging from PyTorch `predict()` behaviour. ([#1131](https://github.com/roboflow/rf-detr/pull/1131))
- Fixed a Python version mismatch in optional-dependency version overrides. ([#1137](https://github.com/roboflow/rf-detr/pull/1137))

---

## [1.8.1] — 2026-06-19

### Changed

- Config path parameters, e.g. `dataset_dir`, `output_dir`, `pretrain_weights`, accept `pathlib.Path` objects in addition to strings. Paths are coerced to `str` automatically via the `expand_paths` validator. No API changes required; existing string usage unaffected. ([#1124](https://github.com/roboflow/rf-detr/pull/1124))
- Keypoint training disables horizontal flip augmentation until keypoint flip-pair swapping is implemented. Flipping was previously applied without reordering keypoint pairs, producing incorrect labels. ([#1122](https://github.com/roboflow/rf-detr/pull/1122))
- Training metric plots improved with optional seaborn error bands, AP@0.75 metric grouping, and custom AP metric group configuration. ([#1122](https://github.com/roboflow/rf-detr/pull/1122))

### Fixed

- Fixed the keypoint encoder in eval mode splitting `num_queries` queries across all group heads, because `group_detr = len(self.enc_out_keypoint_embed)`; an `if self.training else 1` guard now routes all queries through head 0. ([#1135](https://github.com/roboflow/rf-detr/pull/1135))
- Fixed `config.use_return_dict`, deprecated in `transformers`, replaced with `config.return_dict` in the DINOv2 windowed attention backbone. ([#1135](https://github.com/roboflow/rf-detr/pull/1135))
- Fixed epoch metric tables rendering incorrectly when a Rich progress bar callback is active. Tables print through the progress bar's owned Rich console, preventing cursor conflicts with active live displays. ([#1128](https://github.com/roboflow/rf-detr/pull/1128))
- Fixed spurious keypoint fine-tuning checkpoint switches on noisy OKS metrics: selection is stabilised with smoothed (EMA) best-metric comparison, and smoothing state is correctly restored on training resume. ([#1122](https://github.com/roboflow/rf-detr/pull/1122))
- Fixed Group DETR train-time metric evaluation crashing on non-tensor mask outputs from auxiliary decoder layers; it now evaluates only the primary query group. ([#1122](https://github.com/roboflow/rf-detr/pull/1122))
- Fixed `_detect_horizontal_flip` in the Albumentations transform pipeline using `not bboxes`, which mishandles Albumentations 2.x where bboxes is a NumPy array, falsy even when non-empty; it now uses `len(bboxes) == 0`. ([#1126](https://github.com/roboflow/rf-detr/pull/1126))
- Fixed a crash inside `_log_hyperparams` when `tensorboard` is installed alongside a NumPy-2.0-incompatible `tensorflow`; the TensorBoard logger is now disabled gracefully and training degrades to CSV-only logging with a clear warning. ([#1123](https://github.com/roboflow/rf-detr/pull/1123))

---

## [1.8.0] — 2026-06-13

### Added

- `RFDETRKeypointPreview` — keypoint detection model variant with GroupPose-style head, covariance-based uncertainty (precision-Cholesky parameterization), and COCO keypoint AP evaluation. Public config classes: `KeypointTrainConfig`, `RFDETRKeypointPreviewConfig` (from `rfdetr.config`). Utility: `precision_cholesky_to_pixel_covariance` (from `rfdetr.utilities`). Schema helpers `infer_coco_keypoint_schema`, `CocoKeypointSchema`, `active_keypoint_counts` accessible via `rfdetr.datasets._keypoint_schema`. ([#1099](https://github.com/roboflow/rf-detr/pull/1099))
- `RFDETR.export_for_roboflow(output_dir)` — writes a Roboflow upload bundle (`weights.pt` + `class_names.txt`) without a network call; extracted from `deploy_to_roboflow`, which now delegates to it. ([#1086](https://github.com/roboflow/rf-detr/pull/1086))
- Keypoint fine-tuning cookbook (`docs/cookbooks/fine-tune_keypoints.ipynb`) — end-to-end walkthrough: dataset download, schema inference, `KeypointTrainConfig`, training metrics, and inference with covariance uncertainty. ([#1104](https://github.com/roboflow/rf-detr/pull/1104))
- `MetricKeypointOKS` — reusable OKS metric facade over `CocoEvaluator`, exported from `rfdetr.evaluation`. Supports arbitrary keypoint counts, per-category OKS sigma values, DDP-safe evaluation with first-rank-wins deduplication, and an `OKSKey` enum (`mAP`, `mAP@50`, `mAP@75`, `mAR`) for standardised metric keys. ([#1107](https://github.com/roboflow/rf-detr/pull/1107))

### Changed

- DDP strategy enables `find_unused_parameters=True` for all detection, keypoint, and segmentation models under `strategy='ddp'` or `strategy='auto'` with a distributed launcher, previously segmentation only. Opt out via `trainer_kwargs={"strategy": DDPStrategy(find_unused_parameters=False)}`. ([#1094](https://github.com/roboflow/rf-detr/pull/1094))
- `rfdetr.datasets.aug_config` module renamed to `rfdetr.datasets.aug_configs` (plural). Direct imports from `rfdetr.datasets.aug_config` must be updated; the augmentation preset constants (`AUG_AGGRESSIVE`, etc.) are unchanged. ([#1103](https://github.com/roboflow/rf-detr/pull/1103))

### Removed

- `RFDETR.export(simplify=..., force=...)` — both kwargs removed from the signature. Deprecated since v1.6.0 with `remove_in="1.8.0"`; both were no-ops during the deprecation window. Callers passing these args must remove them before upgrading. ([#1102](https://github.com/roboflow/rf-detr/pull/1102))

### Fixed

- Fixed `RFDETR.from_checkpoint()` treating `num_classes` loaded from the checkpoint as a user-supplied override, which silently refused fine-tuning on a dataset with a different class count — the head refused to re-initialise and trained against the stale class count. An explicit `num_classes` kwarg from the caller still wins over both the checkpoint value and the dataset. ([#1106](https://github.com/roboflow/rf-detr/pull/1106))
- Fixed scale jitter missing from the non-square training crop: `RandomCrop` in the `option_b` branch replaced with `RandomSizedCrop`, restoring the scale-augmentation behaviour lost during the Albumentations migration. ([#1088](https://github.com/roboflow/rf-detr/pull/1088))
- Fixed a multi-GPU validation deadlock in COCO mAP synchronization; `_merge_metric_state_across_ranks` is now safe across zero-batch ranks. ([#1085](https://github.com/roboflow/rf-detr/pull/1085))
- Fixed `import rfdetr` failing on NumPy 2.x when a transitive dependency references the removed `np.complex_` alias. ([#1064](https://github.com/roboflow/rf-detr/pull/1064))
- Fixed the `rfdetr_plus` module availability check giving a false-positive hit when the package was partially installed. ([#1083](https://github.com/roboflow/rf-detr/pull/1083))
- Fixed a spurious "Keypoint class-logit boost has N classes but detection head has M" warning on custom, non-Roboflow keypoint datasets: `_align_num_classes_from_dataset` now zero-pads `num_keypoints_per_class` when auto-adjusting `num_classes` beyond the schema length. ([#1113](https://github.com/roboflow/rf-detr/pull/1113))
- Fixed loss scaling for keypoint training under gradient accumulation (`accumulate_grad_batches > 1`). Keypoint models use manual optimization to normalize losses by the accumulated box count across the effective batch; detection and segmentation remain on Lightning's automatic-optimization path. Optimizer-step scheduling, LR warmup/decay, and epoch-boundary flushing are correctly handled in both paths. ([#1117](https://github.com/roboflow/rf-detr/pull/1117))
- Fixed device auto-detection assigning a CUDA device on a machine with CUDA headers but no GPU driver, which then failed at first use; it now verifies accelerator runtime availability first (PyTorch ≥ 2.4: `torch.accelerator.current_accelerator`; older builds: `torch.cuda.is_available()`). ([#1111](https://github.com/roboflow/rf-detr/pull/1111))
- Fixed `RFDETR.from_checkpoint()` and related APIs silently treating an explicit `num_classes` as unset when its value equals the model default, e.g. 80 for COCO, which refused fine-tuning on a different class count. ([#1109](https://github.com/roboflow/rf-detr/pull/1109))
- Fixed `RFDETR.from_checkpoint()` raising an error or silently loading the wrong model class for starter-like checkpoints without an explicit `pretrain_weights` entry; it now infers the model variant from the checkpoint filename when `pretrain_weights` is absent or unset-like — empty string, `None`, whitespace. ([#1065](https://github.com/roboflow/rf-detr/pull/1065))

---

## [1.7.0] — 2026-04-29

### Added

- `augmentation_backend` field on `TrainConfig` (`"cpu"` / `"auto"` / `"gpu"`): opt-in GPU-side augmentation via [Kornia](https://kornia.readthedocs.io), applied in `RFDETRDataModule.on_after_batch_transfer` once the batch is on the GPU. The CPU path is unchanged and remains the default. Install with `pip install 'rfdetr[augment]'`. ([#1003](https://github.com/roboflow/rf-detr/pull/1003))
- Kornia GPU augmentation supports instance segmentation: images, boxes, and per-instance masks augmented in sync on the GPU, where `augmentation_backend="gpu"/"auto"` was previously ignored silently. New public helper `collate_masks`; `build_kornia_pipeline` gains `with_masks: bool = False`; `unpack_boxes` gains an optional `masks_aug` tensor. **Note**: the mask buffer is `[B, N_max, H, W]` float32, roughly 500 MB at `B=8, N_max=50, H=W=560`; use `augmentation_backend="cpu"` on cards with limited VRAM. ([#1003](https://github.com/roboflow/rf-detr/pull/1003), closes [#997](https://github.com/roboflow/rf-detr/issues/997))
- `BuilderArgs` — a `@runtime_checkable` `typing.Protocol` documenting the minimum attribute set consumed by `build_model()`, `build_backbone()`, `build_transformer()`, and `build_criterion_and_postprocessors()`. Enables static type-checker support for custom builder integrations. Exported from `rfdetr.models`. ([#841](https://github.com/roboflow/rf-detr/pull/841))
- `build_model_from_config(model_config, train_config=None, defaults=MODEL_DEFAULTS)` — config-native alternative to `build_model(build_namespace(mc, tc))`; accepts Pydantic config objects directly and constructs the internal namespace automatically. Exported from `rfdetr.models`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `build_criterion_from_config(model_config, train_config, defaults=MODEL_DEFAULTS)` — config-native alternative to `build_criterion_and_postprocessors(build_namespace(mc, tc))`; returns a `(SetCriterion, PostProcess)` tuple. Exported from `rfdetr.models`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `ModelDefaults` dataclass — exposes the 35 hardcoded architectural constants previously buried inside `build_namespace()`. Pass a `dataclasses.replace(MODEL_DEFAULTS, ...)` override to the new config-native builders to customise individual constants. **Note:** fields may be promoted to `ModelConfig`/`TrainConfig` in future phases. Exported from `rfdetr.models`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `MODEL_DEFAULTS` — the canonical `ModelDefaults` singleton with production defaults. Exported from `rfdetr.models`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `RFDETR.predict(include_source_image=...)` — opt-out flag, default `True`, to skip storing the source image in `detections.metadata["source_image"]`; set `False` to reduce memory use when the image is not needed for annotation. ([#912](https://github.com/roboflow/rf-detr/pull/912))
- `model_name` is stored in checkpoint files during training, so `RFDETR.from_checkpoint()` resolves the model class from the checkpoint without a caller-supplied hint. `strip_checkpoint()` preserves it; checkpoints without it still resolve via `pretrain_weights` filename matching. ([#895](https://github.com/roboflow/rf-detr/pull/895))
- `rfdetr_version` is stored in checkpoint files during training for provenance and compatibility hints. `strip_checkpoint()` preserves it; the key is omitted gracefully when the package version cannot be resolved, and checkpoints without it load normally. ([#918](https://github.com/roboflow/rf-detr/pull/918))
- `notes` parameter on `RFDETR.train()` and `RFDETR.export()` — embed arbitrary JSON-serialisable provenance metadata (labeller, date, class names, etc.) into best-model `.pth` checkpoints, under `checkpoint["args"]["notes"]`, and ONNX files, under the `"rfdetr_notes"` metadata property. String values are stored verbatim; all other types are JSON-encoded. ([#1025](https://github.com/roboflow/rf-detr/pull/1025), closes [#1021](https://github.com/roboflow/rf-detr/issues/1021))
- `RF_HOME` environment variable controls where pretrained weights are cached, default `~/.roboflow/models`. Bare filenames passed as `pretrain_weights`, e.g. `"rf-detr-base.pth"`, resolve relative to it; paths with a directory component are used as-is, parent directories created automatically. ([#130](https://github.com/roboflow/rf-detr/pull/130))
- Grayscale and multispectral imagery support: models accept any channel count, not just 3, with pretrained DINOv2 patch-embedding weights adapted to it at construction time and no extra dependencies. ([#180](https://github.com/roboflow/rf-detr/pull/180), closes [#75](https://github.com/roboflow/rf-detr/issues/75))
- Training configuration is saved to `training_config.json` in the output directory after training, capturing the full `TrainConfig`, `ModelConfig`, effective training parameters, class names, and class count. ([#194](https://github.com/roboflow/rf-detr/pull/194))
- `dinov2_registers_windowed_small` backbone is available as a config option in `ModelConfig.encoder`. ([#236](https://github.com/roboflow/rf-detr/pull/236))
- `rfdetr.from_checkpoint(path)` — new top-level convenience function that loads a checkpoint and infers the correct model subclass automatically, without the caller specifying a class. Equivalent to `RFDETR.from_checkpoint(path)` but importable directly from the `rfdetr` package. ([#664](https://github.com/roboflow/rf-detr/pull/664))
- ONNX export filenames include the model variant name, e.g. `rfdetr-medium.onnx`, instead of the generic `inference_model.onnx`. Exporting multiple variants to the same directory no longer overwrites previous exports. ([#910](https://github.com/roboflow/rf-detr/pull/910))
- Background images, those without a matching label file, are included in YOLO detection datasets as empty-detection samples instead of being dropped; detection and segmentation both use `_LazyYoloDetectionDataset`. ([#915](https://github.com/roboflow/rf-detr/pull/915))
- TFLite export via `model.export(format="tflite")`. Converts through ONNX using `onnx2tf`; FP32 and FP16 outputs are always produced, INT8 quantization is available with a calibration image directory: `model.export(format="tflite", quantization="int8", calibration_data="path/to/images/")`. Requires `pip install 'rfdetr[onnx,tflite]'`. ([#920](https://github.com/roboflow/rf-detr/pull/920))
- PyTorch Lightning `.ckpt` files are accepted as `pretrain_weights`; keys are normalized from PTL format automatically (`state_dict` with `model.`-prefixed keys, `hyper_parameters` → `args`), so weight loading, class-name extraction, and compatibility checks need no manual conversion. ([#951](https://github.com/roboflow/rf-detr/pull/951))
- `skip_best_epochs` parameter for `RFDETR.train()` and `TrainConfig`: the first N epochs are excluded from best-checkpoint selection and early-stopping comparison, preventing strong pretrained weights or resumed checkpoints from locking in a suboptimal early score. ([#1000](https://github.com/roboflow/rf-detr/pull/1000), closes [#789](https://github.com/roboflow/rf-detr/issues/789))
- TFLite inference decodes segmentation mask outputs into `sv.Detections.mask`, upsampled to source size with Pillow bilinear resampling and thresholded at zero, matching `PostProcess.forward`. The mask tensor is detected by output name, `"masks"` substring, with a rank-4 shape fallback. ([#1053](https://github.com/roboflow/rf-detr/pull/1053))
- `PretrainWeightsCompatibilityWarning` — new warning class emitted when a `ModelConfig` override, e.g. custom `encoder`, `num_queries`, or `num_feature_levels`, risks breaking pretrained weight loading. Importable as `from rfdetr.config import PretrainWeightsCompatibilityWarning` for targeted filtering. ([#1017](https://github.com/roboflow/rf-detr/pull/1017))

### Changed

- `peft` is no longer installed as part of the default `rfdetr` package; it moved to the `[lora]` and `[train]` optional extras. For LoRA fine-tuning, install with `pip install 'rfdetr[lora]'`. ([#838](https://github.com/roboflow/rf-detr/pull/838))
- Native RLE annotation support in the COCO segmentation pipeline: `convert_coco_poly_to_mask` explicitly detects and decodes both compressed (string counts) and uncompressed (int-list counts) RLE formats alongside existing polygon support. Malformed annotations now raise instead of being silently swallowed. ([#897](https://github.com/roboflow/rf-detr/pull/897))
- Pinned PyTorch Lightning to exclude known-compromised versions. ([#1020](https://github.com/roboflow/rf-detr/pull/1020))

### Deprecated

- `build_namespace(model_config, train_config)` — no longer used internally and deprecated in this release; use `build_model_from_config`, `build_criterion_from_config`, or `_namespace_from_configs` directly. Removal in v1.9; emits a `DeprecationWarning` on use. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `load_pretrain_weights(nn_model, model_config, train_config)` — the `train_config` positional argument is deprecated, removal in v1.9, and is no longer used internally. Omit it: `load_pretrain_weights(nn_model, model_config)`. Passing a non-`None` value emits a `DeprecationWarning`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- `TrainConfig.group_detr`, `TrainConfig.ia_bce_loss`, `TrainConfig.segmentation_head`, `TrainConfig.num_select` → `ModelConfig`; `ModelConfig.cls_loss_coef` → `TrainConfig`. Each emits `DeprecationWarning` when set on the wrong config object and will be **removed** in v1.9. `SegmentationTrainConfig` users: remove the `num_select` override, the model config value is always used. ([#841](https://github.com/roboflow/rf-detr/pull/841))
- `RFDETRBase` — use `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, or `RFDETRLarge` instead. Emits `FutureWarning` on instantiation; scheduled for removal in v2.0. ([#900](https://github.com/roboflow/rf-detr/pull/900))
- `RFDETRSegPreview` — use `RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, or `RFDETRSegLarge` instead. Emits `FutureWarning` on instantiation; scheduled for removal in v2.0. ([#900](https://github.com/roboflow/rf-detr/pull/900))
- `rfdetr.util` and `rfdetr.deploy` sub-modules are deprecated, removal in v1.9. A `__getattr__` hook on the `rfdetr` package emits a clear `ImportError` with migration guidance when these legacy paths are accessed. ([#839](https://github.com/roboflow/rf-detr/pull/839))

### Fixed

- Fixed TFLite export (`format="tflite"`) producing detection scores that collapse to ~0.02, vs ~0.62 from ONNX; cause was an onnx2tf `GridSample` lowering bug ([PINTO0309/onnx2tf#274](https://github.com/PINTO0309/onnx2tf/issues/274)) compounding through RF-DETR's per-decoder-layer `F.grid_sample`. The converter now passes onnx2tf's pseudo-`GridSample` replacement kwarg, logging a warning when it is absent. ([#1041](https://github.com/roboflow/rf-detr/pull/1041))
- Fixed `WindowedDinov2WithRegistersEmbeddings.forward()` failing silently under `-O` when input spatial dimensions are not divisible by `patch_size * num_windows`; it now raises `ValueError` with a clear message identifying the divisor and actual shape. ([#167](https://github.com/roboflow/rf-detr/pull/167))
- Fixed `_namespace.py`: `num_select` in the builder namespace always reads from `ModelConfig`, where `TrainConfig.num_select` (default 300) silently overrode model-specific values of 100–200 for segmentation variants. ([#841](https://github.com/roboflow/rf-detr/pull/841))
- Fixed `models/weights.py`: `load_pretrain_weights` auto-aligns the model head when the checkpoint has fewer classes than the configured default, preventing a silent mismatch when the caller did not set `num_classes`. ([#845](https://github.com/roboflow/rf-detr/pull/845))
- Fixed `models/weights.py`: `load_pretrain_weights` slices `refpoint_embed.weight` and `query_feat.weight` per-group when reshaping checkpoint queries; the previous flat slice scrambled groups 1+ when `num_queries` decreased with `group_detr > 1`, corrupting training-resume. Inference, which reads group 0 only, was unaffected. ([#1019](https://github.com/roboflow/rf-detr/pull/1019))
- Fixed YOLO segmentation training on large datasets hitting OS out-of-memory, caused by `supervision.DetectionDataset.from_yolo(force_masks=True)` eager-rasterising every image's masks at construction time. A new `_LazyYoloDetectionDataset` stores polygons and defers rasterisation to `__getitem__`, keeping RAM proportional to annotation count. ([#851](https://github.com/roboflow/rf-detr/pull/851))
- Fixed ONNX/TRT dynamic batch inference: the tracer baked the training batch size as a compile-time constant, so TRT engines built with smaller `--minShapes` failed with `Reshape: reshaping failed`. Six call sites in `gen_encoder_output_proposals` and `Transformer.forward` now use ONNX-symbolic equivalents, keeping the batch dimension dynamic. ([#950](https://github.com/roboflow/rf-detr/pull/950), closes [#949](https://github.com/roboflow/rf-detr/issues/949))
- Fixed training failure when `square_resize_div_64=False`: the non-square resize pipeline did not guarantee dimensions divisible by `patch_size * num_windows`, raising `ValueError`. A `PadIfNeeded` step is appended after the resize pair in the train and val/test pipelines. ([#991](https://github.com/roboflow/rf-detr/pull/991), closes [#983](https://github.com/roboflow/rf-detr/issues/983))
- Fixed non-square batch padding: `block_size` rounding is applied in the DataLoader collator as well as the transform-level `PadIfNeeded`, so divisibility by `patch_size * num_windows` survives `Compose` reordering and applies to custom evaluation harnesses. ([#992](https://github.com/roboflow/rf-detr/pull/992))
- Fixed `RFDETRModelModule.on_load_checkpoint` crashing with `RuntimeError` when resuming from a checkpoint saved at a different image resolution; DINOv2 positional embeddings are bicubic-interpolated to `model_config.positional_encoding_size` first. ([#1002](https://github.com/roboflow/rf-detr/pull/1002), closes [#998](https://github.com/roboflow/rf-detr/issues/998))
- Fixed `RFDETRLarge` initialization showing two conflicting `ValueError`s, for `patch_size=14` and `patch_size=16`, when the deprecated-config fallback retry also fails; the fallback re-raises the original error without chained context. ([#975](https://github.com/roboflow/rf-detr/pull/975))
- Fixed `RFDETRModelModule.__init__` crashing with `RuntimeError: size mismatch for backbone.0.encoder.encoder.embeddings.position_embeddings` when training segmentation models at a custom resolution, e.g. `RFDETRSegLarge(resolution=1008)`; the training entry path delegates to `load_pretrain_weights`, which interpolates the positional embeddings. ([#1040](https://github.com/roboflow/rf-detr/pull/1040), closes [#1038](https://github.com/roboflow/rf-detr/issues/1038), [#1023](https://github.com/roboflow/rf-detr/issues/1023))
- Fixed TFLite detection scores collapsing for all queries when `GridSample` was used as an onnx2tf pseudo-operator; the node is rewritten to `Gather`-based integer-index arithmetic before conversion. Supersedes the runtime-kwarg approach in [#1041](https://github.com/roboflow/rf-detr/pull/1041). ([#1054](https://github.com/roboflow/rf-detr/pull/1054))
- Fixed `class_name` lookup for pretrained COCO models: sparse COCO category IDs, 1–90 for 80 classes, made flat 0-based indexing return the wrong name. Detection uses a `coco_id → class_name` mapping built from `COCO_CLASSES`; fine-tuned models keep direct 0-based indexing. ([#1051](https://github.com/roboflow/rf-detr/pull/1051))

---

## [1.6.5] — 2026-04-22

### Breaking Changes

- `predict()` stores the source image in `detections.metadata["source_image"]`, not `detections.data["source_image"]`, which supervision indexed per-detection and raised `IndexError` on. Update any code that reads `detections.data["source_image"]`. ([#972](https://github.com/roboflow/rf-detr/pull/972), [#968](https://github.com/roboflow/rf-detr/issues/968))

### Fixed

- Fixed segmentation training crash on T4 and P100 GPUs, caused by cuDNN engine selection for depthwise convolution backward on some CUDA stacks. A custom `autograd.Function` disables cuDNN in forward and backward. ([#967](https://github.com/roboflow/rf-detr/pull/967))
- Fixed `ema_segm_mAP_50_95` and `ema_segm_mAP_50` being computed from the base, non-EMA, metric accumulator instead of the EMA accumulator, producing misleading validation scores for segmentation models. ([#980](https://github.com/roboflow/rf-detr/pull/980))
- Fixed `BestModelCallback` losing the best EMA score on training resume, because `_best_ema` was not persisted in `state_dict()`. ([#973](https://github.com/roboflow/rf-detr/pull/973))
- Fixed `positional_encoding_size` not updating when `resolution` is set at construction time, e.g. `RFDETRLarge(resolution=640)`, causing shape mismatches during forward. A model validator now auto-syncs PE size. ([#956](https://github.com/roboflow/rf-detr/pull/956))
- Fixed a pretrained weight loading crash with custom resolution: DINOv2 positional embeddings are bicubic-interpolated to match the target grid before `load_state_dict`. ([#964](https://github.com/roboflow/rf-detr/pull/964))
- Fixed `validate_checkpoint_compatibility` producing a cryptic `RuntimeError` on `patch_size` mismatch when the checkpoint lacks explicit `args.patch_size`; it now infers `patch_size` from the DINOv2 projection weight shape and raises a descriptive `ValueError`. ([#971](https://github.com/roboflow/rf-detr/pull/971))
- Fixed `predict()` storing `detections.data["source_shape"]` as a Python `tuple`, which raised `TypeError` whenever `sv.Detections` was iterated. The value is now an `np.ndarray` of shape `(N, 2)` and dtype `int64`. ([#966](https://github.com/roboflow/rf-detr/pull/966), [#963](https://github.com/roboflow/rf-detr/issues/963))
- Fixed `predict()` emitting a misleading "class_id out of range" warning for the background/no-object class, class index `num_classes`. Background-class detections map `data["class_name"]` to `"__background__"` without any warning. ([#970](https://github.com/roboflow/rf-detr/issues/970))

## [1.6.4] — 2026-04-10

### Changed

- `predict()` includes `class_name` in `detections.data`, mapping each detection's 0-indexed class ID to its human-readable name. ([#914](https://github.com/roboflow/rf-detr/pull/914))

### Fixed

- Fixed segmentation multi-GPU DDP training crashing with `RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss`, because the segmentation head's `sparse_forward()` leaves parameters unused on some steps: `build_trainer()` wraps `strategy="ddp"` with `DDPStrategy(find_unused_parameters=True)` when `segmentation_head=True`. Non-segmentation DDP and other strategies are unchanged. ([#942](https://github.com/roboflow/rf-detr/pull/942), [#947](https://github.com/roboflow/rf-detr/pull/947))
- Fixed fused AdamW crashing under FP32 multi-GPU training with `RuntimeError: params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout`: `configure_optimizers()` and `clip_gradients()` gate fused AdamW on the trainer's actual precision, not GPU capability, which reports BF16 support on Ampere+ even at `precision="32-true"`. ([#942](https://github.com/roboflow/rf-detr/pull/942), [#947](https://github.com/roboflow/rf-detr/pull/947))
- Fixed multi-GPU DDP training crashing in Jupyter notebooks and Kaggle: the fork-based `ddp_notebook` strategy is replaced with a spawn-based one, avoiding OpenMP thread pool corruption after `fork()`. ([#928](https://github.com/roboflow/rf-detr/pull/928))
- Fixed `RFDETR.train(resolution=...)` being silently ignored; the kwarg is applied to `model_config` before training begins, with validation that the value is divisible by `patch_size * num_windows`. ([#933](https://github.com/roboflow/rf-detr/pull/933))
- Fixed `save_dataset_grids` being silently a no-op; `DatasetGridSaver` is wired into the training loop, saving sample grids to `{output_dir}/dataset_grids/` when enabled. Grid save failures are caught without interrupting training. ([#946](https://github.com/roboflow/rf-detr/pull/946))
- Fixed partial gradient-accumulation windows at the tail of training epochs: the training dataset is padded to an exact multiple of `effective_batch_size * world_size`, so every optimizer step uses a full gradient window. Workaround for [pytorch-lightning#19987](https://github.com/Lightning-AI/pytorch-lightning/issues/19987). ([#937](https://github.com/roboflow/rf-detr/pull/937))
- Fixed `torch.export.export` failing on the transformer decoder, by threading `spatial_shapes_hw` through all decoder layers. ([#936](https://github.com/roboflow/rf-detr/pull/936))
- Fixed `download_pretrain_weights()` overwriting fine-tuned checkpoints that share a filename with a registry model, e.g. `rf-detr-nano.pth`, where an MD5 mismatch silently restored the original COCO checkpoint. It now returns early whenever the file exists and `redownload=False`, warning when the hash differs; pass `redownload=True` to force a fresh download. ([#935](https://github.com/roboflow/rf-detr/pull/935))

## [1.6.3] — 2026-04-02

### Changed

- `predict()` stores the original image and its shape on returned `sv.Detections` objects — `detections.data["source_image"]` (NumPy array) and `detections.data["source_shape"]` (NumPy array of shape `(N, 2)`, each row `[height, width]`) let you annotate results without loading the image separately. ([#892](https://github.com/roboflow/rf-detr/pull/892))
- `RFDETR.train()` auto-detects `num_classes` from the dataset directory when not explicitly set, reinitializing the detection head to the correct class count automatically. A warning is emitted when the configured value differs from the dataset count. ([#893](https://github.com/roboflow/rf-detr/pull/893))
- `optimize_for_inference()` accepts dtype as a string name, e.g. `"float16"`, in addition to a `torch.dtype` object; invalid dtype inputs uniformly raise `TypeError`. ([#899](https://github.com/roboflow/rf-detr/pull/899))

### Fixed

- Fixed `models/lwdetr.py`: `reinitialize_detection_head` replaces `nn.Linear` modules instead of mutating `.data` in place, keeping `out_features` consistent with the weight shape, so ONNX export and `torch.jit.trace` no longer emit stale class counts for fine-tuned models. ([#904](https://github.com/roboflow/rf-detr/pull/904))
- Fixed `RFDETR.optimize_for_inference()` leaking a CUDA context on multi-GPU setups: the deep-copy, export, and JIT-trace steps run inside `torch.cuda.device(device)` to pin the context to the correct device. ([#899](https://github.com/roboflow/rf-detr/pull/899))
- Fixed `optimize_for_inference()` leaving inconsistent state on failure: prior optimized state is reset and flags are committed only after a successful build/trace; temp download files use unique per-process paths to avoid parallel worker collisions.
- Fixed `deploy_to_roboflow` failing with `FileNotFoundError` after the PyTorch Lightning migration: `class_names.txt` is written to the upload directory and `args.class_names` is populated before saving the checkpoint. ([#890](https://github.com/roboflow/rf-detr/pull/890))

## [1.6.2] — 2026-03-27

### Added

- `RFDETR.predict(shape=...)` — optional `(height, width)` tuple overrides the default square inference resolution; useful when matching a non-square ONNX export. Both dimensions must be positive integers divisible by `patch_size × num_windows` as determined by the model configuration. ([#866](https://github.com/roboflow/rf-detr/pull/866))

### Changed

- `ModelConfig.device` and `RFDETR.train(device=...)` accept `torch.device` objects and indexed device strings such as `"cuda:0"`. Values are normalized to canonical torch-style strings. `RFDETR.train()` warns when an unmapped device type is passed to PyTorch Lightning auto-detection. ([#872](https://github.com/roboflow/rf-detr/pull/872))

### Fixed

- Fixed ONNX export ignoring an explicit `patch_size` argument: `export()` and `predict()` resolve `patch_size` from `model_config` by default, validate it strictly (positive integer, not bool), and enforce that `(H, W)` dimensions are divisible by `patch_size × num_windows`. ([#876](https://github.com/roboflow/rf-detr/pull/876))
- Fixed ONNX export for models with dynamic batch dimensions: `H_.expand(N_)` replaced with `torch.full` for Python-int spatial dims, eliminating tracer failures. ([#871](https://github.com/roboflow/rf-detr/pull/871))

## [1.6.1] — 2026-03-25

### Deprecated

- `RFDETR.export(..., simplify=..., force=...)` — both arguments are now no-ops and emit a `DeprecationWarning`. RF-DETR no longer runs ONNX simplification automatically; remove these arguments from your calls. Removal in v1.8. ([#861](https://github.com/roboflow/rf-detr/pull/861))

### Fixed

- Fixed `RFDETR.train()` raising a bare `ModuleNotFoundError` on a missing `rfdetr[train]` install; it now raises an `ImportError` naming the fix, `pip install "rfdetr[train,loggers]"`. ([#858](https://github.com/roboflow/rf-detr/pull/858))
- Fixed `AUG_AGGRESSIVE` preset: `translate_percent` `(0.1, 0.1)` was a degenerate range forcing `Affine` to always translate right/down by exactly 10%, corrected to `(-0.1, 0.1)`. ([#863](https://github.com/roboflow/rf-detr/pull/863))
- Fixed the PTL training path: `latest.ckpt` and per-interval checkpoints (`checkpoint_interval_N.ckpt`) are written and restored on resume. ([#847](https://github.com/roboflow/rf-detr/pull/847))
- Fixed `BestModelCallback` and checkpoint monitor raising `MisconfigurationException` on non-eval epochs when `eval_interval > 1`; monitor key absence is handled gracefully. ([#848](https://github.com/roboflow/rf-detr/pull/848))
- Fixed the `protobuf` version constraint in the `loggers` extra, guarding against the TensorBoard descriptor crash (`TypeError: Descriptors cannot be created directly`) with protobuf ≥ 4. ([#846](https://github.com/roboflow/rf-detr/pull/846))
- Fixed duplicate `ModelCheckpoint` state keys when `checkpoint_interval=1`; `last.ckpt` is omitted in that configuration to avoid collision. ([#859](https://github.com/roboflow/rf-detr/pull/859))

## [1.6.0] — 2026-03-20

### Added

- PyTorch Lightning training building blocks: `RFDETRModelModule`, `RFDETRDataModule`, `build_trainer()`, and callbacks (`RFDETREMACallback`, `COCOEvalCallback`, `BestModelCallback`, `DropPathCallback`, `MetricsPlotCallback`) — standard PTL components, swap/subclass/extend any piece. Level 3: `rfdetr fit --config` CLI, zero Python required. ([#757](https://github.com/roboflow/rf-detr/pull/757), [#794](https://github.com/roboflow/rf-detr/pull/794))
- Multi-GPU DDP via `model.train()`: `strategy`, `devices`, and `num_nodes` added to `TrainConfig`; single-GPU behaviour unchanged when omitted. ([#808](https://github.com/roboflow/rf-detr/pull/808))
- `batch_size='auto'`: CUDA memory probe finds the largest safe micro-batch size, then recommends `grad_accum_steps` to reach a configurable effective batch target, default 16 via `auto_batch_target_effective`. ([#814](https://github.com/roboflow/rf-detr/pull/814))
- `ModelContext` promoted from `_ModelContext` to a public, exported API — inspect `class_names`, `num_classes`, and related metadata via `model.context` after training. ([#835](https://github.com/roboflow/rf-detr/pull/835))
- `backbone_lora` and `freeze_encoder` added as first-class fields in `ModelConfig`. ([#829](https://github.com/roboflow/rf-detr/pull/829))
- `generate_coco_dataset(with_segmentation=True)` produces COCO polygon annotations alongside bounding boxes for segmentation fine-tuning with synthetic data. ([#781](https://github.com/roboflow/rf-detr/pull/781))
- `set_attn_implementation("eager" | "sdpa")` on the DINOv2 backbone — switch attention implementation at runtime. ([#760](https://github.com/roboflow/rf-detr/pull/760))
- `eval_max_dets`, `eval_interval`, and `log_per_class_metrics` added to `TrainConfig`.
- `python -m rfdetr` entry point alongside the `rfdetr` console script.
- `py.typed` marker — RF-DETR is now PEP 561–compliant.

### Changed

- **Breaking:** Minimum `transformers` version bumped to `>=5.1.0,<6.0.0`. The DINOv2 windowed-attention backbone uses the transformers v5 API (`BackboneMixin._init_transformers_backbone()`, removed `head_mask` plumbing). Projects still on transformers v4 must pin `rfdetr<1.6.0`. ([#760](https://github.com/roboflow/rf-detr/pull/760))
- **Breaking:** PyPI install extras renamed — `rfdetr[metrics]` → `rfdetr[loggers]`, `rfdetr[onnxexport]` → `rfdetr[onnx]`.
- `draw_synthetic_shape` returns `Tuple[np.ndarray, List[float]]`, not `np.ndarray`. The second element is a flat COCO-style polygon list `[x1, y1, x2, y2, …]`. Any caller that did `img = draw_synthetic_shape(...)` must be updated to `img, polygon = draw_synthetic_shape(...)`. ([#781](https://github.com/roboflow/rf-detr/pull/781))
- Albumentations version constraint broadened to `>=1.4.24,<3.0.0`; `RandomSizedCrop` configs using `height`/`width` kwargs are adapted automatically to the 2.x `size=(height, width)` API. ([#786](https://github.com/roboflow/rf-detr/pull/786))
- Current learning rate is shown in the training progress bar alongside loss. ([#809](https://github.com/roboflow/rf-detr/pull/809))
- `supervision`, `pytorch_lightning`, and other heavy dependencies are imported lazily, on first use, rather than at module load, reducing cold-import time in inference-only environments. ([#801](https://github.com/roboflow/rf-detr/pull/801))

### Deprecated

- `rfdetr.deploy.*` — redirects to `rfdetr.export.*` with a `DeprecationWarning`. Migrate before v1.7.
- `rfdetr.util.*` — redirects to `rfdetr.utilities.*` with a `DeprecationWarning`. Migrate before v1.7.

### Fixed

- Fixed a cryptic `RuntimeError` / tensor-size mismatch when a checkpoint is incompatible with the current model architecture; a descriptive `ValueError` is raised instead, covering `segmentation_head` mismatch and `patch_size` mismatch. ([#810](https://github.com/roboflow/rf-detr/pull/810))
- Fixed `class_names` not reflecting dataset labels on `model.predict()` after training; class names are synced from the dataset so inference always uses the correct label list. ([#816](https://github.com/roboflow/rf-detr/pull/816))
- Fixed detection head reinitialization overwriting fine-tuned weights when loading a checkpoint with fewer classes than the model default. The second `reinitialize_detection_head` call fires only in the backbone-pretrain scenario. ([#815](https://github.com/roboflow/rf-detr/pull/815), [#509](https://github.com/roboflow/rf-detr/pull/509))
- Fixed `grid_sample` and bicubic interpolation silently falling back to CPU on MPS (Apple Silicon); both run natively on the MPS device. ([#821](https://github.com/roboflow/rf-detr/pull/821))
- Fixed `early_stopping=False` in `TrainConfig` being silently ignored; the setting propagates correctly. ([#835](https://github.com/roboflow/rf-detr/pull/835))
- Fixed an `AttributeError` crash in `update_drop_path` when the DINOv2 backbone layer structure does not match any known pattern.
- Added warning when `drop_path_rate > 0.0` is configured with a non-windowed DINOv2 backbone, where drop-path is silently ignored.
- Fixed `ValueError: matrix entries are not finite` in `HungarianMatcher` when the cost matrix contains NaN or Inf; non-finite entries are replaced with a finite sentinel before `linear_sum_assignment`, warning emitted at most once per matcher instance. ([#787](https://github.com/roboflow/rf-detr/pull/787))
- Fixed YOLO dataset validation rejecting `data.yml`; both `.yaml` and `.yml` are accepted. ([#777](https://github.com/roboflow/rf-detr/pull/777))
- Silently dropped degenerate bounding boxes, zero width or height, before Albumentations validation instead of raising `ValueError`. ([#825](https://github.com/roboflow/rf-detr/pull/825))

---

## [1.5.2] — 2026-03-04

### Added

- Added peak GPU memory (`max_mem` in MB) to training and evaluation progress bars on CUDA; omitted on CPU and MPS. ([#773](https://github.com/roboflow/rf-detr/pull/773))

### Fixed

- Fixed `aug_config` being silently ignored when training on YOLO-format datasets; `build_roboflow_from_yolo` never forwarded the value, so transforms always fell back to the default. ([#774](https://github.com/roboflow/rf-detr/pull/774))
- Fixed segmentation evaluation metrics not being written to `results_mask.json` during validation and test runs. ([#772](https://github.com/roboflow/rf-detr/pull/772))
- Fixed an `AttributeError` crash in `update_drop_path` when the DINOv2 backbone layer structure does not match any known pattern; `_get_backbone_encoder_layers` returns `None` for unrecognised architectures. ([#762](https://github.com/roboflow/rf-detr/pull/762))
- Fixed `drop_path_rate` not being forwarded to the DINOv2 model configuration, so stochastic depth was never applied even when explicitly set. Added a warning when `drop_path_rate > 0.0` is used with a non-windowed backbone. ([#762](https://github.com/roboflow/rf-detr/pull/762))
- Fixed incorrect COCO hierarchy filtering that excluded parent categories from the class list. ([#759](https://github.com/roboflow/rf-detr/pull/759))
- Fixed evaluation metric corruption on 1-indexed Roboflow datasets, caused by a flawed contiguity check in `_should_use_raw_category_ids`. ([#755](https://github.com/roboflow/rf-detr/pull/755))

## [1.5.1] — 2026-02-27

### Added

- Added support for nested Albumentations containers (`OneOf`, `Sequential`) inside `aug_config`. ([#752](https://github.com/roboflow/rf-detr/pull/752))

### Changed

- Migrated dataset transform pipeline to torchvision-native `Compose`, `ToImage`, and `ToDtype`; `Normalize` defaults to ImageNet mean/std. ([#745](https://github.com/roboflow/rf-detr/pull/745))

### Fixed

- Fixed `RFDETRMedium` missing from the public API; `__all__` contained a duplicate `RFDETRSmall` entry. ([#748](https://github.com/roboflow/rf-detr/pull/748))
- Fixed `AR50_90` reporting an incorrect value in `MetricsMLFlowSink`, due to a wrong COCO evaluation index. ([#735](https://github.com/roboflow/rf-detr/pull/735))
- Fixed supercategory filtering in `_load_classes` for COCO datasets with flat or mixed supercategory structures. ([#744](https://github.com/roboflow/rf-detr/pull/744))
- Fixed a crash in geometric transforms when a sample contained zero-area or empty masks. ([#727](https://github.com/roboflow/rf-detr/pull/727))
- Fixed segmentation training on Colab; `DepthwiseConvBlock` disables cuDNN for depthwise separable convolutions. ([#728](https://github.com/roboflow/rf-detr/pull/728))
- Pinned `onnxsim<0.6.0` to prevent `pip install` from hanging indefinitely. ([#749](https://github.com/roboflow/rf-detr/pull/749))

## [1.5.0] — 2026-02-23

### Added

- Added custom training augmentations via `aug_config` in `model.train()` — accepts a dict of Albumentations transforms, a built-in preset (`AUG_CONSERVATIVE`, `AUG_AGGRESSIVE`, `AUG_AERIAL`, `AUG_INDUSTRIAL`), or `{}` to disable. Bounding boxes and segmentation masks are transformed automatically. ([#263](https://github.com/roboflow/rf-detr/pull/263), [#702](https://github.com/roboflow/rf-detr/pull/702))
- Added `save_dataset_grids=True` in `TrainConfig` to write 3×3 JPEG grids of augmented samples to `output_dir` before training begins. ([#153](https://github.com/roboflow/rf-detr/pull/153))
- Added ClearML logger: set `clearml=True` in `TrainConfig` to stream per-epoch metrics to ClearML. ([#520](https://github.com/roboflow/rf-detr/pull/520))
- Added MLflow logger: set `mlflow=True` in `TrainConfig` to log runs and metrics to MLflow with custom tracking URI support. ([#109](https://github.com/roboflow/rf-detr/pull/109))
- Added live progress bar for training and validation with structured per-epoch logs. ([#204](https://github.com/roboflow/rf-detr/pull/204))
- Added `device` field to `TrainConfig` for explicit device selection. ([#687](https://github.com/roboflow/rf-detr/pull/687))
- `ModelConfig` raises an error on unknown parameters, preventing silent misconfiguration. ([#196](https://github.com/roboflow/rf-detr/pull/196))

### Changed

- Deprecated `OPEN_SOURCE_MODELS` constant in favour of `ModelWeights` enum. ([#696](https://github.com/roboflow/rf-detr/pull/696))
- Added MD5 checksum validation for pretrained weight downloads. ([#679](https://github.com/roboflow/rf-detr/pull/679))

### Fixed

- Fixed Albumentations bool-mask crash during segmentation training. ([#706](https://github.com/roboflow/rf-detr/pull/706))
- Fixed `UnboundLocalError` when resuming training from a completed checkpoint. ([#707](https://github.com/roboflow/rf-detr/pull/707))
- Prevented corruption of `checkpoint_best_total.pth` via atomic checkpoint stripping. ([#708](https://github.com/roboflow/rf-detr/pull/708))
- Fixed PyTorch 2.9+ compatibility issue with CUDA capability detection. ([#686](https://github.com/roboflow/rf-detr/pull/686))
- Fixed dtype mismatch error when `use_position_supervised_loss=True`. ([#447](https://github.com/roboflow/rf-detr/pull/447))
- Fixed inconsistent return values from `build_model`. ([#519](https://github.com/roboflow/rf-detr/pull/519))
- Fixed `positional_encoding_size` type annotation (`bool` → `int`). ([#524](https://github.com/roboflow/rf-detr/pull/524))
- Fixed ONNX export `output_names` to include masks when exporting segmentation models. ([#402](https://github.com/roboflow/rf-detr/pull/402))
- Fixed `num_select` not being updated correctly during segmentation model fine-tuning. ([#399](https://github.com/roboflow/rf-detr/pull/399))
- Fixed `np.argwhere` → `np.argmax` misuse. ([#536](https://github.com/roboflow/rf-detr/pull/536))
- Fixed COCO sparse category ID remapping for non-contiguous or offset category IDs. ([#712](https://github.com/roboflow/rf-detr/pull/712))
- Fixed segmentation mask filtering when using aggressive augmentations. ([#717](https://github.com/roboflow/rf-detr/pull/717))

---

## [1.4.3] — 2026-02-16

### Changed

- Pretrained weight downloads validate against an MD5 checksum to detect corrupted files. ([#679](https://github.com/roboflow/rf-detr/pull/679))

### Fixed

- Fixed `deploy_to_roboflow` failing for segmentation model exports. ([#578](https://github.com/roboflow/rf-detr/pull/578))
- Fixed missing `info` key in COCO export format. ([#681](https://github.com/roboflow/rf-detr/pull/681))

## [1.4.2] — 2026-02-12

### Added

- Added `generate_coco_dataset()` utility for generating synthetic COCO-format datasets with configurable class counts, split ratios, and bounding box annotations. ([#617](https://github.com/roboflow/rf-detr/pull/617))
- Added `run_test=False` to `TrainConfig` — skip test-split evaluation when your dataset has no test set. ([#628](https://github.com/roboflow/rf-detr/pull/628))

### Changed

- `model.predict()` accepts image URLs directly, with no need to download images before inference. ([#629](https://github.com/roboflow/rf-detr/pull/629))
- Plus models (`RFDETRXLarge`, `RFDETR2XLarge`) are distributed as a separate `rfdetr_plus` package under the Roboflow Model License. ([#645](https://github.com/roboflow/rf-detr/pull/645))

### Fixed

- Fixed segmentation ONNX export failure. ([#626](https://github.com/roboflow/rf-detr/pull/626))

## [1.4.1] — 2026-01-30

### Added

- Added native YOLO dataset format support alongside COCO. ([#74](https://github.com/roboflow/rf-detr/pull/74))
- Added `--print-freq` CLI argument to control training log frequency. ([#603](https://github.com/roboflow/rf-detr/pull/603))

### Changed

- Pinned `transformers` to `<5.0.0` to prevent incompatibility with the transformers v5 API. ([#599](https://github.com/roboflow/rf-detr/pull/599))

### Fixed

- Fixed class count mismatch in `train_from_config` for Roboflow-uploaded datasets. ([#588](https://github.com/roboflow/rf-detr/pull/588))
- Improved `num_classes` mismatch warning messages to be actionable rather than misleading. ([#261](https://github.com/roboflow/rf-detr/pull/261))
- Fixed CLI crash when specifying the `device` argument. ([#246](https://github.com/roboflow/rf-detr/pull/246))

## [1.4.0] — 2026-01-22

Headline release introducing new pre-trained model sizes — L, XL, and 2XL for object detection, and the full N/S/M/L/XL/2XL range for instance segmentation. Also added YOLO format training support, simplified the dependency footprint by removing several heavy packages (`cython`, `fairscale`, `timm`, `einops`, and others), and fixed per-class precision/recall/F1 computation. Drops Python 3.9 support.
