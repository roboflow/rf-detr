# RF-DETR — Agent Instructions

AI-agent-specific technical context for RF-DETR. Canonical contribution rules, workflows, and human-onboarding docs live in [CONTRIBUTING.md](.github/CONTRIBUTING.md) — this file supplements, never duplicates them.

> [!NOTE]
> **Canonical authority**: when conventions appear in both this file and `.github/copilot-instructions.md`, AGENTS.md is authoritative. Propagate any updates to both files together.

**Canonical Sources:**

- **Contribution Guidelines:** [CONTRIBUTING.md](.github/CONTRIBUTING.md) — authoritative for all human workflows
- **Human Documentation:** [README.md](README.md) — project overview and usage examples
- **Copilot Instructions:** [.github/copilot-instructions.md](.github/copilot-instructions.md) — GitHub Copilot-specific guidance

## Agent Responsibilities

As an AI agent contributing to RF-DETR, you are responsible for:

1. **Following test-driven development practices** — write failing tests first for bug fixes; comprehensive tests for new features; all tests passing before opening PR
2. **Adhering to code quality standards** — run `pre-commit run --all-files` before every commit; follow type hint and docstring requirements
3. **Maintaining agentic documentation** — update `AGENTS.md` when architecture patterns or technical conventions change; update `.github/copilot-instructions.md` when high-level guidance changes; update `.github/CONTRIBUTING.md` when human workflow is affected; apply updates after major PR review feedback
4. **Consulting maintainers before major changes** — open an issue before adding new models or significant features; wait for approval on approach before implementing
5. **Writing secure, minimal code** — avoid over-engineering; prevent injection vulnerabilities; follow existing patterns

## Quick-reference Pointers

All details live in CONTRIBUTING.md. Direct links:

- **Setup:** [Development Environment Setup](.github/CONTRIBUTING.md#development-environment-setup) — `uv sync --all-groups` to start
- **Testing:** [Test-Driven Development](.github/CONTRIBUTING.md#test-driven-development) — TDD rules, parametrize, `@pytest.mark.gpu`
- **Code quality:** [Code Quality and Linting](.github/CONTRIBUTING.md#code-quality-and-linting) — always `pre-commit run --all-files`
- **Type hints & docstrings:** [Google-Style Docstrings](.github/CONTRIBUTING.md#google-style-docstrings-and-mandatory-type-hints) — MANDATORY on all functions/classes
- **Deprecation rules:** [Deprecation Policy](.github/CONTRIBUTING.md#deprecation-policy) — `@deprecated` / `@deprecated_class`, version windows, migration-doc requirement
- **Adding a model:** [Adding a New Model](.github/CONTRIBUTING.md#adding-a-new-model) — always discuss with maintainers first
- **Security:** [Security Considerations](.github/CONTRIBUTING.md#security-considerations) — OWASP basics, no credentials
- **CI/CD workflows:** [CI Testing](.github/CONTRIBUTING.md#ci-testing) — 5 workflow files, OS/Python matrix

> [!IMPORTANT]
> **Agentic test command:** use `-n 1` (not `-n 2`) for reproducible failure output in agentic sessions. CI uses `-n 2`. Full command:
>
> ```bash
> uv run --no-sync pytest src/ tests/ -n 1 -m "not gpu" --ignore=tests/try_instantiate_all_models.py --cov=rfdetr --cov-report=xml --timeout=420 --durations=50
> ```

## Module Map (`src/rfdetr/`)

Internal organization is subject to change as this is an active research project. Current map:

```schema
src/rfdetr/
├── detr.py        # RFDETR base orchestrator — get_model(), train(), predict(), export()
├── inference.py   # ModelContext wrapper returned by get_model()
├── variants.py    # Concrete variant classes (see §Model Variants)
├── config.py      # Pydantic v2 ModelConfig / TrainConfig hierarchies (see §Config System)
├── utilities/     # Canonical utility home since v1.6.0
│   ├── distributed.py  # get_rank, get_world_size, is_main_process, save_on_master
│   ├── logger.py       # get_logger()
│   └── decorators.py   # Re-exports @deprecated / @deprecated_class from deprecate
├── training/      # module_model.py, module_data.py, trainer.py, callbacks/
├── datasets/      # COCO, YOLO, Object365, synthetic, augmentation configs, transforms
├── models/        # Backbone + LWDETR transformer + heads, criterion, matcher, postprocess
├── evaluation/    # COCO mAP, F1 sweep, Hungarian matching
├── export/        # ONNX, TensorRT, TFLite exporters + benchmark
├── platform/      # Plus-models loader hook
│   ├── __init__.py     # _IS_RFDETR_PLUS_AVAILABLE flag
│   └── models.py       # Imports from rfdetr_plus if available; raises ImportError otherwise
├── assets/        # ModelWeights enum, COCO class list
├── cli/           # Command-line entry points
├── visualize/     # Visualization utilities
├── util/          # [DEPRECATED -> utilities/] removed v1.9.0
└── deploy/        # [DEPRECATED -> export/]    removed v1.9.0
```

## Key Code Patterns

### Model Architecture

- `RFDETR` (in `detr.py`) is the base orchestrator; variant classes (in `variants.py`) subclass it — see §Model Variants
- `RFDETR` instances hold a `ModelContext` at `self.model` (defined in `rfdetr/inference.py`). Key attrs: `.model` (underlying `nn.Module` — an LWDETR instance), `.postprocess`, `.device`, `.resolution`, `.args`, `.class_names`, `.inference_model`; reach the trained module via `self.model.model`
- Segmentation models return `pred_masks` as `torch.Tensor` or dict with keys `['spatial_features', 'query_features', 'bias']`

### Imports

```python
# Canonical utility paths (since v1.6.0)
from rfdetr.utilities.distributed import get_rank, get_world_size, is_main_process, save_on_master
from rfdetr.utilities.logger import get_logger

# WARNING: rfdetr.util.* and rfdetr.deploy.* are Phase-1 deprecated shims removed in v1.9.0.
# Never write new code against them — see src/rfdetr/__init__.py docstring for migration details.

logger = get_logger()  # Default name: "rf-detr", reads LOG_LEVEL env var
from tqdm.auto import tqdm  # NOT: from tqdm import tqdm
```

### Plus Models Loader Hook

- `RFDETRXLarge` / `RFDETR2XLarge` (detection) require separate `rfdetr_plus` package (PML 1.0 license); `RFDETRSegXLarge` / `RFDETRSeg2XLarge` (segmentation) are in the main package
- Import chain: `rfdetr.__init__.__getattr__` intercepts Plus class names → delegates to `rfdetr.platform.models` → imports from `rfdetr_plus` if available, else raises `ImportError` with install hint
- `_IS_RFDETR_PLUS_AVAILABLE` flag set in `src/rfdetr/platform/__init__.py`

### Subprocess Usage

```python
import subprocess

result = subprocess.run(
    ["command", "arg1", "arg2"],
    check=True,  # Raise CalledProcessError on failure
    text=True,  # Return stdout/stderr as strings
    capture_output=True,
)
# Note: stderr is already a string, don't decode
```

### Logging Conventions

- `logger.debug()` — detailed tensor/shape information
- `logger.info()` — high-level progress/status

### Checkpoint Handling

Always check file existence before operating on checkpoints — training can be interrupted mid-write.

## Model Variants

Active detection: `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge`
Active segmentation: `RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, `RFDETRSegLarge`
Plus detection (require `rfdetr_plus`): `RFDETRXLarge`, `RFDETR2XLarge`
Plus segmentation (main package): `RFDETRSegXLarge`, `RFDETRSeg2XLarge`

**Deprecated — do not use in new code:**

| Class                   | Deprecated in | Removed in |
| ----------------------- | ------------- | ---------- |
| `RFDETRBase`            | v1.6.0        | v2.0.0     |
| `RFDETRLargeDeprecated` | v1.6.0        | v2.0.0     |
| `RFDETRSegPreview`      | v1.6.0        | v2.0.0     |

**Default size in examples:** `RFDETRSmall` / `'rfdetr-small'` — never use `RFDETRBase`.

## Config System

`src/rfdetr/config.py` defines Pydantic v2 `ModelConfig` and `TrainConfig` hierarchies:

- `BaseConfig` uses `extra="forbid"` + `validate_assignment=True` — unknown kwargs raise a descriptive `ValueError` listing the typo and allowed params (via `catch_typo_kwargs`)
- Each variant pins `_model_config_class`; segmentation variants also pin `_train_config_class = SegmentationTrainConfig`
- `_detect_device()` determines accelerator without creating a CUDA context (important for fork-based DDP)
- Check `config.py` for allowed fields before adding new parameters to model constructors

## Deprecation Decorators

Use `@deprecated` (functions/methods) and `@deprecated_class` (classes) from `deprecate` — never `warnings.warn`. Full rules and removal checklist: [CONTRIBUTING.md §Deprecation Policy](.github/CONTRIBUTING.md#deprecation-policy).

## Inference Example

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall()  # downloads pretrained weights on first call
detections = model.predict("image.jpg", threshold=0.5)
# returns supervision.Detections with .xyxy boxes, .confidence, .class_id
```

For batch inference or custom resolution: `model.predict(["img1.jpg", "img2.jpg"], shape=(640, 640))`.

---

Human-readable project information: [README.md](README.md). Contribution rules: [CONTRIBUTING.md](.github/CONTRIBUTING.md).
