# RF-DETR - Agent Instructions

This file provides detailed context for AI coding agents working with RF-DETR. For human-readable documentation, see [README.md](README.md). For contribution guidelines, see [CONTRIBUTING.md](.github/CONTRIBUTING.md).

## Build & Development Environment

### Prerequisites

- **Python:** >=3.10 (tested on 3.10, 3.11, 3.12, 3.13)
- **Package Manager:** `uv` (install via `pip install uv`)
- **GPU:** Optional (CPU testing supported, GPU required for training)

### Development Setup

```bash
# Full development environment
uv sync --all-groups

# Specific dependency groups
uv sync --group tests      # Testing only
uv sync --group docs       # Documentation only
uv sync --group build      # Build tools only
```

**Post-pull workflow:** Always run `uv sync` after pulling changes to ensure dependencies are synchronized.

### Dependency Groups

See `pyproject.toml` for complete dependency specifications:

- **Core:** PyTorch, torchvision, transformers, pycocotools, supervision, peft, pydantic
- **Optional:** `[plus]` (Plus models), `[onnxexport]` (ONNX export), `[metrics]` (tensorboard, wandb)
- **Development:** `tests`, `docs`, `build` groups

**Important version constraints:**
- PyTorch: >=1.13.0, <=2.8.0 (2.9.0+ excluded due to known issues)
- Transformers: >4.0.0, <5.0.0

## Testing

### Running Tests

```bash
# CPU tests (default for local development)
uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu" --cov=rfdetr --cov-report=xml

# GPU tests (requires GPU)
uv run --no-sync pytest src/ tests/ -n 2 -m gpu

# All tests
uv run --no-sync pytest src/ tests/ -n 2
```

### Test Organization & Conventions

**Structure:**
- Group related tests in classes
- Use `@pytest.mark.parametrize` with `pytest.param(..., id="name")` for parameterized tests
- Mark GPU-dependent tests with `@pytest.mark.gpu`

**Example:**

```python
import pytest

@pytest.mark.gpu  # Marks test as GPU-dependent
@pytest.mark.parametrize("model_variant", [
    pytest.param("nano", id="nano"),
    pytest.param("small", id="small"),
])
class TestModelInference:
    def test_forward_pass(self, model_variant):
        # Test implementation
        pass
```

**Development Workflow:**

1. **Bug fixes:** Write test that replicates the issue first, then implement fix
2. **New features:** Write comprehensive tests covering all major use cases
3. **Before commit:** Run full test suite to ensure no regressions

**CI Testing:**
- Runs on Ubuntu, Windows, macOS
- Python versions: 3.10, 3.11, 3.12, 3.13
- CPU tests: `pytest -m "not gpu"`
- GPU tests: `pytest -m gpu` (separate workflow)

## Code Quality & Linting

### Pre-commit Hooks

Configuration: `.pre-commit-config.yaml`

```bash
# Setup
pre-commit install

# Run manually
pre-commit run --all-files
```

**Hooks include:**
- **ruff:** Python linting and formatting (config in `pyproject.toml`)
- **mdformat:** Markdown formatting
- **prettier:** YAML/TOML formatting
- **codespell:** Spell checking
- **license headers:** Apache 2.0 header enforcement

**License Header (required for all Python files):**

```python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
```

### Manual Linting

```bash
# Run ruff
ruff check --fix .

# Format code
ruff format .
```

## Documentation

### Building Docs

```bash
# Install docs dependencies
uv sync --group docs

# Serve locally (live reload)
mkdocs serve

# Build static site
mkdocs build
```

**Documentation Structure:**
- **Source:** `docs/` directory (Markdown)
- **Config:** `mkdocs.yaml` (uses custom YAML tags: `!!python/name`)
- **Deployment:** GitHub Actions publishes to GitHub Pages

**Note:** `mkdocs.yaml` is excluded from `check-yaml` pre-commit hook due to custom YAML tags.

## Package Building

```bash
# Install build dependencies
uv sync --group build

# Build distributions
uv build

# Validate build
uv run twine check --strict dist/*
```

**Build outputs:**
- Source distribution: `dist/rfdetr-*.tar.gz`
- Wheel: `dist/rfdetr-*.whl`

## Architecture & Conventions

### Code Organization

```
src/rfdetr/
├── main.py              # Core training/eval logic, CLI entry
├── detr.py              # Model wrappers (RFDETR classes)
├── config.py            # Configuration dataclasses
├── engine.py            # Training engine functions
├── cli/                 # Command-line interface
├── datasets/            # Dataset implementations (COCO, custom)
├── deploy/              # Export utilities (ONNX, TensorRT)
├── models/
│   ├── backbone/        # DINOv2 backbone implementations
│   ├── lwdetr.py        # LW-DETR transformer
│   └── segmentation_head.py  # Instance segmentation head
├── platform/
│   └── models.py        # Plus model integration (XLarge, 2XLarge)
└── util/
    ├── logger.py        # Logger configuration
    ├── misc.py          # Distributed training utilities
    └── coco_classes.py  # COCO dataset classes
```

### Key Patterns

**Model Architecture:**
- RFDETR wrappers: `self.model` is `rfdetr.main.Model` instance
- Underlying PyTorch module: `self.model.model`
- Segmentation models return `pred_masks` as `torch.Tensor` or dict with keys `['spatial_features', 'query_features', 'bias']`

**Imports:**
```python
# Distributed training utilities
import rfdetr.util.misc as utils
utils.get_rank()
utils.get_world_size()
utils.is_main_process()
utils.save_on_master()

# Logger
from rfdetr.util.logger import get_logger
logger = get_logger()  # Default name: "rf-detr", reads LOG_LEVEL env var

# TQDM (environment compatibility)
from tqdm.auto import tqdm  # NOT: from tqdm import tqdm
```

**Plus Models (XLarge, 2XLarge):**
- Requires separate `rfdetr_plus` package (PML 1.0 license)
- Import handled lazily via `__getattr__` in `src/rfdetr/platform/models.py`
- Raises `ImportError` if package not installed

**Subprocess Usage:**
```python
import subprocess

result = subprocess.run(
    ["command", "arg1", "arg2"],
    check=True,        # Raise CalledProcessError on failure
    text=True,         # Return stdout/stderr as strings
    capture_output=True
)
# Note: stderr is already a string, don't decode
```

**Logging:**
- Use `logger.debug()` for detailed tensor/shape information (not `logger.info()`)
- Use `logger.info()` for high-level progress/status

**Checkpoint Handling:**
- Always check file existence before operations
- Prevents errors when training is interrupted

### Type Hints & Docstrings

**Mandatory Requirements:**
- Type hints for all function parameters and return types
- Google-style docstrings for all functions and classes

**Example:**

```python
from typing import Optional

def train_model(
    config: TrainConfig,
    checkpoint_path: Optional[str] = None
) -> dict[str, float]:
    """
    Train RF-DETR model with given configuration.

    Args:
        config (TrainConfig): Training configuration with hyperparameters.
        checkpoint_path (Optional[str]): Path to resume from checkpoint.

    Returns:
        dict[str, float]: Training metrics (loss, mAP, etc.).

    Examples:
        >>> config = TrainConfig(epochs=10, batch_size=16)
        >>> metrics = train_model(config)
        >>> print(metrics["mAP"])
        0.452
    """
    # Implementation
    pass
```

**Type Hint Compatibility:**
- Use `Optional[type]` or `from __future__ import annotations` for compatibility
- Target Python version: 3.10

## Common Workflows

### Making Changes

1. **Setup:** `uv sync --all-groups`
2. **Before changes:** Run tests to establish baseline
3. **Development:**
   - Make minimal, focused changes
   - Follow existing patterns and conventions
   - Add type hints and docstrings
4. **Testing:**
   - Bug fixes: Write test first, then fix
   - Features: Test all major use cases
   - Run: `uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu"`
5. **Quality checks:** `pre-commit run --all-files`
6. **Build (if needed):** `uv build`
7. **Commit:** Pre-commit hooks run automatically

### Adding New Model Variants

See `.github/CONTRIBUTING.md` for detailed guidance on adding new model architectures.

### Security Considerations

- **Write secure code:** Avoid injection vulnerabilities (XSS, SQL injection, command injection)
- **Validate inputs:** Especially for file paths, URLs, and user-provided data
- **No credentials:** Never commit API keys, tokens, or credentials
- **Follow OWASP best practices**

## CI/CD Workflows

GitHub Actions workflows in `.github/workflows/`:

- **ci-tests-cpu.yml:** CPU tests across OS/Python versions
- **ci-tests-gpu.yml:** GPU-dependent tests
- **build-package.yml:** Build and validate distributions
- **ci-build-docs.yml:** Documentation builds
- **publish-docs.yml:** Deploy docs to GitHub Pages

**Concurrency:** PRs cancel in-progress runs on new pushes

## Additional Resources

- **Documentation:** https://rfdetr.roboflow.com
- **Repository:** https://github.com/roboflow/rf-detr
- **Issues:** https://github.com/roboflow/rf-detr/issues
- **Discord:** https://discord.gg/GbfgXGJ8Bk
- **Contributing:** `.github/CONTRIBUTING.md`
- **Copilot Instructions:** `.github/copilot-instructions.md`

---

**Note:** This file is designed for AI coding agents. For human-readable project information, see README.md. For contribution guidelines, see CONTRIBUTING.md.
