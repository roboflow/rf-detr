# RF-DETR - Agent Instructions

This file provides detailed technical context for AI coding agents working with RF-DETR.

**Canonical Sources:**
- **Contribution Guidelines:** [CONTRIBUTING.md](.github/CONTRIBUTING.md) - The authoritative source for all contribution practices
- **Human Documentation:** [README.md](README.md) - Project overview and usage
- **Copilot Instructions:** [.github/copilot-instructions.md](.github/copilot-instructions.md) - GitHub Copilot-specific guidance

This document supplements the contribution guidelines with detailed technical information for automated tooling.

## Agent Responsibilities

**Maintaining Agentic Documentation:**

When contributing as an AI agent, if your contribution:
- **Changes project structure or architecture patterns**
- **Introduces new conventions or patterns**
- **Receives major feedback from maintainers in PR review** about structure, patterns, or conventions

**You should update the relevant agentic documents:**
- Update `AGENTS.md` for detailed technical patterns and architecture changes
- Update `.github/copilot-instructions.md` for high-level guidance changes
- Update `.github/CONTRIBUTING.md` if the change affects human contribution workflow

**Rationale:** These documents guide future agent contributions. Keeping them current ensures consistency and reduces repeated feedback on the same issues.

## Build & Development Environment

> **Canonical Reference:** See [Development Environment Setup](.github/CONTRIBUTING.md#development-environment-setup) in CONTRIBUTING.md for complete setup instructions.

### Quick Setup

```bash
# Install uv
pip install uv

# Full development environment
uv sync --all-groups

# Specific dependency groups
uv sync --group tests      # Testing only
uv sync --group docs       # Documentation only
uv sync --group build      # Build tools only
```

**Prerequisites:** Python >=3.10 (tested on 3.10-3.13)

### Dependency Information

See `pyproject.toml` for complete dependency specifications:

- **Core:** PyTorch, torchvision, transformers, pycocotools, supervision, peft, pydantic
- **Optional:** `[plus]` (Plus models), `[onnxexport]` (ONNX export), `[metrics]` (tensorboard, wandb)
- **Development:** `tests`, `docs`, `build` groups

**Important version constraints:**
- PyTorch: >=1.13.0, <=2.8.0 (2.9.0+ excluded due to known issues)
- Transformers: >4.0.0, <5.0.0

## Testing

> **Canonical Reference:** See [Test-Driven Development](.github/CONTRIBUTING.md#test-driven-development) in CONTRIBUTING.md for complete guidelines.
>
> **CI Workflows (Source of Truth):** See `.github/workflows/ci-tests-cpu.yml` and `.github/workflows/ci-tests-gpu.yml` for exact test commands used in CI.

### Quick Commands

```bash
# CPU tests (default for local development) - matches CI
uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu" --cov=rfdetr --cov-report=xml

# GPU tests (requires GPU)
uv run --no-sync pytest src/ tests/ -n 2 -m gpu

# Specific test file or test
uv run --no-sync pytest tests/test_model.py::test_specific_function

# Pre-commit checks (ALWAYS run before committing)
pre-commit run --all-files
```

### Key Testing Principles

**Test-Driven Development:**
1. **Bug fixes:** Write failing test → (optional: commit with "WIP") → Fix code → Verify all tests pass → Commit fix
2. **New features:** Write comprehensive tests → Implement feature → Refactor

**Testing Requirements:**
- ⚠️ **During development:** Tests may fail as you work through TDD cycle
- ✅ **Before opening PR:** Final commit MUST have all tests passing
- ✅ **Before each commit:** Run `pre-commit run --all-files`

**Test Organization:**
- Group related tests in classes
- Use `@pytest.mark.parametrize` with `pytest.param(..., id="name")`
- Mark GPU/heavy tests with `@pytest.mark.gpu`

**CI Testing:**
- Runs on Ubuntu, Windows, macOS with Python 3.10-3.13
- CPU workflow: `pytest -m "not gpu"`
- GPU workflow: `pytest -m gpu`

## Code Quality & Linting

> **Canonical Reference:** See [Code Quality and Linting](.github/CONTRIBUTING.md#code-quality-and-linting) in CONTRIBUTING.md for setup and details.

### Quick Commands

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run ruff only
ruff check --fix .
ruff format .
```

**Configuration Files:**
- `.pre-commit-config.yaml` - Pre-commit hooks (ruff, mdformat, prettier, codespell, license headers)
- `pyproject.toml` - Ruff linting rules (`[tool.ruff]` section)

**License Header (required for all Python files):**

```python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
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

> **Canonical Reference:** See [Google-Style Docstrings and Mandatory Type Hints](.github/CONTRIBUTING.md#google-style-docstrings-and-mandatory-type-hints) in CONTRIBUTING.md for complete requirements and examples.

**Requirements:**
- MANDATORY type hints for all function parameters and return types
- MANDATORY Google-style docstrings for all functions and classes
- Target Python version: 3.10+

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

> **Canonical Reference:** See [Adding a New Model](.github/CONTRIBUTING.md#adding-a-new-model) in CONTRIBUTING.md for detailed guidance.

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
