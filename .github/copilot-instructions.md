# RF-DETR Copilot Instructions

## Repository Overview

RF-DETR is a real-time transformer architecture for object detection and instance segmentation developed by Roboflow. It's built on a DINOv2 vision transformer backbone and delivers state-of-the-art accuracy and latency trade-offs on Microsoft COCO and RF100-VL datasets.

**Key Characteristics:**

- **Type:** Python library for computer vision (PyTorch-based)
- **Size:** ~1.6MB (excluding dependencies)
- **Primary Language:** Python (requires >=3.10)
- **Supported Python Versions:** 3.10, 3.11, 3.12, 3.13
- **Main Framework:** PyTorch with torchvision
- **Key Dependencies:** transformers, pycocotools, supervision, peft, tqdm
- **Target Runtimes:** POSIX, Unix, MacOS (Windows supported for testing)
- **License:** Apache 2.0 (with Plus models under PML 1.0)

## Build and Validation

### Environment Setup

**Required:** Python >=3.10. The project uses `uv` (the Python package installer) for dependency management.

**Install uv (if not already installed):**

```bash
pip install uv
```

### Bootstrap and Installation

**From source (development):**

```bash
# Clone and install dependencies
uv sync --group tests

# For documentation development
uv sync --group docs

# For all development dependencies (tests + docs + build tools)
uv sync --all-groups
```

**IMPORTANT:** Always run `uv sync` after pulling changes to ensure dependencies are up to date.

### Building the Package

```bash
# Install build dependencies
uv pip install -r pyproject.toml --group build

# Build source and wheel distributions
uv build

# Validate the build
uv run twine check --strict dist/*
```

**Expected time:** ~30-60 seconds

### Testing

**Run CPU tests (default):**

```bash
# Run tests excluding GPU-dependent tests
uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu" --cov=rfdetr --cov-report=xml
```

**Run GPU tests:**

```bash
uv run --no-sync pytest src/ tests/ -n 2 -m gpu
```

**Expected time:** ~1-2 minutes for CPU tests

**Test markers:**

- `gpu`: Tests that require GPU or are slow on CPU
- Use `-n 2` for parallel execution with pytest-xdist

**IMPORTANT:** Always run tests before committing changes. Tests run on Python 3.10, 3.11, 3.12, and 3.13 across Ubuntu, Windows, and macOS in CI.

### Linting and Code Quality

**Pre-commit hooks:**

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

**Key linters and formatters:**

- **ruff**: Python linting (select: E, W, F, I; see pyproject.toml for ignored rules)
- **mdformat**: Markdown formatting
- **prettier**: YAML/TOML formatting
- **codespell**: Spell checking
- **license header check**: All Python files must start with Apache 2.0 license header

**Run ruff manually:**

```bash
ruff check --fix .
```

**IMPORTANT:** The pre-commit hooks will auto-format many issues. If pre-commit fails, review the changes and re-run.

### Documentation

**Build documentation locally:**

```bash
# Install docs dependencies
uv sync --group docs

# Serve docs locally (with live reload)
mkdocs serve

# Build static docs
mkdocs build
```

**Expected time:** ~5-10 seconds to build

**Documentation structure:**

- `docs/`: Documentation source (Markdown)
- `mkdocs.yaml`: MkDocs configuration (uses custom YAML tags, excluded from check-yaml hook)
- Documentation is built with mkdocs-material and published via GitHub Actions

## Project Layout and Architecture

### Directory Structure

```
.
├── .github/              # GitHub workflows, issue templates, contributing guidelines
│   ├── workflows/        # CI/CD pipelines (tests, builds, releases)
│   ├── CONTRIBUTING.md   # Contribution guidelines (CLA, docstrings, type hints)
│   └── ISSUE_TEMPLATE/   # Issue templates
├── docs/                 # Documentation source (MkDocs)
├── src/rfdetr/          # Main package source
│   ├── __init__.py      # Package entry point, exports main classes
│   ├── config.py        # Configuration classes (TrainConfig, etc.)
│   ├── main.py          # Core training/evaluation logic (52KB, main entry point)
│   ├── detr.py          # DETR model wrappers and interfaces (23KB)
│   ├── engine.py        # Training engine functions
│   ├── cli/             # Command-line interface
│   ├── datasets/        # Dataset implementations
│   ├── deploy/          # Export and deployment utilities (ONNX, TensorRT)
│   ├── models/          # Model architectures (backbone, transformer, heads)
│   ├── platform/        # Platform integration (Roboflow API)
│   └── util/            # Utilities (logger, distributed training, COCO classes)
├── tests/               # Test suite (9 test files)
├── pyproject.toml       # Project configuration (dependencies, tools)
├── .pre-commit-config.yaml  # Pre-commit hook configuration
├── mkdocs.yaml          # Documentation configuration
└── README.md            # Main documentation
```

### Key Source Files

**Main Entry Points:**

- `src/rfdetr/main.py`: Core training and evaluation logic, CLI entry point
- `src/rfdetr/detr.py`: Model wrappers (RFDETR wrappers set `self.model` to a `rfdetr.main.Model` instance; the underlying torch module is `self.model.model`)
- `src/rfdetr/__init__.py`: Exports main classes (RFDETRNano, RFDETRSmall, etc.)

**Configuration:**

- `src/rfdetr/config.py`: Configuration dataclasses (TrainConfig with run_test: bool = True)

**Models:**

- `src/rfdetr/models/`: Model architectures
    - `backbone/`: DINOv2 backbone implementations
    - `lwdetr.py`: LW-DETR transformer architecture
    - `segmentation_head.py`: Instance segmentation head

**Platform Integration:**

- `src/rfdetr/platform/models.py`: Plus-only models (RFDETRXLarge/RFDETR2XLarge) are imported from rfdetr_plus; missing package is handled by catching ModuleNotFoundError for rfdetr_plus and raising ImportError on access via `__getattr__`

**Utilities:**

- `src/rfdetr/util/misc.py`: Distributed training helpers (`import rfdetr.util.misc as utils` and call functions with `utils.` prefix: get_rank(), get_world_size(), is_main_process(), save_on_master()). The get_sha() function returns a formatted string or "unknown", not a dict.
- `src/rfdetr/util/logger.py`: Logger configuration (`rfdetr.util.logger.get_logger()` with default name "rf-detr", reads LOG_LEVEL env var)

### Configuration Files

- `pyproject.toml`: Main project configuration (dependencies, build system, tool settings)
- `.pre-commit-config.yaml`: Pre-commit hooks configuration
- `mkdocs.yaml`: Documentation configuration (uses custom YAML tags: !!python/name, excluded from check-yaml hook)
- `.codecov.yml`: Code coverage configuration

## CI/CD Workflows

### GitHub Actions Workflows

**Main CI Workflows:**

1. **ci-tests-cpu.yml**: Runs CPU tests on Ubuntu, Windows, macOS with Python 3.10-3.13

    - Command: `uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu" --cov=rfdetr --cov-report=xml`
    - Timeout: 10 minutes
    - Uploads coverage to Codecov

2. **ci-tests-gpu.yml**: Runs GPU tests

    - Command: `pytest -m gpu`
    - Timeout: 30 minutes

3. **build-package.yml**: Builds source and wheel distributions

    - Commands: `uv pip install -r pyproject.toml --group build`, `uv build`, `uv run twine check --strict dist/*`

4. **ci-build-docs.yml**: Builds documentation

    - Command: `mkdocs build`

5. **publish-docs.yml**: Publishes documentation to GitHub Pages

**On Push/PR:**

- Tests run on both `main` and `develop` branches
- Concurrency: `group: pytest-test-${{ github.ref }}`, cancels in-progress for PRs

### Pre-commit Hooks

Automatically run on commit (when installed with `pre-commit install`):

- trailing-whitespace, end-of-file-fixer, mixed-line-ending
- check-yaml (excludes mkdocs.yaml), check-toml, check-case-conflict
- check-executables-have-shebangs, detect-private-key
- ruff-check with --fix
- mdformat with mkdocs support and ruff integration
- codespell for spell checking
- insert-license for Python files (Apache 2.0 header)

**All Python files must start with:**

```python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
```

## Coding Standards and Conventions

### Python Style

**Type Hints:**

- **MANDATORY** for all function definitions (parameters and return types)
- Use `Optional[type]` instead of `type | None` for compatibility (or add `from __future__ import annotations`)
- Project requires Python >=3.10 (target-version py310)

**Docstrings:**

- **MANDATORY** Google-style docstrings for all new functions and classes
- Must include: brief description, Args, Returns, Examples (when applicable)

**Example:**

```python
def sample_function(param1: int, param2: int = 10) -> bool:
    """
    Provides a brief description of function behavior.

    Args:
        param1 (int): Explanation of the first parameter.
        param2 (int): Explanation of the second parameter, defaulting to 10.

    Returns:
        bool: True if the operation succeeds, otherwise False.

    Examples:
        >>> sample_function(5, 10)
        True
    """
    return param1 == param2
```

### Import Conventions

**Standard patterns:**

- `from tqdm.auto import tqdm` (not `from tqdm import tqdm`) for broader environment compatibility
- `import rfdetr.util.misc as utils` and call functions as `utils.get_rank()`, `utils.is_main_process()`, etc.

**Subprocess usage:**

- When using `subprocess.run` with `text=True`, stderr is already a string and should not be decoded
- Use `check=True` to raise CalledProcessError on failure

### Testing Conventions

**Parameterized tests:**

- Use `pytest.mark.parametrize` with `pytest.param(..., id="name")` for parameterized benchmark tests across model variants

**Test markers:**

- `@pytest.mark.gpu`: For tests that require GPU or are slow on CPU
- CI splits pytest runs: CPU workflow runs `pytest -m "not gpu"`, GPU workflow runs `pytest -m gpu`

**Example:**

```python
@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param("nano", id="nano"),
        pytest.param("small", id="small"),
    ],
)
def test_model_inference(model_name):
    # Test code
    pass
```

### Model Architecture Conventions

**Segmentation models:**

- Return pred_masks as either a torch.Tensor or dict with keys 'spatial_features', 'query_features', 'bias'

**Logging:**

- Use `logger.debug()` for detailed shape/tensor information during export and inference (not `logger.info()`)

### File Operations

**Checkpoint handling:**

- Always check if checkpoint files exist before attempting file operations to prevent errors when training is interrupted or weights are not saved

## Dependencies and Requirements

### Core Dependencies

**Required (from pyproject.toml):**

- torch>=1.13.0,\<=2.8.0 (Note: Torch >=2.9.0 is excluded due to known issues)
- torchvision>=0.14.0
- transformers>4.0.0, \<5.0.0
- pycocotools
- scipy
- tqdm
- peft
- pydantic
- supervision
- matplotlib
- roboflow
- requests
- rf100vl

**Optional Dependencies:**

- `[plus]`: rfdetr_plus>=1.0.0 (for XLarge and 2XLarge models, PML 1.0 license)
- `[onnxexport]`: onnx, onnxsim, onnx_graphsurgeon, onnxruntime, polygraphy
- `[metrics]`: tensorboard, wandb

**Development Dependencies:**

- `tests`: pytest, pytest-cov, pytest-xdist
- `docs`: mkdocs-material, mkdocstrings, mkdocstrings-python, mike, mkdocs-jupyter
- `build`: twine, wheel, build

## Common Workflows

### Making Code Changes

1. **Before making changes:**

    - Run existing tests: `uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu"`
    - Understand any existing failures (unrelated issues are not your responsibility)

2. **During development:**

    - Make minimal, surgical changes
    - Follow coding standards (type hints, docstrings, license headers)
    - Use existing libraries and patterns

3. **After making changes:**

    - Run tests: `uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu"`
    - Run linters: `pre-commit run --all-files` or `ruff check --fix .`
    - Build if needed: `uv build`

4. **Before committing:**

    - Ensure all tests pass
    - Ensure pre-commit hooks pass
    - Review changes for minimal scope

### Adding New Models

See `.github/CONTRIBUTING.md` section "Adding a New Model" for detailed guidance.

### Handling Plus Models

- Plus models (XLarge, 2XLarge) require `rfdetr_plus` package
- Import handling: catch `ModuleNotFoundError` for rfdetr_plus and raise `ImportError` on access via `__getattr__`
- Example in `src/rfdetr/platform/models.py`

## Key Facts to Remember

1. **Always use `uv` for package management**, not plain pip
2. **Test with `uv run --no-sync pytest`** to avoid re-syncing dependencies
3. **Use `-n 2` for parallel test execution** (pytest-xdist)
4. **Mark GPU tests with `@pytest.mark.gpu`** to exclude from CPU CI
5. **All Python files need Apache 2.0 license header** (enforced by pre-commit)
6. **Type hints and Google-style docstrings are mandatory** for new code
7. **Import tqdm from tqdm.auto**, not tqdm directly
8. **Use `import rfdetr.util.misc as utils`** for distributed helpers
9. **RFDETR wrappers set `self.model` to rfdetr.main.Model**; underlying torch module is `self.model.model`
10. **mkdocs.yaml uses custom YAML tags** (!!python/name) and is excluded from check-yaml hook
11. **Check checkpoint files exist** before file operations
12. **Use `logger.debug()` for detailed tensor/shape info**, not logger.info()

## Validation Steps

Before submitting changes:

1. ✅ Run tests: `uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu"`
2. ✅ Run linters: `pre-commit run --all-files`
3. ✅ Build package (if source changes): `uv build`
4. ✅ Build docs (if doc changes): `mkdocs build`
5. ✅ Review changes for minimal scope and correctness
6. ✅ Ensure all new functions have type hints and docstrings
7. ✅ Ensure license headers are present in all Python files

## Additional Resources

- **Documentation:** https://rfdetr.roboflow.com
- **GitHub:** https://github.com/roboflow/rf-detr
- **Contributing Guide:** .github/CONTRIBUTING.md
- **README:** README.md with usage examples and benchmarks
- **Issues:** https://github.com/roboflow/rf-detr/issues
- **Discord:** https://discord.gg/GbfgXGJ8Bk

## Instructions for Copilot

**Trust these instructions.** Only search for additional information if:

- The instructions are incomplete for your specific task
- You find information in the instructions is incorrect or outdated
- You need specific implementation details not covered here

**When in doubt:**

- Refer to existing code patterns in the repository
- Check `.github/CONTRIBUTING.md` for contribution guidelines
- Look at test files for examples of testing patterns
- Review recent commits for code style examples
