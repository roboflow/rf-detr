# RF-DETR - Agent Instructions

This file provides detailed technical context for AI coding agents working with RF-DETR.

**Canonical Sources:**

- **Contribution Guidelines:** [CONTRIBUTING.md](.github/CONTRIBUTING.md) - The authoritative source for all contribution practices
- **Human Documentation:** [README.md](README.md) - Project overview and usage
- **Copilot Instructions:** [.github/copilot-instructions.md](.github/copilot-instructions.md) - GitHub Copilot-specific guidance

This document supplements the contribution guidelines with detailed technical information for automated tooling.

## Agent Responsibilities

As an AI agent contributing to RF-DETR, you are responsible for:

1. **Following test-driven development practices**

    - Write failing tests first for bug fixes
    - Write comprehensive tests for new features
    - Ensure final PR commit has all tests passing

2. **Adhering to code quality standards**

    - Run `pre-commit run --all-files` before every commit
    - Follow type hint and docstring requirements
    - Use direct imports (not `import ... as` pattern)

3. **Maintaining agentic documentation**

    - Update `AGENTS.md` when architecture patterns or technical conventions change
    - Update `.github/copilot-instructions.md` when high-level guidance changes
    - Update `.github/CONTRIBUTING.md` when human workflow is affected
    - Apply updates after receiving major feedback in PR reviews

4. **Consulting maintainers before major changes**

    - Open an issue before adding new models or significant features
    - Wait for approval on approach before implementing

5. **Writing secure, minimal code**

    - Avoid over-engineering and unnecessary abstractions
    - Write secure code (prevent injection vulnerabilities)
    - Follow existing patterns in the codebase

> [!NOTE]
> Keeping documentation current ensures consistency across agent contributions and reduces repeated feedback on the same issues.

## Build & Development Environment

> [!NOTE]
> **Canonical Reference:** See [Development Environment Setup](.github/CONTRIBUTING.md#development-environment-setup) in CONTRIBUTING.md for complete setup instructions.

### Setup

```bash
# Install uv (if not already installed)
pip install uv

# Full development environment (always use this)
uv sync --all-groups
```

**Prerequisites:** Python >=3.10 (tested on 3.10-3.13)

### Dependency Information

See `pyproject.toml` for complete dependency specifications:

- **Core:** PyTorch, torchvision, transformers, pycocotools, supervision, peft, pydantic
- **Optional:** `[plus]` (Plus models), `[onnxexport]` (ONNX export), `[metrics]` (tensorboard, wandb)
- **Development:** `tests`, `docs`, `build` groups

**Important version constraints:**

- PyTorch: >=1.13.0, \<=2.8.0 (2.9.0+ excluded due to known issues)
- Transformers: >4.0.0, \<5.0.0

## Testing

> [!NOTE]
> **Canonical Reference:** See [Test-Driven Development](.github/CONTRIBUTING.md#test-driven-development) in CONTRIBUTING.md for complete guidelines.
>
> **CI Workflows (Source of Truth):** See `.github/workflows/ci-tests-cpu.yml` and `.github/workflows/ci-tests-gpu.yml` for exact test commands used in CI.

### Commands

```bash
# CPU tests (default for local development) - matches CI
uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu" --cov=rfdetr --cov-report=xml

# GPU tests (requires GPU)
uv run --no-sync pytest src/ tests/ -n 2 -m gpu

# Pre-commit checks (ALWAYS run before committing)
pre-commit run --all-files
```

### Testing Principles

> [!IMPORTANT]
> **Testing Requirements:**
>
> - ⚠️ **During development:** Tests may fail as you work through TDD cycle
> - ✅ **Before opening PR:** Final commit MUST have all tests passing
> - ✅ **Before each commit:** Run `pre-commit run --all-files`

**Test-Driven Development:**

1. **Bug fixes:** Write failing test → Fix code → Verify all tests pass
2. **New features:** Write comprehensive tests → Implement feature → Refactor

**Test Organization:**

- Group related tests in classes
- Use `@pytest.mark.parametrize` with `pytest.param(..., id="name")`
- Mark GPU/heavy tests with `@pytest.mark.gpu`
- Avoid multiple validation cases in a single test - see [CONTRIBUTING.md](.github/CONTRIBUTING.md#avoid-multiple-validation-cases-in-a-single-test) for details

**CI Information:**
See [CI Testing](.github/CONTRIBUTING.md#ci-testing) in CONTRIBUTING.md for details on OS/Python version matrix and workflow configurations.

## Code Quality & Linting

> [!NOTE]
> **Canonical Reference:** See [Code Quality and Linting](.github/CONTRIBUTING.md#code-quality-and-linting) in CONTRIBUTING.md for setup and details.

### Command

```bash
# Always run full pre-commit (not individual tools)
pre-commit run --all-files
```

> [!TIP]
> Pre-commit hooks will auto-format many issues. Review changes and re-stage files.

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
# Full install (matches CI — required for XLarge/2XLarge model pages)
uv pip install -e ".[plus]" --group docs

# Serve locally (live reload)
uv run mkdocs serve

# Build static site
uv run mkdocs build
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

## Project Structure

> [!NOTE]
> **Canonical Reference:** See [Project Structure](.github/CONTRIBUTING.md#project-structure) in CONTRIBUTING.md for complete project organization, directory descriptions, and configuration files.
>
> **Quick summary:** `src/rfdetr/` (source code), `tests/` (test suite), `docs/` (documentation), `.github/` (CI/CD), `pyproject.toml` (dependencies and config).
>
> Internal package organization within `src/rfdetr/` is subject to change as this is an active research and development project.

## Architecture & Conventions

### Key Patterns

**Model Architecture:**

- RFDETR wrappers: `self.model` is `rfdetr.main.Model` instance
- Underlying PyTorch module: `self.model.model`
- Segmentation models return `pred_masks` as `torch.Tensor` or dict with keys `['spatial_features', 'query_features', 'bias']`

**Imports:**

```python
# Always use direct imports (NOT import ... as pattern)
from rfdetr.util.misc import get_rank, get_world_size, is_main_process, save_on_master
from rfdetr.util.logger import get_logger

# Logger usage
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
    check=True,  # Raise CalledProcessError on failure
    text=True,  # Return stdout/stderr as strings
    capture_output=True,
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

> [!IMPORTANT]
> **Canonical Reference:** See [Google-Style Docstrings and Mandatory Type Hints](.github/CONTRIBUTING.md#google-style-docstrings-and-mandatory-type-hints) in CONTRIBUTING.md for complete requirements and examples.

**Requirements:**

- MANDATORY type hints for all function parameters and return types
- MANDATORY Google-style docstrings for all functions and classes
- **Do not duplicate types in docstrings** - types are in the function signature
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

> [!IMPORTANT]
> **Canonical Reference:** See [Adding a New Model](.github/CONTRIBUTING.md#adding-a-new-model) in CONTRIBUTING.md for detailed guidance.
>
> Always consult maintainers before implementing new models.

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
