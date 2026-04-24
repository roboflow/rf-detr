# Contributing to RF-DETR

Thank you for helping to advance RF-DETR! Your participation is invaluable in evolving our platform—whether you’re squashing bugs, refining documentation, or rolling out new features. Every contribution pushes the project forward.

## Table of Contents

1. [How to Contribute](#how-to-contribute)
2. [Project Structure](#project-structure)
3. [Development Environment Setup](#development-environment-setup)
4. [Test-Driven Development](#test-driven-development)
5. [Code Quality and Linting](#code-quality-and-linting)
6. [Building Documentation](#building-documentation)
7. [CLA Signing](#cla-signing)
8. [Google-Style Docstrings and Mandatory Type Hints](#google-style-docstrings-and-mandatory-type-hints)
9. [Reporting Bugs](#reporting-bugs)
10. [Adding a New Model](#adding-a-new-model)
11. [License](#license)

## How to Contribute

Your contributions can be in many forms—whether it’s enhancing existing features, improving documentation, resolving bugs, or proposing new ideas. Here’s a high-level overview to get you started:

1. [Fork the Repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo): Click the “Fork” button on our GitHub page to create your own copy.
2. [Clone Locally](https://docs.github.com/en/enterprise-server@3.11/repositories/creating-and-managing-repositories/cloning-a-repository): Download your fork to your local development environment.
3. [Create a Branch](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop): Use a descriptive name with appropriate prefix:
    ```bash
    # Branch naming convention: {type}/{issue_number}-name_or_description
    git checkout -b fix/123-authentication_bug
    git checkout -b feat/678-add_export_support
    git checkout -b docs/update_readme
    ```
    **Prefixes:** `fix/` (bug fixes), `feat/` (new features), `docs/` (documentation), `refactor/`, `test/`, `chore/`
4. Develop Your Changes: Make your updates, ensuring your commit messages clearly describe your modifications.
5. [Commit and Push](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop): Run:
    ```bash
    git add .
    git commit -m "A brief description of your changes"
    git push -u origin your-descriptive-name
    ```
6. [Open a Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request): Submit your pull request against the main development branch. Please detail your changes and link any related issues.

Before merging, check that all tests pass and that your changes adhere to our development and documentation standards.

## Project Structure

Understanding the project structure will help you navigate the codebase and make contributions effectively.

```
rf-detr/
├── .github/              # GitHub configuration
│   ├── workflows/        # CI/CD pipelines (tests, builds, docs deployment)
│   ├── CONTRIBUTING.md   # This file - contribution guidelines
│   ├── copilot-instructions.md  # GitHub Copilot-specific guidance
│   └── ISSUE_TEMPLATE/   # Issue templates
├── docs/                 # Documentation source (MkDocs)
│   ├── *.md              # Documentation pages
│   └── assets/           # Images and other assets
├── src/rfdetr/           # Main package source code
│   ├── __init__.py       # Package entry point
│   └── ...               # Other modules (models, datasets, utils, etc.)
├── tests/                # Test suite
│   ├── test_*.py         # Test files
│   └── conftest.py       # Pytest configuration and fixtures
├── pyproject.toml        # Project metadata, dependencies, tool configurations
├── mkdocs.yaml           # Documentation configuration
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── README.md             # Project overview and quick start
├── LICENSE               # Apache 2.0 license
└── AGENTS.md             # AI agent-specific technical documentation
```

**Key Directories:**

- **`src/rfdetr/`** - All source code for the RF-DETR package

    - Contains models, datasets, training logic, deployment utilities, and more
    - Internal organization may change as the project evolves

- **`tests/`** - Comprehensive test suite

    - Unit tests, integration tests, and end-to-end tests
    - Use `@pytest.mark.gpu` for GPU-dependent tests

- **`docs/`** - Documentation source files

    - Written in Markdown, built with MkDocs
    - Published to https://rfdetr.roboflow.com

- **`.github/`** - GitHub-specific configuration

    - CI/CD workflows define automated testing and deployment
    - Contributing guidelines and issue templates

**Important Configuration Files:**

- **`pyproject.toml`** - Single source of truth for:

    - Project metadata and dependencies
    - Tool configurations (ruff, pytest, coverage, etc.)
    - Build system configuration

- **`.pre-commit-config.yaml`** - Defines pre-commit hooks for code quality

- **`mkdocs.yml`** - Documentation site configuration

> [!TIP]
> When contributing, focus on the relevant directory for your change:
>
> - Bug fixes/features → `src/rfdetr/` and `tests/`
> - Documentation → `docs/`
> - CI/build issues → `.github/workflows/` or config files

## Development Environment Setup

RF-DETR uses **`uv`** as the package manager for dependency management. Ensure you have Python >=3.10 installed (supports 3.10, 3.11, 3.12, 3.13).

### Installing uv

```bash
pip install uv
```

### Setting Up Your Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rf-detr.git
cd rf-detr

# Install all development dependencies
uv sync --all-groups

# Or install specific dependency groups
uv sync --group tests      # Testing dependencies only
uv sync --group docs       # Documentation dependencies only
uv sync --group build      # Build tools only
```

**Important:** Always run `uv sync` after pulling changes to ensure your dependencies are up to date.

### Running Tests

> **CI Workflows as Source of Truth:** See `.github/workflows/ci-tests-cpu.yml` and `.github/workflows/ci-tests-gpu.yml` for the exact commands used in continuous integration.

```bash
# Run CPU tests (default for local development)
uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu" --cov=rfdetr --cov-report=xml

# Run GPU tests (requires GPU)
uv run --no-sync pytest src/ tests/ -n 2 -m gpu
```

**Development vs. PR Requirements:**

- **During development:** Tests may fail as you work through TDD cycle (write failing test → implement → fix)
- **Before opening PR:** Your final commit MUST have all tests passing
- **Before each commit:** Run `pre-commit run --all-files` to ensure code quality

### Building the Package

```bash
# Build source and wheel distributions
uv build

# Validate the build
uv run twine check --strict dist/*
```

## Test-Driven Development

We follow test-driven development practices to ensure code quality and prevent regressions.

### For Bug Fixes

1. **Write a test that replicates the issue** - The test should fail initially, demonstrating the bug
2. **Commit the failing test** (optional during development, but commit message should note "WIP" or "test for issue #XXX")
3. **Implement the fix** - Make the minimal change needed to make the test pass
4. **Verify all tests pass** - Ensure your fix doesn't break existing functionality
5. **Commit the fix** - This commit MUST have all tests passing before opening PR

**Note:** It's acceptable to have failing tests in intermediate commits during development. However, your **final commit before opening a PR must have all tests passing**. This aligns with test-driven development: first create a failing test that proves the bug exists, then fix it.

### For New Features

1. **Write tests covering all major use cases** - Think about edge cases, invalid inputs, and expected behaviors
2. **Implement the feature** - Build the feature to satisfy the test requirements
3. **Refactor if needed** - Clean up the implementation while keeping tests green

### Test Organization

**Use test classes to group related tests:**

```python
import pytest


class TestModelInference:
    def test_single_image_inference(self):
        # Test code
        pass

    def test_batch_inference(self):
        # Test code
        pass
```

**Use `pytest.mark.parametrize` to extend test cases:**

```python
import pytest


@pytest.mark.parametrize(
    "model_variant",
    [
        pytest.param("nano", id="nano"),
        pytest.param("small", id="small"),
        pytest.param("medium", id="medium"),
    ],
)
def test_model_loading(model_variant):
    # Test code that runs for each model variant
    pass
```

**Avoid multiple validation cases in a single test:**

Do not write tests that loop through multiple cases internally. Instead, use `@pytest.mark.parametrize` so each case runs as a separate test:

```python
import pytest
from rfdetr.assets.model_weights import ModelWeights


# BAD: Multiple cases in one test - all assertions must pass for test to pass
def test_all_models_have_valid_urls():
    for model in ModelWeights:
        assert model.url.startswith("http")  # Hard to identify which model failed


# GOOD: Parametrized - each model is a separate test case
@pytest.mark.parametrize("model", list(ModelWeights), ids=[m.filename for m in ModelWeights])
def test_all_models_have_valid_urls(model):
    assert model.url.startswith("http")  # Clear which model failed
```

Benefits of parametrization:

- Each case runs as an independent test (failures are isolated)
- Test IDs clearly identify which case failed
- Easier to debug and maintain
- Better test reporting in CI

**Mark GPU-required or computationally heavy tests:**

```python
import pytest


@pytest.mark.gpu  # Use this marker for GPU-dependent or heavy tests (e.g., training)
def test_model_training():
    # Training test code
    pass
```

Tests marked with `@pytest.mark.gpu` are excluded from CPU CI workflows and run separately on GPU infrastructure.

### CI Testing

> [!NOTE]
> **CI Workflows (Source of Truth):** See `.github/workflows/ci-tests-cpu.yml` and `.github/workflows/ci-tests-gpu.yml` for exact commands.

Our continuous integration tests run on:

- **Operating Systems:** Ubuntu, Windows, macOS
- **Python Versions:** 3.10, 3.11, 3.12, 3.13
- **CPU Workflow:** `pytest -m "not gpu"` - Runs on all OS/Python combinations
- **GPU Workflow:** `pytest -m gpu` - Runs separately on GPU infrastructure

This ensures your changes work across all supported platforms and Python versions.

### Running Tests

```bash
# Run tests with parallel execution (recommended)
uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu"

# Run a specific test file
uv run --no-sync pytest tests/test_model.py

# Run a specific test
uv run --no-sync pytest tests/test_model.py::test_model_loading
```

## Code Quality and Linting

All code must pass linting and formatting checks before being merged. We use **pre-commit hooks** to automate this process.

> [!TIP]
> Pre-commit hooks will auto-format many issues. If pre-commit fails, review the changes it made and re-stage the files.

### Setting Up Pre-commit

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

**Configuration:** See `.pre-commit-config.yaml` for all hooks and `pyproject.toml` for tool-specific settings (e.g., `[tool.ruff]`).

## Building Documentation

RF-DETR's documentation is built with [MkDocs](https://www.mkdocs.org/) and the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme. API reference pages are auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

> [!NOTE]
> Building the full documentation locally requires the `plus` extra (`rfdetr[plus]`), which provides the XLarge and 2XLarge model pages. Without it, the build will fail on those reference pages.

### Install Documentation Dependencies

```bash
# Full docs build (matches CI — required for XLarge/2XLarge model pages)
uv pip install -e ".[plus]" --group docs

# Minimal install (skip plus models — XLarge/2XLarge pages will error)
uv sync --group docs
```

### Serve Locally with Live Reload

```bash
uv run mkdocs serve
```

Open [http://localhost:8000](http://localhost:8000) in your browser. The server watches for file changes and reloads automatically — no restart needed as you edit documentation.

### Build Static Site

```bash
# Build static documentation site to the site/ directory
uv run mkdocs build
```

### Documentation Structure

```
docs/
├── index.md              # Home page
├── learn/                # How-to guides and tutorials
│   ├── install.md
│   ├── run/              # Detection and segmentation guides
│   └── train/            # Training guides (parameters, augmentations, loggers, etc.)
├── reference/            # Auto-generated API reference (from docstrings)
├── tutorials/
└── theme/                # Custom theme overrides
mkdocs.yaml               # MkDocs configuration and navigation
```

> [!TIP]
> When adding a new documentation page, add it to the `nav` section in `mkdocs.yaml` so it appears in the site navigation. Pages that exist in `docs/` but are not listed in `nav` will not be included in the site.

## CLA Signing

In order to maintain the integrity of our project, every pull request must include a signed Contributor License Agreement (CLA). This confirms that your contributions are properly licensed under our Apache 2.0 License. After opening your pull request, simply add a comment stating:

```
I have read the CLA Document and I sign the CLA.
```

This step is essential before any merge can occur.

## Google-Style Docstrings and Mandatory Type Hints

For clarity and maintainability, any new functions or classes must include [Google-style docstrings](https://google.github.io/styleguide/pyguide.html) and use Python type hints. Type hints are mandatory in all function definitions, ensuring explicit parameter and return type declarations.

> [!IMPORTANT]
> Type hints are in the function signature. **Do not duplicate types in docstrings** - describe the parameter's purpose instead.

For example:

```python
def sample_function(param1: int, param2: int = 10) -> bool:
    """
    Provides a brief description of function behavior.

    Args:
        param1: Explanation of the first parameter's purpose.
        param2: Explanation of the second parameter, defaulting to 10.

    Returns:
        True if the operation succeeds, otherwise False.

    Examples:
        >>> sample_function(5, 10)
        True
    """
    return param1 == param2
```

Following this pattern helps ensure consistency throughout the codebase.

## Reporting Bugs

Bug reports are vital for continued improvement. When reporting an issue, please include a clear, minimal reproducible example that demonstrates the problem. Detailed bug reports assist us in swiftly diagnosing and addressing issues.

## Adding a New Model

> [!IMPORTANT]
> Before implementing a new model, **discuss with maintainers first**. Project structure and patterns are subject to change.

**General workflow:**

1. **Open an issue** describing the proposed model and approach
    - You may ask maintainers to confirm the expected evaluation protocol (dataset, metrics) before running full benchmarks
2. **Demonstrate improvement** versus reference models on a standard public dataset (e.g., COCO val2017)
    - If the change is for an existing RF-DETR model, show a case where the new approach is Pareto optimal (e.g., better accuracy at similar or lower latency/model size) over the existing model
    - If the change is adding a new functionality, show a case where the new approach is Pareto optimal over comparable third-party models (see the [README model table](../README.md) for reference baselines)
    - Provide a script for us to reproduce your results
3. **Wait for maintainer feedback** on architecture and integration approach
4. **Follow test-driven development:**
    - Write comprehensive tests for the new model
    - Implement the model following approved approach
    - Ensure all tests pass
5. **Add documentation** as directed by maintainers
6. **Submit PR** with reference to the discussion issue

Maintainers will guide you on specific files to modify and patterns to follow based on current project architecture.

## License

By contributing to RF-DETR, you agree that your contributions will be licensed under the Apache 2.0 License as specified in our [LICENSE](/LICENSE) file.

Thank you for your commitment to making RF-DETR better. We look forward to your pull requests and continued collaboration. Happy coding!

### License Headers

All Python files must start with the following header:

```python
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
```
