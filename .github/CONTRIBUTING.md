# Contributing to RF-DETR

Thank you for helping to advance RF-DETR! Your participation is invaluable in evolving our platform—whether you’re squashing bugs, refining documentation, or rolling out new features. Every contribution pushes the project forward.

## Table of Contents

1. [How to Contribute](#how-to-contribute)
2. [Development Environment Setup](#development-environment-setup)
3. [Test-Driven Development](#test-driven-development)
4. [Code Quality and Linting](#code-quality-and-linting)
5. [CLA Signing](#cla-signing)
6. [Google-Style Docstrings and Mandatory Type Hints](#google-style-docstrings-and-mandatory-type-hints)
7. [Reporting Bugs](#reporting-bugs)
8. [Adding a New Model](#adding-a-new-model)
9. [License](#license)

## How to Contribute

Your contributions can be in many forms—whether it’s enhancing existing features, improving documentation, resolving bugs, or proposing new ideas. Here’s a high-level overview to get you started:

1. [Fork the Repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo): Click the “Fork” button on our GitHub page to create your own copy.
2. [Clone Locally](https://docs.github.com/en/enterprise-server@3.11/repositories/creating-and-managing-repositories/cloning-a-repository): Download your fork to your local development environment.
3. [Create a Branch](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop): Use a descriptive name to create a new branch (e.g., `feature/your-descriptive-name`):
    ```bash
    git checkout -b feature/your-descriptive-name
    ```
4. Develop Your Changes: Make your updates, ensuring your commit messages clearly describe your modifications.
5. [Commit and Push](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop): Run:
    ```bash
    git add .
    git commit -m "A brief description of your changes"
    git push -u origin your-descriptive-name
    ```
6. [Open a Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request): Submit your pull request against the main development branch. Please detail your changes and link any related issues.

Before merging, check that all tests pass and that your changes adhere to our development and documentation standards.

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

@pytest.mark.parametrize("model_variant", [
    pytest.param("nano", id="nano"),
    pytest.param("small", id="small"),
    pytest.param("medium", id="medium"),
])
def test_model_loading(model_variant):
    # Test code that runs for each model variant
    pass
```

**Mark GPU-required or computationally heavy tests:**

```python
import pytest

@pytest.mark.gpu  # Use this marker for GPU-dependent or heavy tests (e.g., training)
def test_model_training():
    # Training test code
    pass
```

Tests marked with `@pytest.mark.gpu` are excluded from CPU CI workflows and run separately on GPU infrastructure.

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

### Setting Up Pre-commit

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### What Gets Checked

Pre-commit hooks (configured in `.pre-commit-config.yaml`) include:

- **ruff**: Python linting and formatting (configuration in `pyproject.toml`)
- **mdformat**: Markdown formatting
- **prettier**: YAML/TOML formatting
- **codespell**: Spell checking
- **License headers**: Ensures all Python files have Apache 2.0 header

### Manual Linting

```bash
# Run ruff linter with auto-fix
ruff check --fix .

# Format code with ruff
ruff format .
```

**Note:** Pre-commit hooks will auto-format many issues. If pre-commit fails, review the changes it made and re-stage the files.

## CLA Signing

In order to maintain the integrity of our project, every pull request must include a signed Contributor License Agreement (CLA). This confirms that your contributions are properly licensed under our Apache 2.0 License. After opening your pull request, simply add a comment stating:

```
I have read the CLA Document and I sign the CLA.
```

This step is essential before any merge can occur.

## Google-Style Docstrings and Mandatory Type Hints

For clarity and maintainability, any new functions or classes must include [Google-style docstrings](https://google.github.io/styleguide/pyguide.html) and use Python type hints. Type hints are mandatory in all function definitions, ensuring explicit parameter and return type declarations. These docstrings should clearly explain parameters, return types, and provide usage examples when applicable.

For example:

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

Following this pattern helps ensure consistency throughout the codebase.

## Reporting Bugs

Bug reports are vital for continued improvement. When reporting an issue, please include a clear, minimal reproducible example that demonstrates the problem. Detailed bug reports assist us in swiftly diagnosing and addressing issues.

## Adding a New Model

When adding a new model variant to RF-DETR:

1. **Define the model architecture** in `src/rfdetr/models/`
2. **Add model configuration** to `src/rfdetr/config.py`
3. **Create model wrapper** in `src/rfdetr/detr.py` or relevant module
4. **Write comprehensive tests** covering:
   - Model instantiation
   - Forward pass with various input shapes
   - Training compatibility
   - Export functionality (ONNX, TensorRT if applicable)
5. **Add documentation** to `docs/` directory
6. **Update README.md** with model benchmarks and usage examples

Follow the existing patterns in the codebase for consistency. See existing model implementations (e.g., RFDETRNano, RFDETRSmall) as examples.

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
