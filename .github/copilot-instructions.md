# RF-DETR Copilot Instructions

## Repository Overview

RF-DETR is a real-time transformer architecture for object detection and instance segmentation. Built on DINOv2 vision transformer backbone with PyTorch.

**Project Type:** Python ML library (computer vision)
**Python:** >=3.10 (3.10, 3.11, 3.12, 3.13)
**License:** Apache 2.0 (Plus models under PML 1.0)

> **Configuration:** See `pyproject.toml` for dependencies, build settings, and tool configurations.
> **Contributing:** See `.github/CONTRIBUTING.md` for contribution guidelines, CLA, and coding standards.

## Quick Start

**Package Manager:** This project uses `uv` for all dependency management.

```bash
# Development setup
uv sync --all-groups

# Run tests (always before committing)
uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu" --cov=rfdetr --cov-report=xml

# Build package
uv build
```

**IMPORTANT:** Run `uv sync` after pulling changes to update dependencies.

## Code Quality

**Linting & Formatting:** Configured via `.pre-commit-config.yaml` - enforces ruff, mdformat, prettier, codespell, and license headers.

```bash
# Install and run pre-commit
pre-commit install
pre-commit run --all-files
```

> **Configuration:** See `.pre-commit-config.yaml` for all linting rules and formatters.
> **Ruff settings:** See `[tool.ruff]` in `pyproject.toml` for specific rules and exclusions.

## Project Structure

```
src/rfdetr/
├── main.py           # Core training/evaluation logic, CLI entry point
├── detr.py           # Model wrappers (RFDETR wrappers)
├── config.py         # Configuration dataclasses (TrainConfig)
├── cli/              # Command-line interface
├── datasets/         # Dataset implementations
├── deploy/           # Export utilities (ONNX, TensorRT)
├── models/           # Model architectures (backbone, transformer, heads)
├── platform/         # Platform integration (Roboflow API, Plus models)
└── util/             # Utilities (logger, distributed training)
```

**Key Conventions:**

- RFDETR wrappers: `self.model` → `rfdetr.main.Model` instance, `self.model.model` → underlying torch module
- Distributed utils: `import rfdetr.util.misc as utils`, call as `utils.get_rank()`, `utils.is_main_process()`
- Logger: `rfdetr.util.logger.get_logger()` (reads `LOG_LEVEL` env var)
- Plus models (XLarge/2XLarge): Imported from `rfdetr_plus` with lazy error handling via `__getattr__`

## Testing Strategy

**Test Organization:**

- Group related tests in classes
- Use `@pytest.mark.parametrize` to extend test cases
- Mark GPU-required or computationally heavy tests (e.g., training) with `@pytest.mark.gpu`

**Development Workflow:**

1. **For bug fixes:** Start by writing a test that replicates the issue, then fix the code
2. **For features:** Write tests covering all major use cases
3. Run tests with `-n 2` for parallel execution (pytest-xdist)

```python
# Example: Parameterized test with GPU marker
@pytest.mark.gpu
@pytest.mark.parametrize("model_name", [
    pytest.param("nano", id="nano"),
    pytest.param("small", id="small"),
])
class TestModelTraining:
    def test_train_convergence(self, model_name):
        # Test implementation
        pass
```

**CI/CD:** Tests run on Python 3.10-3.13 across Ubuntu, Windows, macOS. See `.github/workflows/` for workflow configurations.

## Coding Standards

**Type Hints & Docstrings:**

- **MANDATORY** type hints for all function parameters and returns
- **MANDATORY** Google-style docstrings for all new functions/classes
- See `.github/CONTRIBUTING.md` for examples and detailed requirements

**Import Conventions:**

```python
# Distributed training utilities
import rfdetr.util.misc as utils
utils.get_rank(), utils.is_main_process()

# TQDM (for environment compatibility)
from tqdm.auto import tqdm  # NOT from tqdm import tqdm
```

**Project-Specific Patterns:**

- **Logging:** Use `logger.debug()` for detailed tensor/shape info (not `logger.info()`)
- **Segmentation models:** Return `pred_masks` as `torch.Tensor` or dict with keys `['spatial_features', 'query_features', 'bias']`
- **Checkpoint handling:** Always check file existence before operations
- **License headers:** All Python files require Apache 2.0 header (enforced by pre-commit)

**Best Practices:**

- Make minimal, surgical changes - avoid over-engineering
- Use existing patterns and libraries
- Write secure code - avoid injection vulnerabilities (XSS, SQL injection, command injection)
- Follow Python ML development best practices

## Pre-Commit Checklist

Before submitting changes:

1. ✅ Run tests: `uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu"`
2. ✅ Run pre-commit: `pre-commit run --all-files`
3. ✅ Verify new functions have type hints + docstrings
4. ✅ Review changes for minimal scope

## Resources

- **Docs:** https://rfdetr.roboflow.com
- **Contributing:** `.github/CONTRIBUTING.md`
- **Config:** `pyproject.toml`, `.pre-commit-config.yaml`
- **Issues:** https://github.com/roboflow/rf-detr/issues

---

**Note:** These instructions are GitHub Copilot-specific. When in doubt, refer to existing code patterns, contributing guidelines, and test files for examples.
