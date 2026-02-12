# RF-DETR Copilot Instructions

> [!NOTE]
> This document is GitHub Copilot-specific guidance. For canonical contribution guidelines (test-driven development, code quality, docstrings, etc.), see [CONTRIBUTING.md](CONTRIBUTING.md). For detailed agent-specific context, see [AGENTS.md](../AGENTS.md).

## Repository Overview

RF-DETR is a real-time transformer architecture for object detection and instance segmentation. Built on DINOv2 vision transformer backbone with PyTorch.

**Project Type:** Python ML library (computer vision)
**Python:** >=3.10 (3.10, 3.11, 3.12, 3.13)
**License:** Apache 2.0 (Plus models under PML 1.0)

> [!TIP]
>
> - **Configuration:** See `pyproject.toml` for dependencies, build settings, and tool configurations.
> - **Contributing:** See `.github/CONTRIBUTING.md` for contribution guidelines, CLA, and coding standards.

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

> [!IMPORTANT]
> Run `uv sync` after pulling changes to update dependencies.

## Code Quality

**Linting & Formatting:** All code must pass pre-commit checks. See **[Code Quality and Linting](CONTRIBUTING.md#code-quality-and-linting)** in CONTRIBUTING.md for setup and details.

```bash
pre-commit run --all-files
```

> **Configuration:** `.pre-commit-config.yaml` (hooks) and `[tool.ruff]` in `pyproject.toml` (Python linting)

## Key Conventions

> [!NOTE]
> Internal package organization (`src/rfdetr/`) is subject to change as this is an active research project. Explore the codebase to understand current module organization.

**Imports:**

- Always use direct imports: `from rfdetr.util.misc import get_rank, is_main_process`
- Logger: `from rfdetr.util.logger import get_logger` (reads `LOG_LEVEL` env var)
- TQDM: `from tqdm.auto import tqdm` (NOT `from tqdm import tqdm`)

## Testing & Development Workflow

**Test-Driven Development:** Follow TDD practices - write tests first for bugs, comprehensive tests for features. See **[Test-Driven Development](CONTRIBUTING.md#test-driven-development)** in CONTRIBUTING.md for detailed guidelines.

**Quick reference:**

- Bug fixes: Write failing test → Fix → Verify all pass
- Features: Write comprehensive tests → Implement → Refactor
- Use test classes and `@pytest.mark.parametrize` for organization
- Mark GPU/heavy tests with `@pytest.mark.gpu`

**Testing Requirements:**

- ⚠️ During development: Tests may fail (TDD cycle is fine)
- ✅ Before PR: Final commit MUST have all tests passing
- ✅ Before commit: Run `pre-commit run --all-files`

**CI/CD:** See `.github/workflows/` for source of truth. Tests run on Python 3.10-3.13 across Ubuntu, Windows, macOS.

## Coding Standards

**Type Hints & Docstrings:** MANDATORY for all functions/classes. See **[Google-Style Docstrings and Mandatory Type Hints](CONTRIBUTING.md#google-style-docstrings-and-mandatory-type-hints)** in CONTRIBUTING.md for examples.

**Import Conventions:**

```python
# Always use direct imports (NOT import ... as pattern)
from rfdetr.util.misc import get_rank, is_main_process, save_on_master
from rfdetr.util.logger import get_logger

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

## Maintaining Agentic Documentation

**If your contribution:**

- Changes project structure or introduces new patterns
- Receives major feedback in PR review about conventions/patterns

**Then update the relevant documents:**

- This file (copilot-instructions.md) for high-level guidance
- AGENTS.md for detailed technical patterns
- CONTRIBUTING.md if it affects human contribution workflow

This ensures future contributions stay consistent and reduces repeated feedback.

---

**Note:** These instructions are GitHub Copilot-specific. When in doubt, refer to existing code patterns, contributing guidelines, and test files for examples.
