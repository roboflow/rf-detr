# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests that verify rfdetr imports are lazy and don't eagerly load heavy dependencies."""

import subprocess
import sys

import pytest

# Heavy modules that should NOT be loaded by a simple `from rfdetr import RFDETRNano`.
# These are only needed at instantiation/training/inference time.
HEAVY_MODULES = [
    "transformers",
    "peft",
    "scipy",
    "scipy.optimize",
    "supervision",
    "cv2",
    "pycocotools",
    "requests",
]

# All public model classes that must remain importable from the rfdetr package.
PUBLIC_MODEL_CLASSES = [
    "RFDETRBase",
    "RFDETRLarge",
    "RFDETRLargeDeprecated",
    "RFDETRMedium",
    "RFDETRNano",
    "RFDETRSeg2XLarge",
    "RFDETRSegLarge",
    "RFDETRSegMedium",
    "RFDETRSegNano",
    "RFDETRSegPreview",
    "RFDETRSegSmall",
    "RFDETRSegXLarge",
    "RFDETRSmall",
]


def _run_import_check(import_statement: str, forbidden_modules: list[str]) -> str:
    """Run an import in a subprocess and return any forbidden modules that were loaded.

    Args:
        import_statement: The Python import statement to execute.
        forbidden_modules: Module names that should NOT appear in sys.modules.

    Returns:
        Comma-separated string of unexpectedly loaded modules, or empty string.
    """
    modules_check = ", ".join(f'"{m}"' for m in forbidden_modules)
    code = f"""
import sys
{import_statement}
forbidden = [{modules_check}]
loaded = [m for m in forbidden if m in sys.modules]
if loaded:
    print(",".join(loaded))
    exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip(), result.returncode


class TestImportPerformance:
    """Tests ensuring rfdetr uses lazy imports for heavy dependencies."""

    def test_import_rfdetr_does_not_load_detr_eagerly(self) -> None:
        """``import rfdetr`` should not eagerly load rfdetr.detr or heavy modules."""
        loaded, returncode = _run_import_check(
            "import rfdetr",
            [*HEAVY_MODULES, "rfdetr.detr", "rfdetr.main", "torch"],
        )
        assert returncode == 0, f"'import rfdetr' eagerly loaded heavy modules: {loaded}"

    def test_from_rfdetr_import_class_does_not_load_heavy_modules(self) -> None:
        """``from rfdetr import RFDETRNano`` should not load transformers, peft, etc."""
        loaded, returncode = _run_import_check(
            "from rfdetr import RFDETRNano",
            HEAVY_MODULES,
        )
        assert returncode == 0, f"'from rfdetr import RFDETRNano' loaded heavy modules: {loaded}"

    @pytest.mark.parametrize(
        "class_name",
        [pytest.param(name, id=name) for name in PUBLIC_MODEL_CLASSES],
    )
    def test_all_model_classes_importable(self, class_name: str) -> None:
        """Every public model class should be importable from rfdetr."""
        code = f"from rfdetr import {class_name}"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Failed to import {class_name}: {result.stderr}"

    def test_dir_contains_all_public_classes(self) -> None:
        """``dir(rfdetr)`` should list all public model classes."""
        code = f"""
import rfdetr
missing = [name for name in {PUBLIC_MODEL_CLASSES!r} if name not in dir(rfdetr)]
if missing:
    print(",".join(missing))
    exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"dir(rfdetr) is missing classes: {result.stdout.strip()}"
