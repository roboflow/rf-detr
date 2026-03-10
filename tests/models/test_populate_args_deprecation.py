# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the populate_args() deprecation shim — PTL Ch4/T3.

populate_args() must remain fully functional (returns a valid argparse.Namespace)
while emitting a FutureWarning on the first call to guide users toward
`build_trainer()` / `RFDETR.train()`.

Note on warning isolation: the `deprecate` library (num_warns=1 default) fires
the warning only once per Python session.  The live warning test therefore runs
in a subprocess so that other tests in the same worker cannot consume the quota.
"""

import argparse
import subprocess
import sys
import warnings

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call(**kwargs):
    """Minimal valid populate_args() call that suppresses the FutureWarning."""
    from rfdetr.main import populate_args

    defaults = dict(num_classes=3, dataset_dir="/tmp", output_dir="/tmp")
    defaults.update(kwargs)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return populate_args(**defaults)


# ---------------------------------------------------------------------------
# Deprecation warning
# ---------------------------------------------------------------------------


class TestPopulateArgsDeprecationWarning:
    """populate_args() emits a FutureWarning on the first call in a session."""

    def test_warning_emitted_with_correct_message(self):
        """FutureWarning is emitted; message names populate_args, version, and replacement.

        Runs in a subprocess so the num_warns=1 quota is fresh regardless of
        what other tests have already called populate_args().
        """
        script = (
            "import warnings, sys\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    from rfdetr.main import populate_args\n"
            "    populate_args(num_classes=3, dataset_dir='/tmp', output_dir='/tmp')\n"
            "msgs = [str(x.message) for x in w if issubclass(x.category, FutureWarning)]\n"
            "if not msgs:\n"
            "    sys.exit('NO_FUTURE_WARNING')\n"
            "msg = msgs[0]\n"
            "assert 'populate_args' in msg, f'name missing: {msg!r}'\n"
            "assert 'build_trainer' in msg, f'replacement missing: {msg!r}'\n"
            "assert '1.5.1' in msg, f'version missing: {msg!r}'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        assert "OK" in result.stdout

    def test_decorator_applied(self):
        """populate_args has a __wrapped__ attribute indicating the decorator is present."""
        from rfdetr.main import populate_args

        assert hasattr(populate_args, "__wrapped__"), (
            "populate_args must be decorated with @deprecated; __wrapped__ is missing"
        )

    def test_deprecated_metadata_version(self):
        """deprecated_in is set to '1.5.1' in the decorator metadata."""
        from rfdetr.main import populate_args

        meta = populate_args.__deprecated__
        assert meta.get("deprecated_in") == "1.5.1"

    def test_deprecated_metadata_remove_in(self):
        """remove_in is set to '2.0.0' in the decorator metadata."""
        from rfdetr.main import populate_args

        meta = populate_args.__deprecated__
        assert meta.get("remove_in") == "2.0.0"


# ---------------------------------------------------------------------------
# Functional correctness — populate_args() must still work as a shim
# ---------------------------------------------------------------------------


class TestPopulateArgsStillFunctional:
    """populate_args() must remain a working compatibility shim despite deprecation."""

    def test_returns_namespace(self):
        """Return type is argparse.Namespace."""
        assert isinstance(_call(), argparse.Namespace)

    def test_num_classes_forwarded(self):
        """num_classes kwarg is forwarded to the returned Namespace."""
        assert _call(num_classes=7).num_classes == 7

    @pytest.mark.parametrize(
        "field, value",
        [
            pytest.param("lr", 2e-4, id="lr"),
            pytest.param("epochs", 50, id="epochs"),
            pytest.param("batch_size", 8, id="batch_size"),
            pytest.param("grad_accum_steps", 2, id="grad_accum_steps"),
        ],
    )
    def test_kwargs_forwarded_to_namespace(self, field, value):
        """Arbitrary kwargs are faithfully forwarded to the returned Namespace."""
        assert getattr(_call(**{field: value}), field) == value

    def test_segmentation_loss_defaults_present(self):
        """Segmentation loss coefficients default to the legacy expected values."""
        args = _call(segmentation_head=True)
        assert args.mask_ce_loss_coef == pytest.approx(5.0)
        assert args.mask_dice_loss_coef == pytest.approx(5.0)
        assert args.mask_point_sample_ratio == 16

    def test_segmentation_loss_overrides_forwarded(self):
        """Explicit segmentation loss kwargs are forwarded to Namespace."""
        args = _call(
            segmentation_head=True,
            mask_ce_loss_coef=7.5,
            mask_dice_loss_coef=3.25,
            mask_point_sample_ratio=8,
        )
        assert args.mask_ce_loss_coef == pytest.approx(7.5)
        assert args.mask_dice_loss_coef == pytest.approx(3.25)
        assert args.mask_point_sample_ratio == 8
