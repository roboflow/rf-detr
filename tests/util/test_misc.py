# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math

import pytest
import torch

from rfdetr.util.misc import SmoothedValue


class TestSmoothedValue:
    @pytest.mark.parametrize("window_size, fmt", [
        (10, "{avg:.2f}"),
        (20, None),
        (5, "{median:.1f} ({global_avg:.1f})"),
    ])
    def test_init(self, window_size, fmt):
        """
        Validate the initialization of SmoothedValue.

        Check if the window size, format string, and initial counters (total, count)
        are correctly set.
        """
        sv = SmoothedValue(window_size=window_size, fmt=fmt)
        if fmt is None:  # default
            fmt = "{median:.4f} ({global_avg:.4f})"
        assert sv.fmt == fmt
        assert sv.deque.maxlen == window_size
        assert sv.total == 0.0
        assert sv.count == 0

    @pytest.mark.parametrize("updates, expected_count, expected_total", [
        ([(1.0, 1)], 1, 1.0),
        ([(1.0, 1), (2.0, 2)], 3, 5.0),
        ([(1.0, 1), (2.0, 1), (3.0, 1)], 3, 6.0),
    ])
    def test_update(self, updates, expected_count, expected_total):
        """
        Validate the update method of SmoothedValue.

        Check if the count, total, and the most recent value are correctly
        updated after a series of inputs.
        """
        sv = SmoothedValue()
        for val, n in updates:
            sv.update(val, n=n)
            assert sv.value == val

        assert sv.count == expected_count
        assert sv.total == pytest.approx(expected_total)

    @pytest.mark.parametrize("window_size, updates, expected", [
        (3, [1.0, 3.0, 2.0], {"value": 2.0, "max": 3.0, "avg": 2.0, "global_avg": 2.0, "median": 2.0}),
        (3, [1.0, 3.0, 2.0, 4.0], {"value": 4.0, "max": 4.0, "avg": 3.0, "global_avg": 2.5, "median": 3.0}),
    ])
    def test_properties(self, window_size, updates, expected):
        """
        Validate the calculated properties of SmoothedValue.

        Check if value, max, avg, global_avg, and median are correctly computed,
        both before and after the window size is reached.
        """
        sv = SmoothedValue(window_size=window_size)
        for val in updates:
            sv.update(val)

        assert sv.value == expected["value"]
        assert sv.max == expected["max"]
        assert sv.avg == pytest.approx(expected["avg"])
        assert sv.global_avg == pytest.approx(expected["global_avg"])
        assert sv.median == pytest.approx(expected["median"])

    def test_str(self):
        """
        Validate the __str__ representation of SmoothedValue.

        Check if the output string correctly follows the provided format string
        using current values.
        """
        sv = SmoothedValue(window_size=3, fmt="{median:.1f} ({global_avg:.1f})")
        sv.update(1.0)
        sv.update(2.0)

        d = torch.tensor([1.0, 2.0])
        expected_median = d.median().item()
        expected_global_avg = 1.5

        assert f"{expected_median:.1f} ({expected_global_avg:.1f})" == str(sv)

    @pytest.mark.parametrize("property_name, expected_exception, expected_message", [
        ("max", ValueError, r"max\(\) (arg is an empty sequence|iterable argument is empty)"),
        ("value", IndexError, "deque index out of range"),
        ("global_avg", ZeroDivisionError, "float division by zero"),
    ])
    def test_empty_exceptions(self, property_name, expected_exception, expected_message):
        """
        Validate that accessing certain properties on an empty SmoothedValue raises exceptions.

        For example, global_avg should raise ZeroDivisionError if no values
        have been added. The exception message is also verified.
        """
        sv = SmoothedValue()
        with pytest.raises(expected_exception, match=expected_message):
            getattr(sv, property_name)

    @pytest.mark.parametrize("property_name", ["median", "avg"])
    def test_empty_nan(self, property_name):
        """
        Validate that accessing median or avg on an empty SmoothedValue returns NaN.

        This matches the behavior of torch.median and torch.mean on empty tensors.
        """
        sv = SmoothedValue()
        assert math.isnan(getattr(sv, property_name))

    def test_synchronize_between_processes_no_dist(self):
        """
        Validate synchronize_between_processes when a distributed environment is not initialized.

        It should behave as a no-op and not raise any errors.
        """
        sv = SmoothedValue()
        sv.update(1.0)
        sv.synchronize_between_processes()
        assert sv.count == 1
        assert sv.total == 1.0
