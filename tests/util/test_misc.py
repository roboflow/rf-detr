import torch
import pytest
import math
from rfdetr.util.misc import SmoothedValue

class TestSmoothedValue:
    @pytest.mark.parametrize("window_size, fmt", [
        (10, "{avg:.2f}"),
        (20, None),
        (5, "{median:.1f} ({global_avg:.1f})"),
    ])
    def test_init(self, window_size, fmt):
        if fmt is None:
            sv = SmoothedValue(window_size=window_size)
            assert sv.fmt == "{median:.4f} ({global_avg:.4f})"
        else:
            sv = SmoothedValue(window_size=window_size, fmt=fmt)
            assert sv.fmt == fmt
        assert sv.deque.maxlen == window_size
        assert sv.total == 0.0
        assert sv.count == 0

    @pytest.mark.parametrize("updates", [
        [(1.0, 1)],
        [(1.0, 1), (2.0, 2)],
        [(1.0, 1), (2.0, 1), (3.0, 1)],
    ])
    def test_update(self, updates):
        sv = SmoothedValue()
        expected_count = 0
        expected_total = 0.0
        for val, n in updates:
            sv.update(val, n=n)
            expected_count += n
            expected_total += val * n
            assert sv.value == val

        assert sv.count == expected_count
        assert sv.total == pytest.approx(expected_total)

    @pytest.mark.parametrize("window_size, updates, expected", [
        (3, [1.0, 3.0, 2.0], {"value": 2.0, "max": 3.0, "avg": 2.0, "global_avg": 2.0, "median": 2.0}),
        (3, [1.0, 3.0, 2.0, 4.0], {"value": 4.0, "max": 4.0, "avg": 3.0, "global_avg": 2.5, "median": 3.0}),
    ])
    def test_properties(self, window_size, updates, expected):
        sv = SmoothedValue(window_size=window_size)
        for val in updates:
            sv.update(val)

        assert sv.value == expected["value"]
        assert sv.max == expected["max"]
        assert sv.avg == pytest.approx(expected["avg"])
        assert sv.global_avg == pytest.approx(expected["global_avg"])
        assert sv.median == pytest.approx(expected["median"])

    def test_str(self):
        sv = SmoothedValue(window_size=3, fmt="{median:.1f} ({global_avg:.1f})")
        sv.update(1.0)
        sv.update(2.0)

        d = torch.tensor([1.0, 2.0])
        expected_median = d.median().item()
        expected_global_avg = 1.5

        assert f"{expected_median:.1f} ({expected_global_avg:.1f})" == str(sv)

    @pytest.mark.parametrize("property_name, expected_exception", [
        ("max", ValueError),
        ("value", IndexError),
        ("global_avg", ZeroDivisionError),
    ])
    def test_empty_exceptions(self, property_name, expected_exception):
        sv = SmoothedValue()
        with pytest.raises(expected_exception):
            getattr(sv, property_name)

    @pytest.mark.parametrize("property_name", ["median", "avg"])
    def test_empty_nan(self, property_name):
        sv = SmoothedValue()
        assert math.isnan(getattr(sv, property_name))

    def test_synchronize_between_processes_no_dist(self):
        # When distributed is not initialized, it should return immediately without error
        sv = SmoothedValue()
        sv.update(1.0)
        sv.synchronize_between_processes()
        assert sv.count == 1
        assert sv.total == 1.0
