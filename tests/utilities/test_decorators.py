# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import warnings

import pytest

from rfdetr.utilities.decorators import _DeprecatedDict


class TestDeprecatedDict:
    """Test suite for _DeprecatedDict class."""

    @pytest.mark.parametrize(
        "init_args,init_kwargs,expected_data",
        [
            pytest.param(
                ({"key1": "value1", "key2": "value2"},),
                {"deprecated_name": "TEST_DICT", "replacement": "`NewAPI`"},
                {"key1": "value1", "key2": "value2"},
                id="dict_init",
            ),
            pytest.param(
                (),
                {
                    "key1": "value1",
                    "key2": "value2",
                    "deprecated_name": "TEST_DICT",
                    "replacement": "`NewAPI`",
                },
                {"key1": "value1", "key2": "value2"},
                id="kwargs_init",
            ),
        ],
    )
    def test_initialization(self, init_args, init_kwargs, expected_data):
        """Test initialization with different argument patterns."""
        deprecated_dict = _DeprecatedDict(*init_args, **init_kwargs)
        # Suppress warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert len(deprecated_dict) == len(expected_data)
            assert dict(deprecated_dict) == expected_data

    def test_warning_emitted_on_access(self):
        """Test that deprecation warning is emitted on first access."""
        deprecated_dict = _DeprecatedDict(
            {"key": "value"},
            deprecated_name="OLD_DICT",
            replacement="`NewDict`",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = deprecated_dict["key"]

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "OLD_DICT is deprecated" in str(w[0].message)
            assert "Use `NewDict` instead" in str(w[0].message)
            assert value == "value"

    def test_warning_only_shown_once(self):
        """Test that warning is only shown once per instance."""
        deprecated_dict = _DeprecatedDict(
            {"key1": "value1", "key2": "value2"},
            deprecated_name="OLD_DICT",
            replacement="`NewDict`",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _ = deprecated_dict["key1"]
            assert len(w) == 1

            # Subsequent accesses should not emit additional warnings
            _ = deprecated_dict.get("key2")
            _ = list(deprecated_dict.keys())
            assert len(w) == 1

    @pytest.mark.parametrize(
        "access_method,args,expected_result",
        [
            pytest.param(lambda d: d["key1"], (), "value1", id="getitem"),
            pytest.param(lambda d: d.get("key1"), (), "value1", id="get"),
            pytest.param(lambda d: d.get("missing", "default"), (), "default", id="get_default"),
            pytest.param(lambda d: "key1" in d, (), True, id="contains"),
            pytest.param(lambda d: set(d.keys()), (), {"key1", "key2"}, id="keys"),
            pytest.param(lambda d: set(d.values()), (), {"value1", "value2"}, id="values"),
            pytest.param(
                lambda d: set(d.items()),
                (),
                {("key1", "value1"), ("key2", "value2")},
                id="items",
            ),
            pytest.param(lambda d: [k for k in d], (), ["key1", "key2"], id="iter"),
        ],
    )
    def test_dictionary_functionality(self, access_method, args, expected_result):
        """Test that dictionary operations work correctly."""
        data = {"key1": "value1", "key2": "value2"}
        deprecated_dict = _DeprecatedDict(data, deprecated_name="TEST_DICT", replacement="`NewAPI`")

        # Suppress warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = access_method(deprecated_dict)
            if isinstance(expected_result, list):
                assert result == expected_result or set(result) == set(expected_result)
            else:
                assert result == expected_result
