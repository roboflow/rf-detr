# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import warnings

import pytest

from rfdetr.utils.decorators import _DeprecatedDict


class TestDeprecatedDict:
    """Test suite for _DeprecatedDict class."""

    def test_init_with_dict(self):
        """Test initialization with a regular dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        deprecated_dict = _DeprecatedDict(
            data,
            deprecated_name="TEST_DICT",
            replacement="`NewAPI`"
        )
        assert len(deprecated_dict) == 2
        assert dict(deprecated_dict) == data

    def test_init_with_kwargs(self):
        """Test initialization with keyword arguments."""
        deprecated_dict = _DeprecatedDict(
            key1="value1",
            key2="value2",
            deprecated_name="TEST_DICT",
            replacement="`NewAPI`"
        )
        assert len(deprecated_dict) == 2
        assert deprecated_dict["key1"] == "value1"
        assert deprecated_dict["key2"] == "value2"

    def test_getitem_emits_warning(self):
        """Test that __getitem__ emits a deprecation warning on first access."""
        deprecated_dict = _DeprecatedDict(
            {"key": "value"},
            deprecated_name="OLD_DICT",
            replacement="`NewDict`"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = deprecated_dict["key"]
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "OLD_DICT is deprecated" in str(w[0].message)
            assert "Use `NewDict` instead" in str(w[0].message)
            assert value == "value"

    def test_get_emits_warning(self):
        """Test that get() method emits a deprecation warning on first access."""
        deprecated_dict = _DeprecatedDict(
            {"key": "value"},
            deprecated_name="OLD_DICT",
            replacement="`NewDict`"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = deprecated_dict.get("key")
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert value == "value"

    def test_get_with_default(self):
        """Test that get() with default value works correctly."""
        deprecated_dict = _DeprecatedDict(
            {"key": "value"},
            deprecated_name="OLD_DICT",
            replacement="`NewDict`"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = deprecated_dict.get("missing_key", "default_value")
            
            assert len(w) == 1
            assert value == "default_value"

    def test_keys_emits_warning(self):
        """Test that keys() method emits a deprecation warning."""
        deprecated_dict = _DeprecatedDict(
            {"key1": "value1", "key2": "value2"},
            deprecated_name="OLD_DICT",
            replacement="`NewDict`"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            keys = list(deprecated_dict.keys())
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert set(keys) == {"key1", "key2"}

    def test_values_emits_warning(self):
        """Test that values() method emits a deprecation warning."""
        deprecated_dict = _DeprecatedDict(
            {"key1": "value1", "key2": "value2"},
            deprecated_name="OLD_DICT",
            replacement="`NewDict`"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            values = list(deprecated_dict.values())
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert set(values) == {"value1", "value2"}

    def test_items_emits_warning(self):
        """Test that items() method emits a deprecation warning."""
        deprecated_dict = _DeprecatedDict(
            {"key1": "value1", "key2": "value2"},
            deprecated_name="OLD_DICT",
            replacement="`NewDict`"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            items = list(deprecated_dict.items())
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert set(items) == {("key1", "value1"), ("key2", "value2")}

    def test_contains_emits_warning(self):
        """Test that __contains__ (in operator) emits a deprecation warning."""
        deprecated_dict = _DeprecatedDict(
            {"key": "value"},
            deprecated_name="OLD_DICT",
            replacement="`NewDict`"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = "key" in deprecated_dict
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert result is True

    def test_iter_emits_warning(self):
        """Test that __iter__ emits a deprecation warning."""
        deprecated_dict = _DeprecatedDict(
            {"key1": "value1", "key2": "value2"},
            deprecated_name="OLD_DICT",
            replacement="`NewDict`"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            keys = [key for key in deprecated_dict]
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert set(keys) == {"key1", "key2"}

    def test_warning_only_shown_once(self):
        """Test that the deprecation warning is only shown once per instance."""
        deprecated_dict = _DeprecatedDict(
            {"key1": "value1", "key2": "value2"},
            deprecated_name="OLD_DICT",
            replacement="`NewDict`"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # First access - should emit warning
            _ = deprecated_dict["key1"]
            assert len(w) == 1
            
            # Second access - should NOT emit warning
            _ = deprecated_dict["key2"]
            assert len(w) == 1  # Still only 1 warning
            
            # Third access via get() - should NOT emit warning
            _ = deprecated_dict.get("key1")
            assert len(w) == 1  # Still only 1 warning
            
            # Fourth access via keys() - should NOT emit warning
            _ = list(deprecated_dict.keys())
            assert len(w) == 1  # Still only 1 warning

    def test_multiple_instances_show_warnings_independently(self):
        """Test that each instance tracks its own warning state."""
        dict1 = _DeprecatedDict(
            {"key": "value1"},
            deprecated_name="DICT1",
            replacement="`NewDict`"
        )
        dict2 = _DeprecatedDict(
            {"key": "value2"},
            deprecated_name="DICT2",
            replacement="`NewDict`"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Access first dict - should emit warning
            _ = dict1["key"]
            assert len(w) == 1
            assert "DICT1" in str(w[0].message)
            
            # Access second dict - should emit its own warning
            _ = dict2["key"]
            assert len(w) == 2
            assert "DICT2" in str(w[1].message)
            
            # Access first dict again - should NOT emit warning
            _ = dict1["key"]
            assert len(w) == 2  # Still only 2 warnings

    def test_default_deprecated_name_and_replacement(self):
        """Test that default deprecated_name and replacement work correctly."""
        deprecated_dict = _DeprecatedDict({"key": "value"})
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = deprecated_dict["key"]
            
            assert len(w) == 1
            assert "this dictionary is deprecated" in str(w[0].message)
            assert "Use the new API instead" in str(w[0].message)

    def test_dictionary_functionality_preserved(self):
        """Test that the dictionary still works as a normal dictionary."""
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        deprecated_dict = _DeprecatedDict(
            data,
            deprecated_name="OLD_DICT",
            replacement="`NewDict`"
        )
        
        # Suppress warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test basic operations
            assert len(deprecated_dict) == 3
            assert deprecated_dict["key1"] == "value1"
            assert deprecated_dict.get("key2") == "value2"
            assert deprecated_dict.get("missing", "default") == "default"
            assert "key1" in deprecated_dict
            assert "missing" not in deprecated_dict
            
            # Test iteration
            assert set(deprecated_dict.keys()) == {"key1", "key2", "key3"}
            assert set(deprecated_dict.values()) == {"value1", "value2", "value3"}
            assert set(deprecated_dict.items()) == {
                ("key1", "value1"),
                ("key2", "value2"),
                ("key3", "value3")
            }

    def test_warning_stacklevel(self):
        """Test that warning stacklevel points to the caller."""
        deprecated_dict = _DeprecatedDict(
            {"key": "value"},
            deprecated_name="OLD_DICT",
            replacement="`NewDict`"
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This is the line that should be reported in the warning
            value = deprecated_dict["key"]  # noqa: F841
            
            assert len(w) == 1
            # The warning should point to this test function, not to _DeprecatedDict
            assert "test_decorators.py" in w[0].filename
