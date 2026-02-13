# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import os
import tempfile
import pytest

from rfdetr.util.files import _compute_file_md5, _validate_file_md5


class TestFileMD5Validation:
    """Test MD5 hash computation and validation."""

    def test__compute_file_md5(self):
        """Test MD5 hash computation for a simple file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Hello, World!")
            temp_file = f.name

        try:
            # Known MD5 hash for "Hello, World!"
            expected_hash = "65a8e27d8879283831b664bd8b7f0ad4"
            actual_hash = _compute_file_md5(temp_file)
            assert actual_hash == expected_hash
        finally:
            os.unlink(temp_file)

    def test__validate_file_md5_success(self):
        """Test successful MD5 validation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content")
            temp_file = f.name

        try:
            # Compute the actual hash first
            expected_hash = _compute_file_md5(temp_file)

            # Validation should succeed
            assert _validate_file_md5(temp_file, expected_hash) is True
        finally:
            os.unlink(temp_file)

    def test__validate_file_md5_failure(self):
        """Test MD5 validation failure with wrong hash."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content")
            temp_file = f.name

        try:
            # Use a wrong hash
            wrong_hash = "0" * 32

            # Validation should fail
            assert _validate_file_md5(temp_file, wrong_hash) is False
        finally:
            os.unlink(temp_file)

    def test__validate_file_md5_case_insensitive(self):
        """Test that MD5 validation is case-insensitive."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content")
            temp_file = f.name

        try:
            # Get hash in lowercase
            hash_lower = _compute_file_md5(temp_file)
            hash_upper = hash_lower.upper()

            # Both should validate successfully
            assert _validate_file_md5(temp_file, hash_lower) is True
            assert _validate_file_md5(temp_file, hash_upper) is True
        finally:
            os.unlink(temp_file)

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        nonexistent_file = "/tmp/nonexistent_file_xyz.txt"
        assert _validate_file_md5(nonexistent_file, "abc123") is False

    def test__compute_file_md5_empty_file(self):
        """Test MD5 hash computation for empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            # Create empty file
            temp_file = f.name

        try:
            # Known MD5 hash for empty file
            expected_hash = "d41d8cd98f00b204e9800998ecf8427e"
            actual_hash = _compute_file_md5(temp_file)
            assert actual_hash == expected_hash
        finally:
            os.unlink(temp_file)

    def test__compute_file_md5_large_file(self):
        """Test MD5 computation for larger file (tests chunking)."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            # Write 1MB of data
            data = b'A' * (1024 * 1024)
            f.write(data)
            temp_file = f.name

        try:
            # Compute hash (should handle chunking correctly)
            hash_value = _compute_file_md5(temp_file)

            # Verify it's a valid MD5 hash format
            assert len(hash_value) == 32
            assert all(c in '0123456789abcdef' for c in hash_value)
        finally:
            os.unlink(temp_file)
