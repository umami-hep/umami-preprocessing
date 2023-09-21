from __future__ import annotations

from pathlib import Path

from upp.utils import path_append


def test_path_append():
    input_path = Path("file.txt")
    suffix = "new"
    expected_output = Path("file_new.txt")
    result = path_append(input_path, suffix)
    assert result == expected_output
