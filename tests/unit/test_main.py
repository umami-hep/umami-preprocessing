from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from upp.main import parse_args


def test_parse_args_with_config(mocker):
    mocker.patch(
        "sys.argv",
        [
            "preprocess",
            "--config",
            "config_file.yaml",
            "--resample",
            "--split",
            "val",
        ],
    )

    # Call the parse_args function
    parsed_args = parse_args()

    # Check if the parsed arguments match the expected values
    expected_args = Namespace(
        config=Path("config_file.yaml"),
        prep=None,
        resample=True,
        merge=None,
        norm=None,
        plot=None,
        split="val",
    )

    assert parsed_args == expected_args


def test_parse_args_flags_not_given(mocker):
    mocker.patch("sys.argv", ["preprocess", "--config", "config_file.yaml"])

    # Call the parse_args function
    parsed_args = parse_args()

    # Check if the parsed arguments match the expected values
    expected_args = Namespace(
        config=Path("config_file.yaml"),
        prep=True,
        resample=True,
        merge=True,
        norm=True,
        plot=True,
        split="train",
    )

    assert parsed_args == expected_args


def test_parse_args_flags_negative(mocker):
    mocker.patch(
        "sys.argv",
        [
            "preprocess",
            "--config",
            "config_file.yaml",
            "--no-prep",
            "--no-resample",
            "--no-merge",
            "--no-norm",
            "--no-plot",
        ],
    )

    # Call the parse_args function
    parsed_args = parse_args()

    # Check if the parsed arguments match the expected values
    expected_args = Namespace(
        config=Path("config_file.yaml"),
        prep=False,
        resample=False,
        merge=False,
        norm=False,
        plot=False,
        split="train",
    )

    assert parsed_args == expected_args


def test_parse_args_flags_positive(mocker):
    mocker.patch(
        "sys.argv",
        [
            "preprocess",
            "--config",
            "config_file.yaml",
            "--prep",
            "--resample",
            "--merge",
            "--norm",
            "--plot",
        ],
    )

    # Call the parse_args function
    parsed_args = parse_args()

    # Check if the parsed arguments match the expected values
    expected_args = Namespace(
        config=Path("config_file.yaml"),
        prep=True,
        resample=True,
        merge=True,
        norm=True,
        plot=True,
        split="train",
    )

    assert parsed_args == expected_args
