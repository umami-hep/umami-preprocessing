from __future__ import annotations

import argparse
from argparse import Namespace
from unittest.mock import MagicMock, patch

from pytest import fixture

from upp.main import main, parse_args, run_pp


@fixture
def config_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    return config_file


def test_parse_args_with_config(config_file):
    args = [
        "--config",
        str(config_file),
        "--resample",
        "--split",
        "val",
    ]
    parsed_args = parse_args(args)
    expected_args = Namespace(
        config=config_file,
        metadata=False,
        prep=None,
        resample=True,
        merge=None,
        norm=None,
        plot=None,
        split="val",
        component=None,
        region=None,
        container=None,
        grid=False,
        split_components=False,
        reweight=False,
        rw_merge=False,
        rw_merge_idx=None,
        files=None,
        skip_sample_check=False,
    )

    assert parsed_args == expected_args


def test_parse_args_flags_not_given(config_file):
    args = ["--config", str(config_file)]
    parsed_args = parse_args(args)
    expected_args = Namespace(
        config=config_file,
        metadata=False,
        prep=True,
        resample=True,
        merge=True,
        norm=True,
        plot=True,
        split="train",
        component=None,
        region=None,
        container=None,
        grid=False,
        split_components=False,
        reweight=False,
        rw_merge=False,
        rw_merge_idx=None,
        files=None,
        skip_sample_check=False,
    )
    assert parsed_args == expected_args


def test_parse_args_flags_negative(config_file):
    args = [
        "--config",
        str(config_file),
        "--no-prep",
        "--no-resample",
        "--no-merge",
        "--no-norm",
        "--no-plot",
    ]

    # Call the parse_args function
    parsed_args = parse_args(args)

    # Check if the parsed arguments match the expected values
    expected_args = Namespace(
        config=config_file,
        metadata=False,
        prep=False,
        resample=False,
        merge=False,
        norm=False,
        plot=False,
        split="train",
        component=None,
        region=None,
        container=None,
        grid=False,
        split_components=False,
        reweight=False,
        rw_merge=False,
        rw_merge_idx=None,
        files=None,
        skip_sample_check=False,
    )

    assert parsed_args == expected_args


def test_parse_args_flags_positive(config_file):
    args = [
        "--config",
        str(config_file),
        "--prep",
        "--resample",
        "--merge",
        "--norm",
        "--plot",
    ]

    parsed_args = parse_args(args)
    expected_args = Namespace(
        config=config_file,
        metadata=False,
        prep=True,
        resample=True,
        merge=True,
        norm=True,
        plot=True,
        split="train",
        component=None,
        region=None,
        container=None,
        grid=False,
        split_components=False,
        reweight=False,
        rw_merge=False,
        rw_merge_idx=None,
        files=None,
        skip_sample_check=False,
    )

    assert parsed_args == expected_args


def test_parse_args_component(config_file):
    args = [
        "--config",
        str(config_file),
        "--prep",
        "--resample",
        "--merge",
        "--norm",
        "--plot",
        "--component",
        "lowpt_ttbar_ujets",
    ]

    parsed_args = parse_args(args)
    expected_args = Namespace(
        config=config_file,
        metadata=False,
        prep=True,
        resample=True,
        merge=True,
        norm=True,
        plot=True,
        split="train",
        component="lowpt_ttbar_ujets",
        region=None,
        container=None,
        grid=False,
        split_components=False,
        reweight=False,
        rw_merge=False,
        rw_merge_idx=None,
        files=None,
        skip_sample_check=False,
    )

    assert parsed_args == expected_args


def test_parse_args_region(config_file):
    args = [
        "--config",
        str(config_file),
        "--prep",
        "--resample",
        "--merge",
        "--norm",
        "--plot",
        "--region",
        "lowpt",
    ]

    parsed_args = parse_args(args)
    expected_args = Namespace(
        config=config_file,
        metadata=False,
        prep=True,
        resample=True,
        merge=True,
        norm=True,
        plot=True,
        split="train",
        component=None,
        region="lowpt",
        container=None,
        grid=False,
        split_components=False,
        reweight=False,
        rw_merge=False,
        rw_merge_idx=None,
        files=None,
        skip_sample_check=False,
    )

    assert parsed_args == expected_args


def test_parse_args_metadata_flag(config_file):
    args = ["--config", str(config_file), "--metadata"]
    parsed_args = parse_args(args)
    assert parsed_args.metadata is True


def _base_args(config_file: object, **overrides: object) -> argparse.Namespace:
    defaults: dict = dict(
        config=config_file,
        metadata=False,
        prep=False,
        resample=False,
        merge=False,
        norm=False,
        plot=False,
        split="train",
        component=None,
        region=None,
        container=None,
        grid=False,
        split_components=False,
        reweight=False,
        rw_merge=False,
        rw_merge_idx=None,
        files=None,
        skip_sample_check=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_pp_metadata_injection(tmp_path):
    """run_pp with metadata=True constructs and runs MetadataInjector (lines 222-224)."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    args = _base_args(config_file, metadata=True)

    mock_injector = MagicMock()
    with (
        patch("upp.main.PreprocessingConfig.from_file", return_value=MagicMock()),
        patch("upp.main.MetadataInjector", return_value=mock_injector),
    ):
        run_pp(args)

    mock_injector.run.assert_called_once()


def test_run_pp_rw_merge_with_idx(tmp_path):
    """run_pp with rw_merge_idx parses the comma-separated pair (lines 269-272)."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    args = _base_args(config_file, rw_merge=True, rw_merge_idx="0,1")

    mock_rw = MagicMock()
    with (
        patch("upp.main.PreprocessingConfig.from_file", return_value=MagicMock()),
        patch("upp.main.RWMerge", return_value=mock_rw),
    ):
        run_pp(args)

    mock_rw.run.assert_called_once()


def test_main_split_all(tmp_path):
    """main() with split='all' calls run_pp three times (lines 303-310)."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    mock_args = _base_args(config_file, split="all")

    with (
        patch("upp.main.parse_args", return_value=mock_args),
        patch("upp.main.run_pp") as mock_run_pp,
    ):
        main()

    assert mock_run_pp.call_count == 3
