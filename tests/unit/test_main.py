from __future__ import annotations

from argparse import Namespace

from pytest import fixture

from upp.main import parse_args


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
