from __future__ import annotations

import argparse
from pathlib import Path
import pytest

import upp.utils.estimate_jets as ej


def test_parse_args_defaults(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("")
    args = ["--config", str(cfg)]
    parsed = ej.parse_args(args)
    assert parsed.config == cfg
    assert parsed.output == "max_num_jets.log"
    assert parsed.no_prep is False


def test_parse_args_custom(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("")
    args = ["--config", str(cfg), "--output", "custom.log", "--no-prep"]
    parsed = ej.parse_args(args)
    assert parsed.config == cfg
    assert parsed.output == "custom.log"
    assert parsed.no_prep is True


def test_main_calls_run_estimate_jets(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("")

    called_args = []

    def mock_run_estimate_jets(config, output_path, prep):
        called_args.append((config, output_path, prep))

    monkeypatch.setattr(ej, "run_estimate_jets", mock_run_estimate_jets)

    ej.main(["--config", str(cfg), "--output", "test_out.log"])

    assert len(called_args) == 1
    assert called_args[0] == (cfg, "test_out.log", True)
