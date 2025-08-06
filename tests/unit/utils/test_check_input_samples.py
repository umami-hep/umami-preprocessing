from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import upp.utils.check_input_samples as cis


def test_parse_args_defaults(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("")
    args = ["--config", str(cfg)]
    parsed = cis.parse_args(args)
    # Defaults
    assert isinstance(parsed, Namespace)
    assert parsed.config == cfg
    assert parsed.deviation_factor == 10.0
    assert parsed.verbose is False


class TestCheckWithinFactor:
    def test_ok_small_spread(self):
        groups = {
            "grp": {"a": 100, "b": 120, "c": 110},
        }
        # Should not raise with a relaxed factor
        cis.check_within_factor(groups, factor=2.0)

    def test_zero_raises(self):
        groups = {"grp": {"a": 0, "b": 5}}
        with pytest.raises(ValueError) as exc:
            cis.check_within_factor(groups, factor=10.0)
        assert "zero jets" in str(exc.value)

    def test_spread_raises(self):
        groups = {"grp": {"a": 10, "b": 1000}}
        with pytest.raises(ValueError) as exc:
            cis.check_within_factor(groups, factor=5.0)
        assert "spread" in str(exc.value)


def test_run_input_sample_check_builds_groups_and_reads(monkeypatch, tmp_path):
    # Fake logger with no-ops
    class _Log:
        def info(self, *_a, **_k):
            pass

        def error(self, *_a, **_k):
            pass

    monkeypatch.setattr(cis, "setup_logger", lambda: _Log())

    # Fake H5Reader that returns a configurable num_jets
    class _H5:
        def __init__(self, fname, _batch_size, _jets_name):
            # keep args to inspect if desired
            self.fname = fname
            self.num_jets = 100 if "000001" in str(fname) else 110 if "000002" in str(fname) else 50

    monkeypatch.setattr(cis, "H5Reader", _H5)

    # Build a minimal fake PreprocessingConfig
    class _Cfg:
        def __init__(self, ntuple_dir: Path):
            # config structure with a single block containing patterns
            self.config = {
                "blockA": {
                    "pattern": [
                        "user.sample.000001.r13167.other.h5",
                        "user.sample.000002.r13167.other.h5",
                    ]
                }
            }
            self.ntuple_dir = ntuple_dir
            self.batch_size = 1024
            self.jets_name = "jets"

    cfg = _Cfg(tmp_path)

    # Should not raise with a generous factor
    cis.run_input_sample_check(config=cfg, deviation_factor=5.0, verbose=True)


def test_run_input_sample_check_raises_on_spread(monkeypatch, tmp_path):
    class _Log:
        def info(self, *_a, **_k):
            pass

        def error(self, *_a, **_k):
            pass

    monkeypatch.setattr(cis, "setup_logger", lambda: _Log())

    class _H5:
        def __init__(self, fname, _batch_size, _jets_name):
            # Create a big spread
            self.num_jets = 1 if "low" in str(fname) else 1000

    monkeypatch.setattr(cis, "H5Reader", _H5)

    class _Cfg:
        def __init__(self, ntuple_dir: Path):
            self.config = {
                "blockA": {
                    "pattern": [
                        "user.low.123456.r13167.a.h5",
                        "user.high.234567.r13167.b.h5",
                    ]
                }
            }
            self.ntuple_dir = ntuple_dir
            self.batch_size = 1024
            self.jets_name = "jets"

    cfg = _Cfg(tmp_path)

    with pytest.raises(ValueError):
        cis.run_input_sample_check(config=cfg, deviation_factor=2.0, verbose=False)


def test_run_input_sample_check_accepts_string_pattern(monkeypatch, tmp_path):
    class _Log:
        def info(self, *_a, **_k):
            pass

        def error(self, *_a, **_k):
            pass

    monkeypatch.setattr(cis, "setup_logger", lambda: _Log())

    class _H5:
        def __init__(self, _fname, _batch_size, _jets_name):
            self.num_jets = 42

    monkeypatch.setattr(cis, "H5Reader", _H5)

    class _Cfg:
        def __init__(self, ntuple_dir: Path):
            self.config = {"blockA": {"pattern": "user.sample.000001.r13167.h5"}}
            self.ntuple_dir = ntuple_dir
            self.batch_size = 1024
            self.jets_name = "jets"

    cfg = _Cfg(tmp_path)
    cis.run_input_sample_check(config=cfg, deviation_factor=10.0, verbose=False)
