# === add to tests/unit/utils/test_check_input_samples.py ===
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import upp.utils.check_input_samples as cis


def test_parse_args_flags_nondefault(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("")
    args = ["--config_path", str(cfg), "--deviation-factor", "7.5", "-v"]
    parsed = cis.parse_args(args)
    assert parsed.config_path == cfg
    assert parsed.deviation_factor == 7.5
    assert parsed.verbose is True


def test_run_input_sample_check_handles_missing_ids_and_rtags(monkeypatch, tmp_path):
    # Logger stub to exercise error branches without blowing up
    class _Log:
        def info(self, *_a, **_k):
            pass

        def error(self, *_a, **_k):
            pass

    monkeypatch.setattr(cis, "setup_logger", lambda: _Log())

    # H5Reader stub: we don't care about values here, just that it's called
    class _H5:
        def __init__(self, **_kwargs):
            self.num_jets = 123

    monkeypatch.setattr(cis, "H5Reader", _H5)

    # Patterns that trigger each name-building branch:
    # - missing DSID (no 6-digit number between dots)
    missing_dsid = "user.sample.notdsid.r13167.h5"
    # - missing rtag (no _rNNNNN)
    missing_rtag = "user.sample.123456.no_rtag.h5"
    # - missing both
    missing_both = "justname.h5"

    class _Cfg:
        def __init__(self, ntuple_dir: Path):
            self.config = {"blockX": {"pattern": [missing_dsid, missing_rtag, missing_both]}}
            self.ntuple_dir = ntuple_dir
            self.batch_size = 1
            self.jets_name = "jets"

    cfg = _Cfg(tmp_path)
    # Should run to completion (we only log errors for missing ids/rtags)
    cis.run_input_sample_check(config=cfg, deviation_factor=10.0, verbose=False)


def test_run_input_sample_check_unsupported_pattern_type_is_logged_and_skipped(
    monkeypatch, tmp_path
):
    class _Log:
        def __init__(self):
            self._errors = 0

        def info(self, *_a, **_k):
            pass

        def error(self, *_a, **_k):
            self._errors += 1

    log = _Log()
    monkeypatch.setattr(cis, "setup_logger", lambda: log)

    # H5Reader never called since pattern is invalid type
    class _H5:
        def __init__(self, **_kwargs):
            raise AssertionError("H5Reader should not be instantiated for invalid pattern type")

    monkeypatch.setattr(cis, "H5Reader", _H5)

    class _Cfg:
        def __init__(self, ntuple_dir: Path):
            # Intentionally wrong type: should be str or list[str]
            self.config = {"blockY": {"pattern": {"oops": "dict-not-supported"}}}
            self.ntuple_dir = ntuple_dir
            self.batch_size = 1
            self.jets_name = "jets"

    cfg = _Cfg(tmp_path)
    cis.run_input_sample_check(config=cfg, deviation_factor=10.0, verbose=False)
    # At least one error was logged
    assert log._errors >= 1


def test_main_calls_pipeline_with_parsed_args(monkeypatch, tmp_path):
    # Build fake args returned by parse_args
    ns = Namespace(config_path=tmp_path / "cfg.yaml", deviation_factor=3.0, verbose=True)
    ns.config_path.write_text("")  # so valid path conversion is happy if reached

    # Monkeypatch parse_args to return our namespace regardless of input
    monkeypatch.setattr(cis, "parse_args", lambda _args=None: ns)

    # Monkeypatch PreprocessingConfig.from_file to avoid reading anything real
    class _Cfg:
        def __init__(self):  # minimal attributes used by run_input_sample_check stub
            self.config = {}
            self.ntuple_dir = tmp_path
            self.batch_size = 1
            self.jets_name = "jets"

    called = {"run": False, "cfg": None, "df": None, "v": None}

    def _fake_from_file(*_args, **_kwargs):
        return _Cfg()

    monkeypatch.setattr(
        cis.PreprocessingConfig,
        "from_file",
        classmethod(_fake_from_file),
    )

    # Monkeypatch the checker to record the call (and cover the main wrapper)
    def _spy_run_input_sample_check(*, config, deviation_factor, verbose):
        called["run"] = True
        called["cfg"] = config
        called["df"] = deviation_factor
        called["v"] = verbose

    monkeypatch.setattr(cis, "run_input_sample_check", _spy_run_input_sample_check)

    # Execute main()
    cis.main([])

    assert called["run"] is True
    assert isinstance(called["cfg"], _Cfg)
    assert called["df"] == 3.0
    assert called["v"] is True
