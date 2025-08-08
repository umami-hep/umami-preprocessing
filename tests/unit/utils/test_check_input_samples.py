# === add to tests/unit/utils/test_check_input_samples.py ===
from __future__ import annotations

import argparse
import runpy
import sys
import types
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


def test_builds_entry_name_with_dsid_and_rtag(monkeypatch, tmp_path):
    """Covers line 198: entry_name = f"{dsid} / {rtag_to_campaign_dict.get(rtag)}"."""
    import upp.utils.check_input_samples as cis

    class _Log:
        def info(self, *_a, **_k):
            pass

        def error(self, *_a, **_k):
            pass

    monkeypatch.setattr(cis, "setup_logger", lambda: _Log())

    # H5Reader stub
    class _H5:
        def __init__(self, **_kwargs):
            self.num_jets = 10

    monkeypatch.setattr(cis, "H5Reader", _H5)

    # Spy to capture the fully built groups dict passed to the checker
    captured = {}

    def _spy_check_within_factor(*, groups, **_kwargs):
        captured.update(groups)

    monkeypatch.setattr(cis, "check_within_factor", _spy_check_within_factor)

    # Pattern has DSID and rtag -> should map r13167 -> MC20a
    class _Cfg:
        def __init__(self, ntuple_dir: Path):
            self.config = {"block": {"pattern": ["x.123456.y_r13167.z.h5"]}}
            self.ntuple_dir = ntuple_dir
            self.batch_size = 1
            self.jets_name = "jets"

    cfg = _Cfg(tmp_path)
    cis.run_input_sample_check(config=cfg, deviation_factor=10.0, verbose=False)

    # Assert the constructed key used the r-tag mapping
    keys = list(captured["block"].keys())
    assert len(keys) == 1
    assert keys[0].startswith("123456 / MC20a")


def test_builds_entry_name_with_only_rtag(monkeypatch, tmp_path):
    """Covers line 201: entry_name = f"{sample} / {rtag_to_campaign_dict.get(rtag)}"."""
    import upp.utils.check_input_samples as cis

    class _Log:
        def info(self, *_a, **_k):
            pass

        def error(self, *_a, **_k):
            pass

    monkeypatch.setattr(cis, "setup_logger", lambda: _Log())

    class _H5:
        def __init__(self, **_kwargs):
            self.num_jets = 10

    monkeypatch.setattr(cis, "H5Reader", _H5)

    captured = {}

    def _spy_check_within_factor(*, groups, **_kwargs):
        captured.update(groups)

    monkeypatch.setattr(cis, "check_within_factor", _spy_check_within_factor)

    # Pattern has rtag but no 6-digit DSID between dots
    sample = "user.sample.no_dsid_r13167.h5"

    class _Cfg:
        def __init__(self, ntuple_dir: Path):
            self.config = {"block": {"pattern": [sample]}}
            self.ntuple_dir = ntuple_dir
            self.batch_size = 1
            self.jets_name = "jets"

    cfg = _Cfg(tmp_path)
    cis.run_input_sample_check(config=cfg, deviation_factor=10.0, verbose=False)

    keys = list(captured["block"].keys())
    assert len(keys) == 1
    # Key should start with the original sample name + mapped campaign
    assert keys[0].startswith(f"{sample} / MC20a")


def test_script_entry_point_executes_main(tmp_path):
    """Executes the module as a script to cover the `if __name__ == '__main__': main()` line."""
    # Build fake external modules the script imports
    fake_cli_utils = types.ModuleType("ftag.cli_utils")
    fake_cli_utils.valid_path = lambda p: Path(p)
    fake_cli_utils.HelpFormatter = argparse.HelpFormatter

    fake_hdf5 = types.ModuleType("ftag.hdf5")

    class _H5:
        def __init__(self, **_kwargs):
            self.num_jets = 1

    fake_hdf5.H5Reader = _H5

    fake_ppcfg = types.ModuleType("upp.classes.preprocessing_config")

    class _Cfg:
        def __init__(self):
            self.config = {}
            self.ntuple_dir = tmp_path
            self.batch_size = 1
            self.jets_name = "jets"

        @classmethod
        def from_file(cls, *_a, **_k):  # signature compatible
            return _Cfg()

    fake_ppcfg.PreprocessingConfig = _Cfg

    fake_logger = types.ModuleType("upp.utils.logger")

    class _Log:
        def info(self, *_a, **_k):
            pass

        def error(self, *_a, **_k):
            pass

    fake_logger.setup_logger = lambda: _Log()

    # Inject fakes into sys.modules so the script can import them
    sys.modules["ftag"] = types.ModuleType("ftag")  # namespace package placeholder
    sys.modules["ftag.cli_utils"] = fake_cli_utils
    sys.modules["ftag.hdf5"] = fake_hdf5
    sys.modules["upp"] = types.ModuleType("upp")
    sys.modules["upp.classes"] = types.ModuleType("upp.classes")
    sys.modules["upp.classes.preprocessing_config"] = fake_ppcfg
    sys.modules["upp.utils"] = types.ModuleType("upp.utils")
    sys.modules["upp.utils.logger"] = fake_logger

    # Provide a fake parse_args inside the script execution context
    # We'll override it via init_globals after the file loads.
    # Easiest is to point runpy at the actual file and intercept parse_args after import.
    file_path = Path(cis.__file__)  # path to the current module file
    # Run the file as __main__ so the guard executes
    # The module defines parse_args itself, so we also seed argv via sys.argv
    old_argv = sys.argv[:]
    try:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text("")  # valid file for valid_path
        sys.argv = [str(file_path), "--config_path", str(cfg_path)]
        runpy.run_path(str(file_path), run_name="__main__")
    finally:
        sys.argv = old_argv
