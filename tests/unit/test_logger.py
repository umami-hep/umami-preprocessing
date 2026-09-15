from __future__ import annotations

import logging
import re

import pytest

from upp.utils import logger as upp_logger
from upp.utils.logger import LOG_LEVEL_ENV, banner, resolve_log_level, setup_logger


@pytest.fixture(autouse=True)
def reset_logger(monkeypatch):
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    monkeypatch.setattr(upp_logger, "_configured_level", None)


def test_setup_logger(capsys):
    logger = setup_logger(level="DEBUG")
    logger.debug("Debug message")
    assert "Debug message" in capsys.readouterr().out


def test_resolve_log_level_default():
    assert resolve_log_level() == "INFO"


def test_resolve_log_level_from_env(monkeypatch):
    monkeypatch.setenv(LOG_LEVEL_ENV, "debug")
    assert resolve_log_level() == "DEBUG"


def test_resolve_log_level_argument_wins(monkeypatch):
    monkeypatch.setenv(LOG_LEVEL_ENV, "DEBUG")
    assert resolve_log_level("WARNING") == "WARNING"


def test_resolve_log_level_invalid(monkeypatch):
    monkeypatch.setenv(LOG_LEVEL_ENV, "LOUD")
    with pytest.raises(ValueError, match="Invalid log level LOUD"):
        resolve_log_level()


def test_setup_logger_uses_env(monkeypatch):
    monkeypatch.setenv(LOG_LEVEL_ENV, "DEBUG")
    setup_logger()
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logger_keeps_first_level(monkeypatch):
    setup_logger(level="DEBUG")
    monkeypatch.setenv(LOG_LEVEL_ENV, "ERROR")
    setup_logger()
    assert logging.getLogger().level == logging.DEBUG


def test_records_are_split_between_stdout_and_stderr(capsys):
    setup_logger(level="DEBUG")
    logging.debug("a debug record")
    logging.info("an info record")
    logging.warning("a warning record")
    logging.error("an error record")

    captured = capsys.readouterr()
    assert "a debug record" in captured.out
    assert "an info record" in captured.out
    assert "a warning record" not in captured.out
    assert "a warning record" in captured.err
    assert "an error record" in captured.err
    assert "an info record" not in captured.err


def test_records_have_a_timestamp(capsys):
    setup_logger(level="INFO")
    logging.info("a timestamped record")
    captured = capsys.readouterr()
    assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", captured.out)


def test_banner_fits_the_console_width(capsys):
    setup_logger(level="INFO")
    logging.info(banner(" Title "))
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 1
    assert " Title " in lines[0]
