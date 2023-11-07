from __future__ import annotations

import logging

from upp.logger import setup_logger


def test_setup_logger(caplog):
    caplog.set_level(logging.DEBUG)
    logger = setup_logger(level="DEBUG")
    logger.debug("Debug message")
    assert "Debug message" in caplog.text
