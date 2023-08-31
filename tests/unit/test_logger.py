import logging
import unittest
from unittest.mock import patch

from rich.console import Console

from upp.logger import ProgressBar, setup_logger


class TestScript(unittest.TestCase):

    def test_setup_logger(self):
        logger = setup_logger(level="DEBUG")
        logger.debug("Debug message")

if __name__ == '__main__':
    unittest.main()