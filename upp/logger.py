from __future__ import annotations

import logging
from functools import partial

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

is_terminal = Console().is_terminal
ProgressBar = partial(
    Progress,
    TextColumn("[task.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("•"),
    TimeRemainingColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    refresh_per_second=1 if is_terminal else 0.05,
    speed_estimate_period=30 if is_terminal else 120,
    console=Console(width=100, force_terminal=True),
)


def setup_logger(level="INFO"):
    FORMAT = "%(message)s"
    console = None if is_terminal else Console(width=120)
    handler = RichHandler(
        show_time=False, show_path=False, markup=True, rich_tracebacks=True, console=console
    )
    logging.basicConfig(level=level, format=FORMAT, datefmt="[%X]", handlers=[handler])
    return logging
