from __future__ import annotations

import logging
import sys
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

# Detect if the program is executed in an interactive terminal
_IS_TTY = sys.stderr.isatty()

# One console object is reused everywhere so that Rich keeps a consistent idea
# of whether it may emit ANSI control codes / animations.
_console = Console(
    width=100,
    force_terminal=_IS_TTY,
    force_interactive=_IS_TTY,
    no_color=not _IS_TTY,
)

# Template for the progress bar
ProgressBar = partial(
    Progress,
    TextColumn("[task.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("•"),
    TimeRemainingColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    refresh_per_second=1 if _IS_TTY else 0.05,
    speed_estimate_period=30 if _IS_TTY else 120,
    console=_console,
    disable=not _IS_TTY,
    transient=_IS_TTY,
)


# Helper for setup the logger
def setup_logger(level: str = "INFO"):
    """Set up the logger.

    Configure Rich logging so that colourful / interactive output is used when
    the program is attached to a terminal and plain text is written when it is
    executed under a batch system such as Slurm (where stdout / stderr are files).
    """
    FORMAT = "%(message)s"

    # In a batch job we create a console that never emits colour codes.
    console = None
    if not _IS_TTY:
        console = Console(
            width=120,
            force_terminal=False,
            force_interactive=False,
            no_color=True,
        )

    handler = RichHandler(
        show_time=False,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        console=console,
    )

    logging.basicConfig(level=level, format=FORMAT, handlers=[handler])
    return logging
