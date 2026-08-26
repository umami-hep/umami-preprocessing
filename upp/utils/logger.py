from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from functools import partial
from types import ModuleType

from ftag.utils.logging import get_log_level
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Name of the environment variable used to set the log level
LOG_LEVEL_ENV = "UPP_LOG_LEVEL"

# Log level which is used when neither the command line nor the environment define one
DEFAULT_LOG_LEVEL = "INFO"

# Timestamp which is prepended to every log message
LOG_TIME_FORMAT = "[%Y-%m-%d %H:%M:%S]"

# Width of the time and level columns that Rich renders in front of every message,
# each including their trailing separator
_TIME_COL_WIDTH = len(datetime(2000, 1, 1).strftime(LOG_TIME_FORMAT)) + 1
_LEVEL_COL_WIDTH = 9

# Detect if the program is executed in an interactive terminal
_IS_TTY = sys.stderr.isatty()

# Rich measures the terminal itself when no width is given. Under a batch system there is
# no terminal to measure, so fall back to a fixed width which COLUMNS can override.
_WIDTH = None if _IS_TTY else int(os.environ.get("COLUMNS", 120))

# One console object is reused everywhere so that Rich keeps a consistent idea
# of whether it may emit ANSI control codes / animations. It is shared by the progress
# bar and the log records written to stdout (the .out file of a batch job), so that log
# messages are rendered above a running progress bar instead of on top of it.
_console = Console(
    width=_WIDTH,
    force_terminal=_IS_TTY,
    force_interactive=_IS_TTY,
    no_color=not _IS_TTY,
)

# Console used for the log records which are written to stderr (the .err file of a batch job)
_stderr_console = Console(
    stderr=True,
    width=_WIDTH,
    force_terminal=_IS_TTY,
    force_interactive=_IS_TTY,
    no_color=not _IS_TTY,
)

# Level which was used in the last call of setup_logger()
_configured_level: str | None = None

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


def banner(title: str = "", style: str = "bold green") -> str:
    """Build a horizontal rule with the title centred in the available log width.

    Parameters
    ----------
    title : str, optional
        Title to centre in the rule, by default ""
    style : str, optional
        Rich markup style applied to the rule, by default "bold green"

    Returns
    -------
    str
        Rule with Rich markup, ready to be passed to the logger
    """
    width = max(_console.width - _TIME_COL_WIDTH - _LEVEL_COL_WIDTH, 20)
    return f"[{style}]{title:-^{width}}"


def resolve_log_level(level: str | None = None) -> str:
    """Resolve the log level from the command line, the environment or the default.

    Parameters
    ----------
    level : str | None, optional
        Log level given on the command line, which wins over the environment
        variable, by default None

    Returns
    -------
    str
        Name of the resolved log level

    Raises
    ------
    ValueError
        If the resolved level is not a valid log level
    """
    resolved = (level or os.environ.get(LOG_LEVEL_ENV) or DEFAULT_LOG_LEVEL).upper()
    try:
        get_log_level(resolved)
    except ValueError as error:
        raise ValueError(f"Invalid log level {resolved}") from error
    return resolved


def _make_handler(console: Console) -> RichHandler:
    """Create a Rich log handler which writes to the given console.

    Parameters
    ----------
    console : Console
        Console the handler writes to

    Returns
    -------
    RichHandler
        Handler with the timestamp column enabled
    """
    return RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        log_time_format=LOG_TIME_FORMAT,
        omit_repeated_times=False,
    )


# Helper for setup the logger
def setup_logger(level: str | None = None) -> ModuleType:
    """Set up the logger.

    Configure Rich logging so that colourful / interactive output is used when
    the program is attached to a terminal and plain text is written when it is
    executed under a batch system such as Slurm (where stdout / stderr are files).
    Debug and info records are written to stdout, warnings and above to stderr.

    Parameters
    ----------
    level : str | None, optional
        Log level to use. If None, the level is taken from the UPP_LOG_LEVEL
        environment variable and falls back to INFO. By default None

    Returns
    -------
    ModuleType
        The logging module, already configured
    """
    global _configured_level

    # Without an explicit level, keep the configuration of the first call
    if level is None and _configured_level is not None:
        return logging

    FORMAT = "%(message)s"
    resolved = resolve_log_level(level)

    stdout_handler = _make_handler(_console)
    stdout_handler.addFilter(lambda record: record.levelno < logging.WARNING)

    stderr_handler = _make_handler(_stderr_console)
    stderr_handler.setLevel(logging.WARNING)

    logging.basicConfig(
        level=get_log_level(resolved),
        format=FORMAT,
        handlers=[stdout_handler, stderr_handler],
        force=True,
    )
    _configured_level = resolved
    return logging
