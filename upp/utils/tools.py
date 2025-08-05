from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


def path_append(path: Path, suffix: str):
    return path.parent / f"{path.stem}_{suffix}{path.suffix}"
