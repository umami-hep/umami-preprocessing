from __future__ import annotations


def path_append(path, suffix):
    return path.parent / f"{path.stem}_{suffix}{path.suffix}"
