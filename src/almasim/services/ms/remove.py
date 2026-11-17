"""Removing a MeasurementSet that may be a link to where it really is."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def remove_measurement_set(path: str | os.PathLike[str]) -> None:
    """Remove the MS at ``path``: the folder, or a link and the folder it points to.

    Redoing a stage over an MS that a link stands for replaces the MS itself,
    not only the link.
    """
    path = Path(path)
    if path.is_symlink():
        target = path.resolve()
        path.unlink()
        if target.is_dir():
            shutil.rmtree(target)
    elif path.exists():
        shutil.rmtree(path)
