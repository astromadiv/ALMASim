"""Remove intermediate calibration data after a full success (MADIV-28 T4.8)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from ...utils import LogFn, report

logger = logging.getLogger(__name__)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _remove_tree_after_success(
    path: str | os.PathLike[str], protected: list[Path], logger_fn: LogFn = None
) -> None:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return
    for protected_path in protected:
        protected_path = protected_path.resolve()
        if target == protected_path or _is_relative_to(protected_path, target):
            report(
                logger,
                logger_fn,
                f"Skipping cleanup of {target}; it contains protected output {protected_path}",
                logging.WARNING,
            )
            return
    shutil.rmtree(target)
    report(logger, logger_fn, f"Removed intermediate data: {target}")


def cleanup_intermediate_calibration_data(
    raw_ms_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    calibrated_mss: list[Path],
    original_data_root: str | os.PathLike[str] | None = None,
    logger_fn: LogFn = None,
) -> None:
    """Remove raw MS, working copies and (optionally) original data after a full success.

    Every path that contains a calibrated output (or the output root itself) is
    protected and left untouched. In a delivery tree the raw MS of each
    calibrated one is removed from its member's ``raw/`` folder, with its flag
    versions and markers; the tree itself stays.
    """
    from ..delivery import is_delivery_root
    from ..unpack import raw_ms_path

    output_path = Path(output_root).expanduser().resolve()
    protected = [*calibrated_mss, output_path]
    if is_delivery_root(raw_ms_root):
        for calibrated in calibrated_mss:
            raw = raw_ms_path(raw_ms_root, Path(calibrated).name.removesuffix(".ms.split.cal"))
            for suffix in ("", ".flagversions"):
                _remove_tree_after_success(raw.with_name(raw.name + suffix), protected, logger_fn)
            for marker in (".done", ".failed"):
                raw.with_name(raw.name + marker).unlink(missing_ok=True)
    else:
        _remove_tree_after_success(raw_ms_root, protected, logger_fn=logger_fn)
    _remove_tree_after_success(output_path / "working", protected, logger_fn=logger_fn)
    if original_data_root is not None:
        _remove_tree_after_success(original_data_root, protected, logger_fn=logger_fn)
