"""Self-calibration rounds, their stop criterion and the noise products.

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from almasim.services.imaging.cube import (
    stack_channel_images,
)
from almasim.services.imaging.products import (
    CUBE_NAME,
    CUBE_SELFCAL_CORRECTION_FILENAME,
    CUBE_TOTAL_RESIDUAL_FILENAME,
    SELFCAL_COLUMN,
    SELFCAL_CONTINUUM_COLUMN,
    SELFCAL_CORRECTION_FILENAME,
    SELFCAL_GAINS_DIRNAME,
    SELFCAL_LINE_COLUMN,
    TOTAL_RESIDUAL_FILENAME,
)
from almasim.services.imaging.selfcal.noise_products import (
    _single_image,
    add_images,
)
from almasim.services.imaging.wsclean.command import (
    WSCleanCommand,
    _dirty_command,
    _selfcal_command,
)
from almasim.services.runtime.child import call_in_child
from almasim.services.runtime.env import (
    casa_environment,
    wsclean_environment,
)

if TYPE_CHECKING:
    from almasim.services.imaging.task_run import TaskRun


# The module each child-process function lives in (MADIV-28 T8.4: they were
# all reached through the archive_imaging re-export module).
_CHILD_FUNCTIONS = {
    "adopt_corrected_data": "almasim.services.imaging.selfcal.casa_ops",
    "casa_selfcal_apply": "almasim.services.imaging.selfcal.casa_ops",
    "casa_selfcal_solve": "almasim.services.imaging.selfcal.casa_ops",
    "export_gain_table": "almasim.services.imaging.selfcal.casa_ops",
    "selfcal_prepare": "almasim.services.imaging.selfcal.casa_ops",
    "selfcal_correction_columns": "almasim.services.imaging.selfcal.noise_products",
    "drop_corrected_data": "almasim.services.imaging.uv_contsub",
    "image_peak": "almasim.services.imaging.selfcal.loop",
    "image_snr": "almasim.services.imaging.selfcal.loop",
}


def _call_in_child(
    function: str,
    kwargs: Mapping[str, object],
    *,
    what: str,
    timeout: float | None = None,
    log_path: str | os.PathLike[str] | None = None,
    casa: bool = False,
) -> dict[str, object]:
    """``function(**kwargs)`` (named in ``_CHILD_FUNCTIONS``) in a child process; its JSON result.

    casacore and CASA hold the GIL for the whole call, which would freeze a
    Dask worker (see ``extract_single_window_ms_subprocess``), and
    python-casacore and casatools must not share a process. The child's
    stderr is appended to ``log_path``; ``casa`` gives it ``casa_environment``.
    """
    return call_in_child(
        f"{_CHILD_FUNCTIONS[function]}:{function}",
        kwargs,
        what=what,
        env=casa_environment() if casa else wsclean_environment(1),
        timeout=timeout,
        log_path=log_path,
    )


def image_snr(image_path: Path, residual_path: Path) -> float:
    """Peak of the restored image over the robust noise (1.4826 MAD) of the residual."""
    import numpy as np

    from almasim.services.imaging.fits_io import read_image
    from almasim.services.imaging.stats import robust_sigma

    image = read_image(image_path).squeeze()
    residual = read_image(residual_path).squeeze()
    noise = robust_sigma(residual)
    return float(np.nanmax(image) / noise) if noise > 0 else 0.0


def image_peak(image_path: Path) -> float:
    """Peak of the restored image."""
    import numpy as np

    from almasim.services.imaging.fits_io import read_image

    return float(np.nanmax(read_image(image_path)))


def parse_selfcal_rounds(solints: Sequence[str]) -> list[tuple[str, str]]:
    """``(calmode, solint)`` per round: ``inf`` or ``30s`` is a phase round,
    ``ap:inf`` an amplitude-and-phase one. An amplitude round must come after
    a phase round: it is solved on top of the phase solution."""
    rounds = []
    for item in solints:
        calmode, _, solint = item.rpartition(":")
        calmode = calmode or "p"
        if calmode not in ("p", "ap") or not re.fullmatch(r"inf|int|\d+(\.\d+)?s", solint):
            raise ValueError(
                f"self-calibration rounds are [ap:]inf, [ap:]int or [ap:]<seconds>s, not {item!r}"
            )
        rounds.append((calmode, solint))
    if rounds and rounds[0][0] == "ap":
        raise ValueError("an amplitude (ap:) round needs a phase round before it")
    return rounds


def self_calibrate(  # noqa: C901
    ms_path: Path,
    continuum_command: list[str],
    *,
    task: TaskRun,
    log,
    solints: Sequence[str],
    min_snr: float,
    casa_data: str,
    restart: bool = False,
    keep_gains: bool = True,
) -> dict[str, object]:
    """Self-calibrate ``ms_path`` against its own continuum, round by round.

    The continuum is imaged (``continuum_command`` into ``selfcal-0`` under
    ``task.out``, writing
    MODEL_DATA); below ``min_snr`` (peak over residual noise) nothing is
    done. Each round (``parse_selfcal_rounds``: ``inf``, ``30s`` for phase,
    ``ap:inf`` for amplitude and phase) then solves gains of DATA against the
    model at its solution interval with CASA ``gaincal``
    (``casa_selfcal_solve``), an amplitude round on top of the kept phase
    table, applies the kept phase table and the new one into CORRECTED_DATA
    (``casa_selfcal_apply``) and reimages.
    A round is kept only if the image S/N rose; otherwise the best earlier
    gains are restored (or CORRECTED_DATA dropped) and the rounds stop.
    The temporary images and calibration tables are removed; CASA, run with
    ``casa_data`` as its runtime data, logs to ``casa-selfcal.log`` in
    ``out``.

    With ``restart`` every image is cleaned in two passes, as the continuum
    is (``continuum_restart_commands``), so the model the gains are solved
    against is complete. With ``keep_gains`` the applied tables are kept in
    ``selfcal_gains/`` (``_keep_gain_tables``).

    ALMASim's own StefCal-style solver gave the same S/N within ~1 % on four
    cycle11 cubes and was dropped for CASA (2026-10-17).
    """
    out, timeout, fail = task.out, task.timeout, task.fail
    plan = parse_selfcal_rounds(solints)
    rounds: list[dict[str, object]] = []
    casa_log = str(out / "casa-selfcal.log")
    prepared: dict[str, object] = {}

    def image(index: int) -> tuple[float, float]:
        name = f"selfcal-{index}"
        command = _selfcal_command(continuum_command, out, name)
        task.run(command, log)
        if restart:
            task.run(WSCleanCommand.parse(command).continued().argv(), log)
        image_path = out / f"{name}-image.fits"
        return image_snr(image_path, out / f"{name}-residual.fits"), image_peak(image_path)

    def call(function: str, **kwargs: object) -> dict[str, object]:
        try:
            return _call_in_child(
                function,
                kwargs,
                what=function,
                timeout=timeout,
                log_path=log.name,
                casa=function.startswith("casa_"),
            )
        except Exception as exc:
            raise fail(f"Self-calibration failed: {exc}") from exc

    def solve(gains: Path, calmode: str, solint: str, pre_apply: list[Path]) -> dict:
        return call(
            "casa_selfcal_solve",
            ms_path=str(ms_path),
            caltable=str(gains),
            solint=solint,
            spw=prepared["spw"],
            refant=prepared["refant"],
            casa_data=casa_data,
            log_file=casa_log,
            calmode=calmode,
            pre_apply=[str(t) for t in pre_apply],
        )

    def apply(tables: list[Path]) -> None:
        if not tables:
            call("drop_corrected_data", ms_path=str(ms_path))
        else:
            call(
                "casa_selfcal_apply",
                ms_path=str(ms_path),
                caltables=[str(t) for t in tables],
                casa_data=casa_data,
                log_file=casa_log,
            )

    try:
        if call("adopt_corrected_data", ms_path=str(ms_path)).get("adopted"):
            log.write("# self-calibration: incoming CORRECTED_DATA adopted as DATA\n")
        best_snr, best_peak = image(0)
        log.write(
            f"# self-calibration: initial continuum S/N {best_snr:.1f}, "
            f"peak {best_peak:.4g} Jy/beam\n"
        )
        if best_snr < min_snr:
            log.write(f"# self-calibration skipped: S/N below {min_snr:g}\n")
            log.flush()
            return {"applied": False, "initial_snr": best_snr, "rounds": rounds}
        prepared = call("selfcal_prepare", ms_path=str(ms_path))
        log.write(
            f"# self-calibration with CASA gaincal: refant {prepared['refant']}, "
            f"{prepared['line_free_channels']} line-free channels\n"
        )
        # The kept tables: the best phase table, then the best amplitude one.
        best_phase: Path | None = None
        best_amp: Path | None = None
        for index, (calmode, solint) in enumerate(plan, start=1):
            gains = out / f"selfcal-{index}.{'G' if calmode == 'p' else 'A'}"
            # An ap round always follows a kept phase round: parse_selfcal_rounds
            # puts a phase round first, and the rounds stop at the first rejection.
            pre_apply = [best_phase] if calmode == "ap" else []
            solved = solve(gains, calmode, solint, pre_apply)
            trial = [gains] if calmode == "p" else [best_phase, gains]
            apply(trial)
            snr, peak = image(index)
            kept = snr > best_snr
            rounds.append({**solved, "snr": round(snr, 1), "peak": peak, "kept": kept})
            detail = (
                f"median |phase| {solved['median_phase_deg']} deg"
                if calmode == "p"
                else f"median |amp - 1| {solved['median_amp_dev']}"
            )
            log.write(
                f"# self-calibration round {index} ({calmode}, solint {solint}): "
                f"{solved['solutions']}/{solved['possible']} antenna solutions, {detail}; "
                f"S/N {best_snr:.1f} -> {snr:.1f}, peak {best_peak:.4g} -> {peak:.4g} Jy/beam"
                f"{'' if kept else ', rejected'}\n"
            )
            log.flush()
            if not kept:
                apply([t for t in (best_phase, best_amp) if t])
                break
            best_snr, best_peak = snr, peak
            if calmode == "p":
                best_phase, best_amp = gains, None
            else:
                best_amp = gains
        applied = [t.name for t in (best_phase, best_amp) if t]
        outcome = f"applied ({', '.join(applied)})" if applied else "not applied"
        log.write(f"# self-calibration {outcome}: final continuum S/N {best_snr:.1f}\n")
        kept_gains = (
            _keep_gain_tables(
                [(t, kind) for t, kind in ((best_phase, "phase"), (best_amp, "amplitude")) if t],
                out,
                call,
            )
            if keep_gains
            else []
        )
        if kept_gains:
            log.write(
                f"# self-calibration gains kept in {SELFCAL_GAINS_DIRNAME}/: "
                f"{', '.join(kept_gains)}\n"
            )
        log.flush()
        return {"applied": bool(applied), "snr": best_snr, "rounds": rounds, "gains": kept_gains}
    finally:
        for leftover in out.glob("selfcal-*"):
            if leftover.is_dir():
                shutil.rmtree(leftover, ignore_errors=True)
            else:
                leftover.unlink(missing_ok=True)


def _keep_gain_tables(
    tables: Sequence[tuple[Path, str]], out: Path, call: Callable[..., dict[str, object]]
) -> list[str]:
    """Copy the kept calibration tables to ``selfcal_gains/`` as ``<kind>.cal``,
    each with a CSV of its solutions (``export_gain_table``); the names written."""
    if not tables:
        return []
    target = out / SELFCAL_GAINS_DIRNAME
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    written = []
    for table, kind in tables:
        shutil.copytree(table, target / f"{kind}.cal")
        call("export_gain_table", caltable=str(table), csv_path=str(target / f"{kind}.csv"))
        written += [f"{kind}.cal", f"{kind}.csv"]
    return written


def _selfcal_noise_products(  # noqa: C901
    ms_path: Path,
    *,
    continuum_command: Sequence[str] | None,
    cube_command: Sequence[str] | None,
    fit: dict[str, object] | None,
    task: TaskRun,
    log,
) -> list[str]:
    """Image what self-calibration removed and form the total residuals.

    The kept residual is the dirty image (same grid and weighting) of the
    self-calibrated data minus the model. The correction column
    (``selfcal_correction_columns``, split like the data when the continuum
    was subtracted in the visibilities) imaged the same way gives
    ``wsclean[-cube]-selfcal-correction.fits``; imaging with fixed weights is
    linear, so residual + correction is the dirty image of the data before
    self-calibration minus the model, everything the final model does not
    explain: ``wsclean[-cube]-residual-total.fits``. Returns the files written.
    """
    out, fail = task.out, task.fail
    split = fit is not None
    try:
        _call_in_child(
            "selfcal_correction_columns",
            {
                "ms_path": str(ms_path),
                "line_ranges": fit["line_ranges"] if fit is not None else None,
                "order": fit["order"] if fit is not None else 1,
            },
            what="selfcal_correction_columns",
            timeout=task.timeout,
            log_path=log.name,
        )
    except Exception as exc:
        raise fail(f"Could not write the self-calibration correction: {exc}") from exc
    written: list[str] = []
    products = (
        (
            continuum_command,
            "wsclean",
            SELFCAL_CONTINUUM_COLUMN,
            SELFCAL_CORRECTION_FILENAME,
            TOTAL_RESIDUAL_FILENAME,
            False,
        ),
        (
            cube_command,
            CUBE_NAME,
            SELFCAL_LINE_COLUMN,
            CUBE_SELFCAL_CORRECTION_FILENAME,
            CUBE_TOTAL_RESIDUAL_FILENAME,
            True,
        ),
    )
    for command, prefix, split_column, correction_name, total_name, is_cube in products:
        if not command:
            continue
        name = f"{prefix}-selfcal-correction"
        task.run(_dirty_command(command, out, name, split_column if split else SELFCAL_COLUMN), log)
        image: Path | None = None
        if is_cube:
            try:
                stacked = stack_channel_images(
                    out, prefix=name, keep=("dirty", "image"), provenance=task.header()
                )
            except Exception as exc:
                raise fail(f"Could not stack the self-calibration correction cube: {exc}") from exc
            for duplicate in out.glob(f"{name}-MFS-*.fits"):
                duplicate.unlink(missing_ok=True)
            image = next(
                (
                    out / f"{name}-{product}.fits"
                    for product in ("dirty", "image")
                    if product in stacked
                ),
                None,
            )
        image = image or _single_image(out, name)
        if image is None:
            raise fail(f"WSClean wrote no {name} image under {out}")
        image.replace(out / correction_name)
        for leftover in out.glob(f"{name}-*.fits"):
            leftover.unlink(missing_ok=True)
        written.append(correction_name)
        residual = out / f"{prefix}-residual.fits"
        if residual.is_file():
            try:
                add_images(
                    residual,
                    out / correction_name,
                    out / total_name,
                    "ALMASim total residual: residual + self-calibration correction",
                )
            except Exception as exc:
                raise fail(f"Could not form {total_name}: {exc}") from exc
            written.append(total_name)
    log.write(f"# self-calibration noise products: {', '.join(written) or 'none'}\n")
    log.flush()
    return written
