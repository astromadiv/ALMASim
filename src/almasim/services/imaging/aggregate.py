"""The continuum of all the windows of one field in one image (MADIV-28).

A window task images its own window. The aggregate task builds the
single-window MS of every window of one field (of one execution block, or
of a member's execution blocks combined), prepares each as its window task
does (self-calibration, the continuum fitted to the visibilities), and hands
them all to one WSClean run. WSClean grids them into one multi-frequency
image: one output channel per window (``-channels-out`` split at the gaps
between the windows, ``-channel-division-frequencies``), cleaned together
(``-join-channels``) with a spectral polynomial per component
(``-fit-spectral-pol``), so a source's slope across the windows does not
stay in the residual.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from almasim.services.imaging.cont_dat import cont_dat_log
from almasim.services.imaging.markers import (
    imaging_failure_marker_path,
    imaging_log_path,
    imaging_marker_path,
    write_imaging_failure_marker,
    write_imaging_marker,
)
from almasim.services.imaging.ms.member_window import build_member_window_ms
from almasim.services.imaging.ms.single_window import (
    DEFAULT_MAX_WEIGHT,
    extract_single_window_ms_subprocess,
)
from almasim.services.imaging.products import (
    CONTINUUM_COLUMN,
    IMAGE_FILENAME,
    SCRATCH_PLACEHOLDER,
    SINGLE_WINDOW_PLACEHOLDER,
)
from almasim.services.imaging.provenance import Provenance, stamp_fits
from almasim.services.imaging.selfcal.loop import self_calibrate
from almasim.services.imaging.task_run import TaskRun
from almasim.services.imaging.uv_contsub import subtract_uv_continuum_subprocess
from almasim.services.imaging.wsclean.command import WSCleanCommand
from almasim.services.imaging.wsclean.runner import _remove_reorder_files
from almasim.services.runtime.child import call_in_child
from almasim.services.runtime.env import wsclean_environment

# The aggregate task's folder: CONT where a window's is SPW-<n>.
AGGREGATE_NAME = "CONT"
# WSClean's name for the image of all the output channels together.
_MFS = "wsclean-MFS-"
# Terms of the polynomial each clean component is fitted with over frequency.
SPECTRAL_TERMS = 2


def aggregate_directory(window_directory: Path, spw: int) -> Path:
    """The aggregate task's folder beside a window task's: ``SPW-3_FIELD-2`` ->
    ``CONT_FIELD-2``."""
    window_directory = Path(window_directory)
    name = window_directory.name.replace(f"SPW-{int(spw)}", AGGREGATE_NAME, 1)
    return window_directory.with_name(name)


def band_edges(ms_path: str | os.PathLike[str]) -> tuple[float, float]:
    """The lowest and highest frequency (Hz) a single-window MS's channels cover."""
    from almasim.services.imaging.ms_geometry import import_casacore_tables
    from almasim.services.ms.tables import open_table

    with open_table(f"{ms_path}::SPECTRAL_WINDOW", opener=import_casacore_tables()) as windows:
        freq = windows.getcell("CHAN_FREQ", 0)
        width = abs(windows.getcell("CHAN_WIDTH", 0))
    return float((freq - width / 2).min()), float((freq + width / 2).max())


def division_frequencies(edges: Sequence[tuple[float, float]]) -> list[float]:
    """Where to split the output channels so each holds whole windows: the
    middle of every gap between the windows' bands. Overlapping windows share
    a channel, so ``len(result) + 1`` channels hold them all."""
    bands: list[list[float]] = []
    for low, high in sorted(edges):
        if bands and low < bands[-1][1]:
            bands[-1][1] = max(bands[-1][1], high)
        else:
            bands.append([low, high])
    return [(left[1] + right[0]) / 2 for left, right in zip(bands, bands[1:], strict=False)]


def joint_commands(
    command: Sequence[str],
    ms_paths: Sequence[str | os.PathLike[str]],
    divisions: Sequence[float],
    *,
    data_column: str | None = None,
    restart: bool = False,
) -> list[list[str]]:
    """``command`` (one MS, one plane) imaging all of ``ms_paths`` together,
    in one pass, or two with ``restart`` (the second continues the first)."""
    joint = WSCleanCommand.parse(command)
    if divisions:
        channels = len(divisions) + 1
        joint = replace_options(
            joint,
            "-join-channels",
            "-channels-out",
            str(channels),
            "-channel-division-frequencies",
            ",".join(f"{f:.1f}" for f in divisions),
            "-fit-spectral-pol",
            str(min(SPECTRAL_TERMS, channels)),
        )
    if data_column:
        joint = joint.with_option("-data-column", data_column)
    passes = [joint]
    if restart:
        first = joint.with_model_update(True)
        passes = [first, first.continued()]
    # The options are rewritten around one MS; the others are added last.
    return [[*one.argv()[:-1], *(str(ms) for ms in ms_paths)] for one in passes]


def replace_options(command: WSCleanCommand, *options: str) -> WSCleanCommand:
    """``command`` with ``options`` appended, and without ``-no-reorder``:
    WSClean has to reorder to fill several output channels in one pass."""
    kept = tuple(part for part in command.options if part != "-no-reorder")
    return WSCleanCommand(command.program, (*kept, *options), command.ms)


def copy_data_to_corrected(ms_path: str, rows_per_chunk: int = 20_000) -> dict[str, object]:
    """Give an MS a CORRECTED_DATA equal to its DATA. WSClean reads the column
    its first MS has from every MS, so MSs imaged together must agree."""
    from casacore.tables import makecoldesc, table

    ms = table(str(ms_path), readonly=False, ack=False)
    try:
        if "CORRECTED_DATA" in ms.colnames():
            return {"copied": False}
        ms.addcols(makecoldesc("CORRECTED_DATA", ms.getcoldesc("DATA")))
        for start in range(0, ms.nrows(), rows_per_chunk):
            count = min(rows_per_chunk, ms.nrows() - start)
            ms.putcol("CORRECTED_DATA", ms.getcol("DATA", start, count), start, count)
        return {"copied": True, "rows": int(ms.nrows())}
    finally:
        ms.close()


def has_corrected_data(ms_path: str | os.PathLike[str]) -> bool:
    from almasim.services.imaging.ms_geometry import import_casacore_tables
    from almasim.services.ms.tables import open_table

    with open_table(str(ms_path), opener=import_casacore_tables()) as ms:
        return "CORRECTED_DATA" in ms.colnames()


def run_aggregate_continuum_task(  # noqa: C901
    *,
    command: Sequence[str],
    output_dir: str,
    windows: Sequence[Mapping[str, Any]],
    threads: int,
    timeout: float | None = None,
    scratch_root: str | None = None,
    retries: int = 3,
    max_weight: float | None = DEFAULT_MAX_WEIGHT,
    uv_contsub_order: int | None = None,
    pipeline_line_free: bool = False,
    ous_index: str | None = None,
    selfcal_solints: Sequence[str] | None = None,
    selfcal_min_snr: float = 20.0,
    casa_data: str | None = None,
    continuum_restart: bool = False,
    pb_dish_diameter: float | None = None,
    provenance: Provenance | None = None,
    **_window_task_options: object,
) -> dict[str, object]:
    """Image the continuum of several windows of one field together.

    Runs on the worker, like ``run_wsclean_task``, and leaves the same
    markers and log beside ``output_dir``. ``windows`` lists, per window,
    ``ms_path``, ``spw``, ``field_ids``, ``command`` (the window's own
    continuum command, which its self-calibration images with) and
    ``member_windows`` (the (MS, window, field ids) of a member's execution
    blocks, or none). ``command`` is the joint image's: the grid of all the
    windows, one MS placeholder.

    Each window's single-window MS is built as in its own task, then, with
    ``selfcal_solints``, self-calibrated on its own continuum, and, with
    ``uv_contsub_order``, its continuum is fitted to its visibilities. With
    the fit, the joint image is made from the fitted continua, and a window
    whose fit is skipped (too few line-free channels) is left out. Without
    it, every channel of every window is imaged. The other keyword arguments
    of a window task are accepted and ignored (cubes, noise products).

    The products are named as a window task's: ``wsclean-image.fits``,
    ``wsclean-residual.fits``, ``wsclean-model.fits``, ``wsclean-psf.fits``
    and, with ``pb_dish_diameter``, the primary-beam corrected image.
    """
    from almasim.services.imaging.task import pb_correct_task, pb_note

    out = Path(output_dir)
    log_path = imaging_log_path(out)
    if len(windows) < 2:
        raise ValueError("an aggregate continuum needs at least two windows")
    if any(w.get("member_windows") for w in windows) and not casa_data:
        raise ValueError("member windows need casa_data")
    if selfcal_solints and not casa_data:
        raise ValueError("selfcal_solints needs casa_data")
    out.mkdir(parents=True, exist_ok=True)
    imaging_marker_path(out).unlink(missing_ok=True)
    imaging_failure_marker_path(out).unlink(missing_ok=True)
    for stale in out.glob("wsclean-*.fits"):
        stale.unlink(missing_ok=True)
    for stale in out.glob("*.single*.ms"):
        shutil.rmtree(stale, ignore_errors=True)
    _remove_reorder_files(out)
    log_path.write_text("", encoding="utf-8")

    first_ms, first_spw = str(windows[0]["ms_path"]), int(windows[0]["spw"])
    spws = [int(w["spw"]) for w in windows]
    if provenance is not None:
        from dataclasses import replace

        provenance = replace(provenance, spw=None)  # an image of several windows
    task = TaskRun(
        out=out,
        log_path=log_path,
        ms_path=first_ms,
        spw=first_spw,
        env=wsclean_environment(threads),
        timeout=timeout,
        retries=retries,
        provenance=provenance,
    )
    scratch_dir: Path | None = None
    if scratch_root:
        import tempfile

        Path(scratch_root).mkdir(parents=True, exist_ok=True)
        scratch_dir = Path(tempfile.mkdtemp(prefix=f"{out.name}-", dir=scratch_root))

    def placed(cmd: Sequence[str], ms: Path | None = None) -> list[str]:
        """``cmd`` with the scratch folder, and the window's MS, filled in."""
        filled = WSCleanCommand.parse(cmd)
        if scratch_dir is not None:
            filled = filled.substituted(SCRATCH_PLACEHOLDER, scratch_dir)
        if ms is not None:
            filled = filled.substituted(SINGLE_WINDOW_PLACEHOLDER, ms)
        return filled.argv()

    started = time.time()
    built: list[Path] = []
    try:
        kept: list[tuple[int, Path]] = []
        with log_path.open("a", encoding="utf-8") as log:
            for position, window in enumerate(windows):
                spw = int(window["spw"])
                ms = out / f"w{position}.spw{spw}.single.ms"
                built.append(ms)
                members = window.get("member_windows") or None
                try:
                    if members:
                        build_member_window_ms(
                            members,
                            ms,
                            casa_data=str(casa_data),
                            timeout=timeout,
                            log_path=log_path,
                            max_weight=max_weight,
                        )
                    else:
                        extract_single_window_ms_subprocess(
                            window["ms_path"],
                            spw,
                            list(window.get("field_ids") or ()) or None,
                            ms,
                            timeout=timeout,
                            log_path=log_path,
                            max_weight=max_weight,
                        )
                except Exception as exc:
                    raise task.fail(f"Could not build the MS of window {spw}: {exc}") from exc
                log.write(f"# window {spw}: its single-window MS is {ms.name}\n")
                log.flush()
                pipeline_ms = str(window["ms_path"]) if pipeline_line_free else None
                if selfcal_solints:
                    log.write(f"# window {spw}: self-calibration on its own continuum\n")
                    self_calibrate(
                        ms,
                        placed(window["command"], ms),
                        task=task,
                        log=log,
                        solints=selfcal_solints,
                        min_snr=selfcal_min_snr,
                        casa_data=str(casa_data),
                        restart=continuum_restart,
                        keep_gains=False,
                        pipeline_ms=pipeline_ms,
                        ous_index=ous_index,
                    )
                if uv_contsub_order is not None:
                    try:
                        fit = subtract_uv_continuum_subprocess(
                            ms,
                            order=uv_contsub_order,
                            timeout=timeout,
                            log_path=log_path,
                            pipeline_ms=pipeline_ms,
                            ous_index=ous_index,
                        )
                    except Exception as exc:
                        raise task.fail(
                            f"Could not fit the continuum of window {spw}: {exc}"
                        ) from exc
                    if "cont_dat" in fit:
                        log.write(f"# window {spw}: continuum fit: {cont_dat_log(fit)}\n")
                    if fit.get("skipped"):
                        log.write(
                            f"# window {spw} left out of the aggregate continuum: "
                            f"{fit['skipped']}\n"
                        )
                        task.note(f"window {spw} left out: {fit['skipped']}")
                        continue
                    log.write(
                        f"# window {spw}: continuum fitted (order {fit['order']}) over "
                        f"{fit['line_free_channels']} of {fit['usable_channels']} channels, "
                        f"lines in channels {fit['line_ranges'] or 'none'}\n"
                    )
                kept.append((spw, ms))
            if not kept:
                raise task.fail("No window has a continuum to image: every fit was skipped")
            ms_paths = [ms for _, ms in kept]
            data_column = CONTINUUM_COLUMN if uv_contsub_order is not None else None
            if data_column is None:
                corrected = [has_corrected_data(ms) for ms in ms_paths]
                if any(corrected) and not all(corrected):
                    # WSClean reads the first MS's column from all of them.
                    for ms, has in zip(ms_paths, corrected, strict=True):
                        if not has:
                            call_in_child(
                                "almasim.services.imaging.aggregate:copy_data_to_corrected",
                                {"ms_path": str(ms)},
                                what="copy of DATA into CORRECTED_DATA",
                                env=wsclean_environment(1),
                                timeout=timeout,
                                log_path=log_path,
                            )
                    log.write("# DATA copied into CORRECTED_DATA where no gains were applied\n")
            divisions = division_frequencies([band_edges(ms) for ms in ms_paths])
            commands = joint_commands(
                placed(command),
                ms_paths,
                divisions,
                data_column=data_column,
                restart=continuum_restart,
            )
            imaged = ", ".join(str(spw) for spw, _ in kept)
            note = (
                f"aggregate continuum of windows {imaged} "
                f"({len(divisions) + 1} output channel(s)"
                + (", from the fitted continua)" if data_column else ")")
            )
            log.write(f"# {note}\n")
            log.flush()
            task.note(note)
            attempts = 0
            for cmd in commands:
                attempts = task.run(cmd, log)
        # WSClean names the image of all channels <name>-MFS-<product>.fits and
        # leaves one set per output channel beside it.
        for mfs in sorted(out.glob(f"{_MFS}*.fits")):
            mfs.replace(out / mfs.name.replace(_MFS, "wsclean-", 1))
        for plane in out.glob("wsclean-[0-9][0-9][0-9][0-9]-*.fits"):
            plane.unlink(missing_ok=True)
        if not (out / IMAGE_FILENAME).is_file():
            raise task.fail(f"WSClean exited 0 but wrote no {IMAGE_FILENAME} under {out}", 0)
        header = task.header()
        for path in sorted(out.glob("*.fits")):
            try:
                stamp_fits(path, header)
            except (OSError, ValueError) as exc:  # the images stand without it
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"# provenance NOT written into {path.name}: {exc}\n")
        if pb_dish_diameter:
            try:
                corrected_images = pb_correct_task(out, pb_dish_diameter)
            except Exception as exc:
                raise task.fail(f"Could not primary-beam correct the image: {exc}") from exc
            with log_path.open("a", encoding="utf-8") as log:
                for name, info in corrected_images.items():
                    log.write(f"# primary-beam corrected {name}: {pb_note(info)}\n")
        elapsed = time.time() - started
        write_imaging_marker(
            out,
            ms_path=first_ms,
            spw=first_spw,
            command=commands[0] if len(commands) == 1 else None,
            commands=commands if len(commands) > 1 else None,
            seconds=elapsed,
            images=[IMAGE_FILENAME],
        )
        return {
            "ms": first_ms,
            "spw": first_spw,
            "spws": spws,
            "imaged_spws": [spw for spw, _ in kept],
            "output": str(out),
            "image": str(out / IMAGE_FILENAME),
            "seconds": round(elapsed, 1),
            "attempts": attempts,
            "channels": len(divisions) + 1,
        }
    except RuntimeError:
        raise
    except Exception as exc:
        write_imaging_failure_marker(out, ms_path=first_ms, spw=first_spw, error=str(exc))
        raise RuntimeError(str(exc)) from exc
    finally:
        if scratch_dir is not None:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        for ms in built:
            shutil.rmtree(ms, ignore_errors=True)
        for stale in out.glob("*.single*.ms"):
            shutil.rmtree(stale, ignore_errors=True)
