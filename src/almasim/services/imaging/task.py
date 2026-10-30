"""One imaging task: extraction, uv-contsub, self-cal, WSClean, cubes and pbcor.

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from almasim.services.imaging.continuum import (
    subtract_continuum,
)
from almasim.services.imaging.cube import (
    _line_mask_pass,
    stack_channel_images,
)
from almasim.services.imaging.markers import (
    imaging_failure_marker_path,
    imaging_log_path,
    imaging_marker_path,
    write_imaging_failure_marker,
    write_imaging_marker,
)
from almasim.services.imaging.ms.single_window import (
    DEFAULT_MAX_WEIGHT,
    extract_single_window_ms_subprocess,
)
from almasim.services.imaging.products import (
    CONTINUUM_COLUMN,
    CONTSUB_FILENAME,
    CUBE_IMAGE_FILENAME,
    CUBE_NAME,
    CUBE_PB_FILENAME,
    CUBE_PBCOR_FILENAME,
    CUBE_SELFCAL_CORRECTION_FILENAME,
    CUBE_TOTAL_RESIDUAL_FILENAME,
    IMAGE_FILENAME,
    LINE_COLUMN,
    PB_FILENAME,
    PBCOR_FILENAME,
    SCRATCH_PLACEHOLDER,
    SELFCAL_CORRECTION_FILENAME,
    SELFCAL_GAINS_DIRNAME,
    SINGLE_WINDOW_PLACEHOLDER,
    TOTAL_RESIDUAL_FILENAME,
)
from almasim.services.imaging.provenance import Provenance, stamp_fits
from almasim.services.imaging.selfcal.loop import _selfcal_noise_products, self_calibrate
from almasim.services.imaging.task_run import TaskRun
from almasim.services.imaging.uv_contsub import (
    subtract_uv_continuum_subprocess,
)
from almasim.services.imaging.wsclean.command import (
    WSCleanCommand,
    _with_data_column,
    continuum_restart_commands,
)
from almasim.services.imaging.wsclean.runner import _remove_reorder_files
from almasim.services.runtime.env import (
    wsclean_environment,
)


def run_wsclean_task(  # noqa: C901
    *,
    command: Sequence[str],
    output_dir: str,
    ms_path: str,
    spw: int,
    threads: int,
    timeout: float | None = None,
    scratch_root: str | None = None,
    retries: int = 3,
    single_window: bool = False,
    field_ids: Sequence[int] | None = None,
    max_weight: float | None = DEFAULT_MAX_WEIGHT,
    cube_command: Sequence[str] | None = None,
    cube_products: Sequence[str] | None = None,
    contsub_order: int | None = None,
    uv_contsub_order: int | None = None,
    selfcal_solints: Sequence[str] | None = None,
    selfcal_min_snr: float = 20.0,
    casa_data: str | None = None,
    continuum_restart: bool = False,
    cube_line_mask: bool = False,
    noise_products: bool = False,
    pb_dish_diameter: float | None = None,
    provenance: Provenance | None = None,
) -> dict[str, object]:
    """Run one WSClean task in a subprocess and leave a marker whatever happens.

    Runs on the worker. Output is streamed to ``SPW-<n>.log`` as it arrives.
    A non-zero exit, a signal, a timeout, a binary that cannot start, or a
    clean exit that produced no ``wsclean-image.fits`` all end in a ``.failed``
    marker and a ``RuntimeError`` so the driver counts the task as failed;
    success ends in a ``.done`` marker.

    A death by signal (a segfault, say) is retried up to ``retries`` times
    before the task is declared failed: WSClean 3.7's multi-threaded
    reordering crashes non-deterministically on some inputs and passes on the
    next run, and a crash costs seconds while the task costs minutes. Exits
    with an error status are WSClean's own diagnostics and are not retried.

    ``command`` makes the continuum image (empty: none); ``cube_command``,
    run after it on the same single-window MS, the spectral cube. The cube
    run leaves one FITS per channel; they are stacked into
    ``wsclean-cube-<product>.fits`` (only ``cube_products`` kept, all when
    None) and, with ``contsub_order``, the image cube is also written
    continuum-subtracted (``subtract_continuum``) before the task counts as
    done. Only the images this task makes are removed beforehand, so a cube
    task leaves an existing continuum image alone.

    With ``uv_contsub_order`` (needs ``single_window`` and a cube) the
    continuum is subtracted from the visibilities of the single-window MS
    first (``subtract_uv_continuum``): the continuum image is made from the
    fitted continuum and the cube from what is left, the lines. When too few
    channels are line-free the subtraction is skipped and noted in the log.

    With ``selfcal_solints`` (needs ``single_window`` and a continuum
    command) the single-window MS is phase self-calibrated first, one round
    per solution interval (``self_calibrate``); everything after it, the
    continuum subtraction included, reads the corrected data.

    With ``noise_products``, where self-calibration was applied, what it
    removed is imaged too (``_selfcal_noise_products``): the correction and
    the total residual (kept residual + correction) of the continuum and the
    cube; its gain tables are kept in ``selfcal_gains/``.

    With ``pb_dish_diameter`` (the antennas' diameter in metres) the
    continuum image and the image cube are primary-beam corrected
    (``pb_correct_task``): ``wsclean[-cube]-pb.fits`` and
    ``wsclean[-cube]-image.pbcor.fits``.

    Every FITS the task leaves carries ``provenance`` (by default the MS's
    execution block and the window) and its steps, each WSClean command among
    them (see :mod:`almasim.services.imaging.provenance`).
    """
    out = Path(output_dir)
    log_path = imaging_log_path(out)
    commands = [[str(part) for part in c] for c in (command, cube_command or []) if c]
    if not commands:
        raise ValueError("run_wsclean_task needs a continuum or a cube command")
    if uv_contsub_order is not None and not (single_window and cube_command):
        raise ValueError("uv_contsub_order needs single_window and a cube_command")
    if selfcal_solints and not (single_window and command and casa_data):
        raise ValueError("selfcal_solints needs single_window, a continuum command and casa_data")
    if continuum_restart and command:
        if not single_window:
            raise ValueError("continuum_restart writes MODEL_DATA: it needs single_window")
        commands[:1] = continuum_restart_commands(commands[0])
    if cube_line_mask and cube_command:
        if not single_window:
            raise ValueError("cube_line_mask writes MODEL_DATA: it needs single_window")
        # The second pass continues from this one's model visibilities.
        commands[-1] = WSCleanCommand.parse(commands[-1]).with_model_update(True).argv()
    required = [IMAGE_FILENAME] * bool(command) + [CUBE_IMAGE_FILENAME] * bool(cube_command)
    if cube_command and contsub_order is not None:
        required.append(CONTSUB_FILENAME)
    # Only after the arguments are checked, so a rejected call leaves no directory.
    out.mkdir(parents=True, exist_ok=True)
    imaging_marker_path(out).unlink(missing_ok=True)
    imaging_failure_marker_path(out).unlink(missing_ok=True)
    for name in required:
        (out / name).unlink(missing_ok=True)
    # The noise products of what this task regenerates: a rerun whose
    # self-calibration is skipped or rejected must not leave the previous
    # run's beside the new images. Self-calibration runs with the continuum.
    stale_products = [SELFCAL_CORRECTION_FILENAME, TOTAL_RESIDUAL_FILENAME] * bool(command)
    stale_products += [CUBE_SELFCAL_CORRECTION_FILENAME, CUBE_TOTAL_RESIDUAL_FILENAME] * bool(
        cube_command
    )
    stale_products += [PB_FILENAME, PBCOR_FILENAME] * bool(command)
    stale_products += [CUBE_PB_FILENAME, CUBE_PBCOR_FILENAME] * bool(cube_command)
    for name in stale_products:
        (out / name).unlink(missing_ok=True)
    if command:
        shutil.rmtree(out / SELFCAL_GAINS_DIRNAME, ignore_errors=True)
    log_path.write_text("", encoding="utf-8")
    # A previous attempt that crashed or was killed with its worker leaves
    # WSClean's reorder files (hundreds of GB across a run) and the
    # single-window copy behind; nothing else ever removes them.
    _remove_reorder_files(out)
    for stale in out.glob("spw*.single.ms"):
        shutil.rmtree(stale, ignore_errors=True)
    for stale in out.glob(f"{CUBE_NAME}-[0-9][0-9][0-9][0-9]-*.fits"):
        stale.unlink(missing_ok=True)

    single_window_ms: Path | None = None
    extraction_note = ""
    if single_window:
        # Extract the window (and fields) into a per-task MS so WSClean never
        # reorders; ``SINGLE_WINDOW_PLACEHOLDER`` in the command is replaced.
        single_window_ms = out / f"spw{int(spw)}.single.ms"
        try:
            extracted = extract_single_window_ms_subprocess(
                ms_path,
                spw,
                field_ids,
                single_window_ms,
                timeout=timeout,
                log_path=log_path,
                max_weight=max_weight,
            )
        except Exception as exc:
            error = f"Could not extract spectral window {spw} into a single-window MS: {exc}"
            write_imaging_failure_marker(out, ms_path=ms_path, spw=spw, error=error)
            raise RuntimeError(error) from exc
        fields_note = list(field_ids) if field_ids else "all"
        size_gb = float(extracted.get("bytes", 0) or 0) / 1e9
        kept = int(extracted.get("rows") or 0)
        dropped = int(extracted.get("dropped_rows") or 0)
        drop_note = (
            f", dropped {dropped} rows ({100.0 * dropped / max(kept + dropped, 1):.1f} %) "
            f"with WEIGHT > {float(extracted.get('max_weight') or 0):g}"
            if extracted.get("max_weight")
            else ""
        )
        extraction_note = (
            f"extracted spectral window {int(spw)} fields {fields_note} into "
            f"{single_window_ms.name}: {kept} rows{drop_note}"
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"# extracted spectral window {int(spw)} fields {fields_note} into "
                f"{single_window_ms.name}: {kept} rows, {size_gb:.1f} GB, "
                f"{extracted.get('seconds')} s{drop_note}\n"
            )
        commands = [
            WSCleanCommand.parse(c).substituted(SINGLE_WINDOW_PLACEHOLDER, single_window_ms).argv()
            for c in commands
        ]
    scratch_dir: Path | None = None
    if scratch_root:
        # Per-task scratch for WSClean's reordered visibilities, created here
        # on the worker and removed whatever happens. ``__SCRATCH__`` in the
        # command (the -temp-dir value) is replaced by it.
        import tempfile

        Path(scratch_root).mkdir(parents=True, exist_ok=True)
        scratch_dir = Path(tempfile.mkdtemp(prefix=f"{out.name}-", dir=scratch_root))
        commands = [
            WSCleanCommand.parse(c).substituted(SCRATCH_PLACEHOLDER, scratch_dir).argv()
            for c in commands
        ]
    task = TaskRun(
        out=out,
        log_path=log_path,
        ms_path=ms_path,
        spw=spw,
        env=wsclean_environment(threads),
        timeout=timeout,
        retries=retries,
        provenance=provenance,
    )
    if extraction_note:
        task.note(extraction_note)
    started = time.time()

    try:
        selfcal: dict[str, object] = {}
        fit: dict[str, object] | None = None
        if single_window_ms is not None and (selfcal_solints or uv_contsub_order is not None):
            with log_path.open("a", encoding="utf-8") as log:
                if selfcal_solints:
                    selfcal = self_calibrate(
                        single_window_ms,
                        commands[0],
                        task=task,
                        log=log,
                        solints=selfcal_solints,
                        min_snr=selfcal_min_snr,
                        casa_data=str(casa_data),
                        restart=continuum_restart,
                        keep_gains=noise_products,
                    )
                if uv_contsub_order is not None:
                    commands, fit = _subtract_uv_continuum_step(
                        single_window_ms,
                        commands,
                        has_continuum=bool(command),
                        order=uv_contsub_order,
                        task=task,
                        log=log,
                    )
        after_images = None
        if noise_products and selfcal.get("applied") and single_window_ms is not None:
            sw_ms, final_commands, final_fit = single_window_ms, list(commands), fit

            def after_images() -> None:
                with log_path.open("a", encoding="utf-8") as log:
                    _selfcal_noise_products(
                        sw_ms,
                        continuum_command=final_commands[0] if command else None,
                        cube_command=final_commands[-1] if cube_command else None,
                        fit=final_fit,
                        task=task,
                        log=log,
                    )

        return _run_and_mark(
            task,
            commands=commands,
            required=required,
            started=started,
            cube_products=cube_products,
            contsub_order=contsub_order,
            cube_line_mask=bool(cube_line_mask and cube_command),
            after_images=after_images,
            pb_dish_diameter=pb_dish_diameter,
        )
    finally:
        if scratch_dir is not None:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        if single_window_ms is not None:
            shutil.rmtree(single_window_ms, ignore_errors=True)


def _run_and_mark(  # noqa: C901
    task: TaskRun,
    *,
    commands: list[list[str]],
    required: Sequence[str],
    started: float,
    cube_products: Sequence[str] | None = None,
    contsub_order: int | None = None,
    cube_line_mask: bool = False,
    after_images: Callable[[], None] | None = None,
    pb_dish_diameter: float | None = None,
) -> dict[str, object]:
    out, log_path, fail = task.out, task.log_path, task.fail
    attempts = 0
    commands = list(commands)
    # run_wsclean_task truncated the log; for a single-window task it already
    # holds the extraction notes, so append.
    with log_path.open("a", encoding="utf-8") as log:
        for cmd in list(commands):
            attempts = task.run(cmd, log)
        if cube_line_mask:
            second = _line_mask_pass(commands[-1], out, log)
            if second:
                commands.append(second)
                attempts = task.run(second, log)
    cube_channels: dict[str, int] = {}
    if CUBE_IMAGE_FILENAME in required:
        try:
            cube_channels = stack_channel_images(
                out, prefix=CUBE_NAME, keep=cube_products, provenance=task.header()
            )
        except Exception as exc:
            raise fail(f"Could not stack the channel images into cubes: {exc}") from exc
        # WSClean also writes a <name>-MFS-<product>.fits for a -channels-out
        # run: a copy of the continuum the continuum run already made.
        for duplicate in out.glob(f"{CUBE_NAME}-MFS-*.fits"):
            duplicate.unlink(missing_ok=True)
        notes = ", ".join(f"{product} ({n} channels)" for product, n in cube_channels.items())
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"# stacked channel images into cubes: {notes or 'none'}\n")
        if contsub_order is not None and (out / CUBE_IMAGE_FILENAME).is_file():
            try:
                fit = subtract_continuum(
                    out / CUBE_IMAGE_FILENAME, out / CONTSUB_FILENAME, order=contsub_order
                )
            except Exception as exc:
                raise fail(f"Could not subtract the continuum from the cube: {exc}") from exc
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"# continuum-subtracted cube (order {contsub_order}): "
                    f"{fit['line_fraction']:.1%} of the fitted pixels' channels clipped as line\n"
                )
    if after_images is not None:
        after_images()
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise fail(f"WSClean exited 0 but wrote no {', '.join(missing)} under {out}", 0)
    # The cubes got the cards as they were stacked, and what was made from them
    # copied them; WSClean's own images (the continuum's) get them here, before
    # the primary-beam correction copies them on.
    provenance = task.header()
    for path in sorted(out.glob("*.fits")):
        try:
            stamp_fits(path, provenance)
        except (OSError, ValueError) as exc:  # the images stand without it
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"# provenance NOT written into {path.name}: {exc}\n")
    if pb_dish_diameter:
        try:
            corrected = pb_correct_task(out, pb_dish_diameter)
        except Exception as exc:
            raise fail(f"Could not primary-beam correct the images: {exc}") from exc
        with log_path.open("a", encoding="utf-8") as log:
            for name, info in corrected.items():
                log.write(f"# primary-beam corrected {name}: {pb_note(info)}\n")

    elapsed = time.time() - started
    write_imaging_marker(
        out,
        ms_path=task.ms_path,
        spw=task.spw,
        command=commands[0] if len(commands) == 1 else None,
        commands=commands if len(commands) > 1 else None,
        seconds=elapsed,
        images=required,
    )
    first = out / required[0]
    return {
        "ms": task.ms_path,
        "spw": int(task.spw),
        "output": str(out),
        "image": str(first),
        "seconds": round(elapsed, 1),
        "attempts": attempts,
        **({"channels": max(cube_channels.values())} if cube_channels else {}),
    }


def pb_correct_task(
    output_dir: str | os.PathLike[str], dish_diameter: float, cutoff: float | None = None
) -> dict[str, dict[str, Any]]:
    """Primary-beam correct a task's continuum image and image cube, where present.

    Returns ``{image file name: primary_beam_correct's summary}``.
    """
    from .primary_beam import PB_CUTOFF, primary_beam_correct

    out = Path(output_dir)
    corrected: dict[str, dict[str, Any]] = {}
    for image, pb, pbcor in (
        (IMAGE_FILENAME, PB_FILENAME, PBCOR_FILENAME),
        (CUBE_IMAGE_FILENAME, CUBE_PB_FILENAME, CUBE_PBCOR_FILENAME),
    ):
        if (out / image).is_file():
            corrected[image] = primary_beam_correct(
                out / image,
                out / pb,
                out / pbcor,
                dish_diameter,
                cutoff=PB_CUTOFF if cutoff is None else cutoff,
            )
    return corrected


def pb_note(info: Mapping[str, Any]) -> str:
    return (
        f"{info['model']} ({float(info['dish_diameter']):g} m dish), FWHM "
        f'{info["fwhm_arcsec"]}" at the middle channel, {info["channels"]} channel(s), '
        f"{100 * float(info['blanked_fraction']):.1f} % blanked below {info['cutoff']}"
    )


def _subtract_uv_continuum_step(
    ms_path: Path,
    commands: list[list[str]],
    *,
    has_continuum: bool,
    order: int,
    task: TaskRun,
    log,
) -> tuple[list[list[str]], dict[str, object] | None]:
    """Run ``subtract_uv_continuum`` and point the commands at its columns.

    Returns the commands and the fit's result (None when it was skipped).
    """
    try:
        fit = subtract_uv_continuum_subprocess(
            ms_path, order=order, timeout=task.timeout, log_path=log.name
        )
    except Exception as exc:
        raise task.fail(f"Could not subtract the continuum from the visibilities: {exc}") from exc
    if fit.get("skipped"):
        log.write(
            f"# continuum NOT subtracted from the visibilities: {fit['skipped']}; "
            "the cube keeps the continuum\n"
        )
        log.flush()
        return commands, None
    log.write(
        f"# continuum subtracted from {fit['source_column']} (order "
        f"{fit['order']}, {fit['seconds']} s): {fit['line_free_channels']} of "
        f"{fit['usable_channels']} channels line-free ({fit['channels']} in the window), "
        "lines in channels "
        f"{fit['line_ranges'] or 'none'}\n"
    )
    log.flush()
    task.note(
        f"continuum subtracted from the visibilities' {fit['source_column']} (order "
        f"{fit['order']}): {fit['line_free_channels']} of {fit['usable_channels']} "
        f"channels line-free, lines in channels {fit['line_ranges'] or 'none'}"
    )
    # Every continuum pass (two with continuum_restart), then the cube.
    continuum = (
        [_with_data_column(c, CONTINUUM_COLUMN) for c in commands[:-1]] if has_continuum else []
    )
    return [*continuum, _with_data_column(commands[-1], LINE_COLUMN)], fit
