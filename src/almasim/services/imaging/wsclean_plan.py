"""Imaging tasks and WSClean command lines from the geometry rows (MADIV-28 T3.6).

Moved out of ``cli/image.py``: the cube channel choice, the per-row command
arguments, the task directories and the cube plan.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from almasim.services.imaging.member_plan import is_member_row, member_task_name, member_windows
from almasim.services.imaging.ms_geometry import (
    SPECTRAL_COLUMNS,
    image_override,
    image_pixels,
)
from almasim.services.imaging.provenance import Provenance, find_ous_index
from almasim.services.instrument.constants import SPEED_OF_LIGHT_M_S


@dataclass(frozen=True)
class CubeOptions:
    """How ``cube_channels`` picks the number of output channels.

    * ``spectral_sampling``: output channels per spectral resolution element,
      the spectral counterpart of ``beam_sampling``. 2 (Nyquist) keeps the
      native channels of a Hanning-smoothed window.
    * ``velocity_resolution_km_s``: never make channels narrower than this
      (0: no limit).
    * ``max_cube_gb``: largest cube, in GB of 32-bit pixels, per product
      (image, residual ...); wide images get fewer channels.
    """

    spectral_sampling: float = 2.0
    velocity_resolution_km_s: float = 0.0
    max_cube_gb: float = 4.0


def _largest_divisor_at_most(n: int, limit: int) -> int:
    """The largest divisor of ``n`` not above ``limit`` (at least 1)."""
    for candidate in range(max(min(int(limit), int(n)), 1), 0, -1):
        if n % candidate == 0:
            return candidate
    return 1


def cube_channels(row: pd.Series, n_pixels: int, options: CubeOptions) -> int:
    """Output channels (WSClean ``-channels-out``) for one spectral window.

    The channel width is the widest of: the native channel, the spectral
    resolution divided by ``spectral_sampling``, and ``velocity_resolution``
    at the window's reference frequency. The count it gives is then capped so
    that an ``n_pixels`` x ``n_pixels`` cube stays below ``max_cube_gb``, and
    rounded down to a divisor of the native channel count, so every output
    channel averages the same number of native channels and the cube has one
    channel width.

    A row with the pipeline's cube header (``image_n_channels`` of an
    ``image_specmode`` ``cube``, from ``image fetch-headers``) takes the
    channel averaging tclean imaged with instead of the one the options give:
    the native count over the whole number of native channels per tclean
    channel. tclean regrids the window (its cube often has a few channels
    fewer than the window), which WSClean does not, so the count itself is not
    taken. The size cap and the divisor rule hold as they do without a header.
    """
    missing = [column for column in SPECTRAL_COLUMNS if column not in row.index]
    if missing:
        raise ValueError(
            f"Cube imaging needs the spectral columns {', '.join(missing)}; "
            "rerun compute-parameters or use add_spectral_columns"
        )
    n_native = int(row["n_channels"])
    width = float(row["channel_width_hz"])
    resolution = float(row["spectral_resolution_hz"]) or width
    target_width = max(width, resolution / options.spectral_sampling)
    if options.velocity_resolution_km_s > 0:
        velocity_width = (options.velocity_resolution_km_s * 1e3 / SPEED_OF_LIGHT_M_S) * float(
            row["reference_frequency"]
        )
        target_width = max(target_width, velocity_width)
    wanted = max(int(math.floor(n_native * width / target_width + 1e-9)), 1)
    image_channels = _image_cube_channels(row)
    if image_channels is not None:
        wanted = max(n_native // max(round(n_native / image_channels), 1), 1)
    plane_bytes = 4 * n_pixels * n_pixels
    affordable = (
        wanted
        if math.isinf(options.max_cube_gb)
        else max(int(options.max_cube_gb * 1e9 // plane_bytes), 1)
    )
    return _largest_divisor_at_most(n_native, min(wanted, affordable))


def _image_niter(row: pd.Series) -> int:
    """WSClean's ``-niter``: tclean's iteration limit of the image, or 100000.

    The pipeline delivers an image it found nothing to clean in with
    ``niter = 0``; that is its finding on its own image, not a limit, so the
    default applies and the thresholds decide where the clean stops.
    """
    try:
        niter = int(float(image_override(row, "image_niter")))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 100000
    return niter if niter > 0 else 100000


def _image_cube_channels(row: pd.Series) -> int | None:
    """The channels of the pipeline's cube of this window, when the row has them."""
    if str(image_override(row, "image_specmode") or "").strip().lower() != "cube":
        return None
    channels = image_override(row, "image_n_channels")
    try:
        count = int(float(channels)) if channels is not None else 0
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def imaging_parameter_to_command_arg(
    imaging_parameters: pd.Series,
    fov_fraction: float,
    beam_sampling: float,
    auto_threshold: float | None = None,
    auto_mask: float | None = None,
    cube: CubeOptions | None = None,
    min_beams_across: float = 0.0,
    multiscale: bool = False,
    join_channels: bool = False,
) -> list[str]:
    """WSClean geometry and deconvolution flags for one spectral window.

    ``auto_threshold`` / ``auto_mask`` (in units of the residual noise) give
    CLEAN a stopping point; without them ``-niter`` is the only limit and every
    task cleans into the noise until the iteration cap.

    With ``cube`` the window is imaged as a spectral cube of ``cube_channels``
    channels (``-channels-out``, each channel cleaned on its own); without it,
    as one continuum plane. ``multiscale`` cleans with WSClean's multi-scale
    algorithm, which models extended emission that delta components leave
    in the residual. ``join_channels`` makes a cube find its components on
    the sum of the channels (WSClean ``-join-channels``) and fit them per
    channel, so emission too faint to clean in one channel is still cleaned.
    """
    spw = imaging_parameters["spectral_window_id"]
    n_pixels = image_pixels(imaging_parameters, fov_fraction, beam_sampling, min_beams_across)
    # An image-derived cell (image fetch-headers) replaces the MS-derived one.
    cell = image_override(imaging_parameters, "image_cell_arcsec")
    cell_arcsec = (
        float(cell)
        if cell is not None
        else float(imaging_parameters["synthetized_beam_size"]) / beam_sampling
    )
    niter = _image_niter(imaging_parameters)
    threshold_jy = image_override(imaging_parameters, "image_threshold_jy")
    cmd_args = [
        "-scale",
        f"{cell_arcsec}asec",
        "-size",
        str(n_pixels),
        str(n_pixels),
        "-spws",
        str(spw),
        "-mgain",
        "0.85",
        "-niter",
        str(niter),
        "-pol",
        "I",
        "-make-psf",
        "-weight",
        *_weight_arguments(imaging_parameters),
    ]
    cmd_args += _abs_threshold_arg(threshold_jy)
    phasecenter = _phasecenter_arguments(imaging_parameters)
    if phasecenter is not None:
        cmd_args += ["-shift", *phasecenter]
    if auto_mask is not None and auto_mask > 0:
        cmd_args += ["-auto-mask", str(auto_mask)]
    if auto_threshold is not None and auto_threshold > 0:
        cmd_args += ["-auto-threshold", str(auto_threshold)]
    if multiscale:
        cmd_args.append("-multiscale")
    if cube is not None:
        channels = cube_channels(imaging_parameters, n_pixels, cube)
        if channels > 1:
            cmd_args += ["-channels-out", str(channels)]
            if join_channels:
                cmd_args.append("-join-channels")
    return cmd_args


def _weight_arguments(row: pd.Series) -> list[str]:
    """WSClean's ``-weight`` values: the image's tclean weighting, or Briggs 0.5.

    tclean's ``briggs`` and ``briggsbwtaper`` are WSClean's ``briggs <robust>``;
    ``natural`` and ``uniform`` take no robust value. Any other tclean scheme
    has no WSClean counterpart and images with the default.
    """
    weighting = str(image_override(row, "image_weighting") or "briggs").strip().lower()
    if weighting in {"natural", "uniform"}:
        return [weighting]
    robust = image_override(row, "image_robust") if weighting.startswith("briggs") else None
    try:
        value = float(robust) if robust is not None else 0.5
    except (TypeError, ValueError):
        value = 0.5
    return ["briggs", str(value)]


def _abs_threshold_arg(threshold_jy: object) -> list[str]:
    """``-abs-threshold`` flag from an image-derived threshold, or ``[]``."""
    try:
        value = float(threshold_jy)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    # NaN is not above zero either.
    return ["-abs-threshold", f"{value:g}Jy"] if value > 0 else []


def _phasecenter_arguments(row: pd.Series) -> tuple[str, str] | None:
    """``-shift`` arguments when the image's phase centre differs from the MS's.

    WSClean images at the MeasurementSet's phase centre; the pipeline's image
    header records the one tclean used. When they differ (a mosaic, say), the
    shift aligns the two. Both are read as degrees and formatted as WSClean's
    ``-shift`` expects.
    """
    image_ra = image_override(row, "phasecenter_ra_deg")
    image_dec = image_override(row, "phasecenter_dec_deg")
    ms_ra = image_override(row, "ms_phasecenter_ra_deg")
    ms_dec = image_override(row, "ms_phasecenter_dec_deg")
    if image_ra is None or image_dec is None or ms_ra is None or ms_dec is None:
        return None
    # The shorter way round in RA: 359.999 and 0.001 degrees are 7 arcsec apart.
    delta_ra = (float(image_ra) - float(ms_ra) + 180.0) % 360.0 - 180.0
    separation = math.hypot(
        delta_ra * math.cos(math.radians(float(image_dec))),
        float(image_dec) - float(ms_dec),
    )
    if separation * 3600.0 < 1.0:  # below an arcsecond: the MS centre is the same
        return None
    return _format_ra(float(image_ra)), _format_dec(float(image_dec))


def _format_ra(ra_deg: float) -> str:
    """``213.2914`` as ``14:13:09.936`` (WSClean's ``-shift`` RA format)."""
    # Whole milliseconds first, so that 59.9996 s carries into the minutes.
    milliseconds = round((ra_deg % 360.0) / 15.0 * 3_600_000) % (24 * 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{milliseconds / 1000:06.3f}"


def _format_dec(dec_deg: float) -> str:
    """``-65.3392`` as ``-65.20.21.12`` (WSClean's ``-shift`` Dec format)."""
    sign = "-" if dec_deg < 0 else "+"
    # Whole hundredths of an arcsecond first, so that 59.996" carries.
    centiarcsec = round(abs(dec_deg) * 360_000)
    minutes, centiarcsec = divmod(centiarcsec, 6_000)
    degrees, minutes = divmod(minutes, 60)
    return f"{sign}{degrees:02d}.{minutes:02d}.{centiarcsec / 100:05.2f}"


@dataclass(frozen=True)
class ImagingTask:
    """One imaging task: one MeasurementSet, one spectral window (and field).

    ``command`` makes the continuum image (empty when only a cube is made);
    ``cube_command``, when set, the spectral cube. A member task
    (``member_parameters``) lists every execution block's MS, window and fields
    in ``member_windows``; ``ms_path`` and ``spw`` are the earliest one's.
    """

    label: str
    ms_path: Path
    spw: int
    output_dir: Path
    command: list[str]
    field_ids: tuple[int, ...] = ()
    cube_command: list[str] | None = None
    dish_diameter: float | None = None
    provenance: Provenance | None = None
    member_windows: tuple[tuple[Path, int, tuple[int, ...]], ...] = ()
    # An aggregate continuum task: the windows it images together, each with
    # its own task's values (``aggregate_window``); ``command`` is the joint one.
    aggregate: tuple[dict[str, Any], ...] = ()


def aggregate_window(task: ImagingTask) -> dict[str, Any]:
    """A window task as plain values, for the aggregate task that images it too."""
    return {
        "ms_path": str(task.ms_path),
        "spw": int(task.spw),
        "field_ids": list(task.field_ids),
        "command": list(task.command),
        "member_windows": [(str(ms), spw, list(f)) for ms, spw, f in task.member_windows],
    }


def _field_arguments(row: pd.Series) -> list[str]:
    """``-field`` selection from the ``target_field_ids`` column, if the CSV has one."""
    value = row.get("target_field_ids") if hasattr(row, "get") else None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.endswith(".0"):  # a single id that pandas read as a float
        text = text[:-2]
    return ["-field", text]


def row_dish_diameter(row: pd.Series) -> float | None:
    """The dish diameter (m) behind a parameter row's ``fov_per_frequency``.

    ``compute_imaging_parameters`` sets it to the primary beam's half-power
    width for the smallest dish, so D comes back exactly (12 or 7 m for ALMA). None when
    the row lacks the field of view or the frequency.
    """
    from almasim.services.imaging.primary_beam import dish_diameter_from_fov

    try:
        diameter = dish_diameter_from_fov(
            float(row["fov_per_frequency"]), float(row["reference_frequency"])
        )
    except (KeyError, TypeError, ValueError):
        return None
    return round(diameter, 2) if math.isfinite(diameter) else None


def row_provenance(
    row: pd.Series,
    ous_index_csv: str | Path | None = None,
    indexes: dict[Path | None, dict[str, dict[str, str]]] | None = None,
) -> Provenance:
    """A parameter row's provenance: its MSs, window and field, and the OUSs of its EBs.

    A member row (``member_parameters``) has the MSs of all its EBs, any other
    row its one MS. The OUSs come from ``ous_index_csv``, or from the
    ``ous_index.csv`` nearest above the (first) MS when it is None (none: the
    OUSs are left out). ``indexes`` keeps the indexes read, by path, across the
    rows of one run.
    """
    from almasim.services.archive.delivery import read_ous_index

    ms = Path(row["filename"]).expanduser().absolute()
    ms_paths = [m for m, _, _ in member_windows(row)] or [ms]
    path = Path(ous_index_csv) if ous_index_csv else find_ous_index(ms)
    indexes = {} if indexes is None else indexes
    if path not in indexes:
        indexes[path] = read_ous_index(path) if path is not None else {}
    field = row.get("field_name") if hasattr(row, "get") else None
    if field is None or (isinstance(field, float) and math.isnan(field)):
        field = ""
    return Provenance.for_ms(
        ms_paths,
        indexes[path],
        spw=int(row["spectral_window_id"]),
        field=str(field),
    )


def delivery_members(output_directory: Path) -> dict[str, str] | None:
    """``{eb: member folder}`` when ``output_directory`` is a delivery tree's root, else None.

    A delivery root holds the tree's ``ous_index.csv`` (``write_ous_index``);
    the member folders are relative to it.
    """
    from almasim.services.archive.delivery import INDEX_FILENAME

    index = Path(output_directory) / INDEX_FILENAME
    try:
        stamp = index.stat().st_mtime_ns
    except OSError:
        return None
    return _read_delivery_members(index, stamp)


@functools.lru_cache(maxsize=4)
def _read_delivery_members(index: Path, stamp: int) -> dict[str, str]:
    """The index's member folders by EB, read once per version of the file (``stamp``)."""
    from almasim.services.archive.delivery import read_ous_index

    return {eb: row["member_folder"] for eb, row in read_ous_index(index).items()}


def task_directory(output_directory: Path, row: pd.Series) -> tuple[Path, tuple[int, ...]]:
    """A parameter row's task directory under ``output_directory`` and its field ids.

    One field per row gets its own directory and markers, so the fields of
    one window never share an image; rows without a field, or with a list
    (CSVs from before per-field rows), keep SPW-<n>. A member row's directory is
    ``<member folder>/SPW-<n>_<field name>`` (``member_task_name``), its field 0.

    When ``output_directory`` is the root of a delivery tree (it holds
    ``ous_index.csv``) the tasks go into the members' own folders:
    ``<member folder>/images/SPW-<n>_<field name>`` for a member row and
    ``<member folder>/images/<ms>/SPW-<n>...`` for the MS of one of its EBs. An MS
    the index does not list keeps ``<output>/<ms>/SPW-<n>...``.
    """
    from almasim.services.archive.delivery import IMAGES_FOLDER

    output_directory = Path(output_directory)
    members = delivery_members(output_directory)
    if is_member_row(row):
        folder = str(row["member_folder"])
        if members is not None:
            return output_directory / folder / IMAGES_FOLDER / member_task_name(row), (0,)
        return output_directory / Path(folder).name / member_task_name(row), (0,)
    input_filename = Path(row["filename"]).expanduser().absolute()
    spw = int(row["spectral_window_id"])
    field_args = _field_arguments(row)
    field_ids = tuple(int(f) for f in field_args[1].split(",")) if field_args else ()
    single_field = field_ids[0] if len(field_ids) == 1 else None
    task_name = f"SPW-{spw}" if single_field is None else f"SPW-{spw}_FIELD-{single_field}"
    parent = output_directory
    eb = input_filename.name.split(".ms", 1)[0]
    if members is not None and eb in members:
        parent = output_directory / members[eb] / IMAGES_FOLDER
    return parent / input_filename.stem / task_name, field_ids


def _check_member_rows(parameters: pd.DataFrame, single_window: bool) -> None:
    if not single_window and any(is_member_row(row) for _, row in parameters.iterrows()):
        raise ValueError("A member is imaged from its combined single-window MS")


def _task_label(outdir: Path, ms: Path, spw: int, single_field: int | None, member: bool) -> str:
    """``<ms stem>_<spw>[_f<field>]``, or ``<member folder>_<task folder>`` for a member."""
    if member:
        folder = outdir.parent
        if folder.name == "images":  # <member folder>/images/<task> in a delivery tree
            folder = folder.parent
        return f"{folder.name}_{outdir.name}"
    return f"{ms.stem}_{spw}" + ("" if single_field is None else f"_f{single_field}")


def build_imaging_tasks(  # noqa: C901
    parameters: pd.DataFrame,
    output_directory: Path,
    *,
    fov_fraction: float,
    beam_sampling: float,
    num_cores: int,
    max_cores_per_node: int,
    wsclean_bin: str = "wsclean",
    overwrite_outputs: bool = False,
    trust_existing_images: bool = False,
    auto_threshold: float | None = None,
    auto_mask: float | None = None,
    update_model: bool = False,
    task_memory_gb: float | None = None,
    scratch_dir: str | None = None,
    single_window: bool = True,
    continuum: bool = True,
    cube: CubeOptions | None = None,
    cube_products: Sequence[str] | None = None,
    cube_reorder: bool = True,
    min_beams_across: float = 0.0,
    multiscale: bool = False,
    join_channels: bool = False,
    ous_index_csv: str | Path | None = None,
    aggregate_continuum: bool = False,
) -> tuple[list[ImagingTask], int]:
    """Turn parameter rows into WSClean tasks, skipping the ones already done.

    A task makes the continuum image (``continuum``, ``wsclean-*.fits``)
    and/or, with ``cube``, a spectral cube of ``cube_channels`` channels
    (``wsclean-cube-*.fits``); both runs share the task's single-window MS.
    The cube gets a dirty image only when ``cube_products`` asks for it.

    The cube run lets WSClean reorder (``cube_reorder``) even on a
    single-window MS: with ``-no-reorder`` it reads the whole MS about four
    times per output channel, and a 480-channel Band 1 cube
    (uid___A002_X1142854_X18e99 SPW 25) took about 2 h against 4.5 min
    reordered, with the same images (2026-10-13). The 3.7 reordering crash
    needs several windows in the MS; the continuum run keeps ``-no-reorder``.

    Returns ``(tasks, skipped)``. A task is done when its ``SPW-<n>.done``
    marker and every image it makes exist, so a cube run over a folder of
    continuum images (``continuum=False``) redoes only the missing cubes;
    ``trust_existing_images`` also accepts a continuum image produced before
    markers existed and writes its marker. ``overwrite_outputs`` redoes
    everything.

    With ``aggregate_continuum`` (needs ``continuum`` and ``single_window``)
    every field imaged in two or more windows also gets a task that images
    their continuum together (``almasim.services.imaging.aggregate``), in
    ``CONT[_<field>]`` beside the windows' ``SPW-<n>[_<field>]``: on the
    finest of the windows' cells, over the widest of their fields. It is done
    when its marker and its image exist, whatever its windows' tasks are.

    Each task carries its data's provenance (``row_provenance``) for the
    products' FITS headers: the OUSs from ``ous_index_csv``, or from the
    ``ous_index.csv`` nearest above each MS when it is None.
    """
    from almasim.services.imaging.aggregate import aggregate_directory
    from almasim.services.imaging.markers import is_imaging_complete, write_imaging_marker
    from almasim.services.imaging.products import (
        CUBE_IMAGE_FILENAME,
        CUBE_NAME,
        IMAGE_FILENAME,
        SCRATCH_PLACEHOLDER,
        SINGLE_WINDOW_PLACEHOLDER,
    )
    from almasim.services.imaging.wsclean.command import WSCleanCommand

    if not continuum and cube is None:
        raise ValueError("Nothing to image: neither the continuum nor a cube was requested")
    if aggregate_continuum and not (continuum and single_window):
        raise ValueError("an aggregate continuum needs the continuum and single-window MSs")
    required = (IMAGE_FILENAME,) * continuum + (CUBE_IMAGE_FILENAME,) * (cube is not None)
    tasks: list[ImagingTask] = []
    skipped = 0
    # WSClean's -mem is a PERCENTAGE of system memory. Passing the fraction
    # (0.1 for 10 cores of 96) limited every task to 0.1 % ≈ 0.4 GB and made
    # WSClean segfault on the first gridding pass.
    mem_percent = min(100.0, 100.0 * num_cores / max_cores_per_node)
    # WSClean runs with its task directory as cwd, so a relative path in the
    # command resolves against that directory, not the caller's: a run started
    # with output "ms-imaged-flagged" looked for its single-window MS under
    # .../SPW-5/ms-imaged-flagged/... and every task failed (2026-10-09).
    output_directory = Path(output_directory).expanduser().absolute()
    indexes: dict[Path | None, dict[str, dict[str, str]]] = {}
    groups: dict[Path, list[pd.Series]] = {}
    _check_member_rows(parameters, single_window)

    def window_task(row: pd.Series) -> ImagingTask:
        """The task of one parameter row (its continuum and, with ``cube``, its cube)."""
        input_filename = Path(row["filename"]).expanduser().absolute()
        spw = int(row["spectral_window_id"])
        outdir, field_ids = task_directory(output_directory, row)
        single_field = field_ids[0] if len(field_ids) == 1 else None

        def wsclean_command(name: str, with_cube: bool) -> list[str]:
            command_args = imaging_parameter_to_command_arg(
                row,
                fov_fraction,
                beam_sampling,
                auto_threshold=auto_threshold,
                auto_mask=auto_mask,
                cube=cube if with_cube else None,
                min_beams_across=min_beams_across,
                multiscale=multiscale,
                join_channels=join_channels,
            )
            if with_cube and cube_products is not None and "dirty" not in cube_products:
                command_args.append("-no-dirty")
            command = WSCleanCommand(
                wsclean_bin,
                (
                    "-name",
                    str(outdir / name),
                    "-j",
                    str(num_cores),
                    *command_args,
                    # An absolute limit when the caller knows the worker's
                    # allocation (WSClean's -mem is a share of the *node*, which
                    # can be far more than the Slurm job may use); otherwise the
                    # node share.
                    *(
                        ["-abs-mem", f"{task_memory_gb:g}"]
                        if task_memory_gb
                        else ["-mem", f"{mem_percent:g}"]
                    ),
                    # Reordered visibilities go next to the task output (or into a
                    # per-task scratch directory), not next to the MS: several
                    # spectral windows of one MS run at the same time and would
                    # otherwise clobber each other's <ms>-part*.tmp files.
                    "-temp-dir",
                    SCRATCH_PLACEHOLDER if scratch_dir else str(outdir),
                    # Writing the model back into MODEL_DATA makes concurrent tasks
                    # write the same MS from different nodes and grows every
                    # calibrated MS; the model image is on disk anyway.
                    "-update-model-required" if update_model else "-no-update-model-required",
                    # -field is needed on the single-window MS too: WSClean images
                    # field 0 by default and the extracted MS holds the target
                    # fields under their original ids.
                    *_field_arguments(row),
                ),
                str(input_filename),
            )
            if single_window:
                # The wrapper hands WSClean a per-task MS holding only this
                # window and the target fields, so no -spws selection and no
                # reordering (WSClean 3.7's reordering reads out of bounds on
                # multi-window ALMA sets; see extract_single_window_ms).
                command = replace(command.without("-spws"), ms=SINGLE_WINDOW_PLACEHOLDER)
                if not (with_cube and cube_reorder):
                    command = command.with_option("-no-reorder")
            return command.argv()

        return ImagingTask(
            label=_task_label(outdir, input_filename, spw, single_field, is_member_row(row)),
            ms_path=input_filename,
            spw=spw,
            output_dir=outdir,
            command=wsclean_command("wsclean", False) if continuum else [],
            field_ids=field_ids,
            cube_command=wsclean_command(CUBE_NAME, True) if cube is not None else None,
            dish_diameter=row_dish_diameter(row),
            provenance=row_provenance(row, ous_index_csv, indexes),
            member_windows=member_windows(row),
        )

    for _, row in parameters.iterrows():
        input_filename = Path(row["filename"]).expanduser().absolute()
        spw = int(row["spectral_window_id"])
        outdir, _ = task_directory(output_directory, row)
        if aggregate_continuum:
            # Before the skip: a window already imaged is still in the aggregate.
            groups.setdefault(aggregate_directory(outdir, spw), []).append(row)
        if not overwrite_outputs:
            if is_imaging_complete(outdir, required):
                skipped += 1
                continue
            if (
                trust_existing_images
                and required == (IMAGE_FILENAME,)
                and (outdir / IMAGE_FILENAME).is_file()
            ):
                write_imaging_marker(outdir, ms_path=str(input_filename), spw=spw)
                skipped += 1
                continue

        tasks.append(window_task(row))
    for outdir, rows in groups.items():
        if len(rows) < 2:
            continue  # one window: its own image is the field's continuum
        if not overwrite_outputs and is_imaging_complete(outdir, (IMAGE_FILENAME,)):
            skipped += 1
            continue
        tasks.append(_aggregate_task(outdir, [window_task(row) for row in rows], rows))
    return tasks, skipped


def _aggregate_task(
    outdir: Path, windows: Sequence[ImagingTask], rows: Sequence[pd.Series]
) -> ImagingTask:
    """The task imaging ``windows`` (one field's) together into ``outdir``: the
    first window's continuum command on the finest cell of them all, wide
    enough for the widest of their images."""
    from almasim.services.imaging.wsclean.command import WSCleanCommand

    grids = []
    for window in windows:
        command = WSCleanCommand.parse(window.command)
        cell = float(str(command.value("-scale")).removesuffix("asec"))
        grids.append((cell, cell * int(str(command.value("-size")))))
    cell = min(cell for cell, _ in grids)
    pixels = math.ceil(max(extent for _, extent in grids) / cell)
    pixels += pixels % 2
    first = windows[0]
    joint = WSCleanCommand.parse(first.command)
    options = list(joint.options)
    at = options.index("-size")
    options[at + 1 : at + 3] = [str(pixels), str(pixels)]
    joint = replace(joint, options=tuple(options))
    joint = joint.with_option("-scale", f"{cell}asec").with_option("-name", outdir / "wsclean")
    diameters = [w.dish_diameter for w in windows if w.dish_diameter]
    return ImagingTask(
        label=_task_label(outdir, first.ms_path, first.spw, None, True)
        if is_member_row(rows[0])
        else f"{first.ms_path.stem}_{outdir.name}",
        ms_path=first.ms_path,
        spw=first.spw,
        output_dir=outdir,
        command=joint.argv(),
        field_ids=first.field_ids,
        dish_diameter=min(diameters) if diameters else None,
        provenance=first.provenance,
        member_windows=first.member_windows,
        aggregate=tuple(aggregate_window(w) for w in windows),
    )


CUBE_PRODUCTS = ("image", "residual", "psf", "model", "dirty")


def cube_plan(
    parameters: pd.DataFrame,
    fov_fraction: float,
    beam_sampling: float,
    options: CubeOptions,
    min_beams_across: float = 0.0,
) -> pd.DataFrame:
    """Per row: image side, native and cube channels, and one cube's size in GB.

    ``size_limited`` marks the rows that ``max_cube_gb`` gave fewer channels
    than the spectral sampling asked for.
    """
    unlimited = replace(options, max_cube_gb=math.inf)
    pixels = parameters.apply(
        lambda row: image_pixels(row, fov_fraction, beam_sampling, min_beams_across), axis=1
    )
    channels, wanted = [], []
    for (_, row), n in zip(parameters.iterrows(), pixels, strict=True):
        channels.append(cube_channels(row, int(n), options))
        wanted.append(cube_channels(row, int(n), unlimited))
    plan = pd.DataFrame(
        {
            "n_pixels": pixels.astype(int).to_numpy(),
            "n_channels": parameters["n_channels"].astype(int).to_numpy(),
            "cube_channels": channels,
            "size_limited": np.array(channels) < np.array(wanted),
        },
        index=parameters.index,
    )
    plan["cube_gb"] = 4 * plan["n_pixels"] ** 2 * plan["cube_channels"] / 1e9
    return plan
