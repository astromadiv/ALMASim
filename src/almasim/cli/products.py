"""Product resolution and download commands for the ALMASim CLI."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any

import typer
from tqdm.auto import tqdm

from ..services.archive.pipeline import (  # noqa: F401  (moved; re-exported)
    _NO_PROGRESS_TIMEOUT_S,
    _PROGRESS_HEARTBEAT_S,
    StageError,
    StageUsageError,
    _archive_done_marker,
    _calibrate_single_uid,
    _casa_bundled_lib_dir,
    _compute_jobs_with_progress,
    _extract_asdm_uids_from_download_root,
    _extract_single_archive,
    _extract_uids_from_raw_ms_root,
    _find_archives,
    _one_uid_per_worker,
    _partition_completed_calibrations,
    _preflight_casa_data,
    _run_calibrate_jobs,
    _run_extract_jobs,
    _run_parallel_archive_jobs,
    _run_uid_stage,
    _run_unpack_jobs,
    _safe_extract_tar_archive,
    _safe_uid,
    _stage_child_env,
    _stage_log_path,
    _unpack_single_uid,
    _worker_failure,
    create_backend,
)
from ..services.compute.progress import (
    TaskFailure,
    report_failures,
)
from ..services.download import (
    MAX_PARALLEL_PER_MIRROR,
    MAX_PARALLEL_TOTAL,
    PRODUCT_TYPES,
    download_products,
    filter_products,
    format_bytes,
    load_products_csv,
    resolve_products,
    save_products_csv,
)
from ._options import (
    Backend,
    PostprocessBackend,
    SchedulerHost,
    SlurmCores,
    SlurmMemory,
    SlurmOptions,
    SlurmProject,
    SlurmQueue,
    SlurmWalltime,
)
from ._shared import dedupe_keep_order, default_output_path, fail, split_csv_values, usage_error


@contextmanager
def _stage_errors() -> Iterator[None]:
    """Map the archive stages' input errors to the command's exit codes."""
    try:
        yield
    except StageUsageError as exc:
        usage_error(str(exc))
    except StageError as exc:
        fail(str(exc))


# The product filter's choices as an enum, so typer checks them and the
# console's catalogue lists them (MADIV-28 S2.2). PRODUCT_TYPES is a set: sorted.
ProductFilter = StrEnum(  # type: ignore[misc]
    "ProductFilter", {name: name for name in sorted(PRODUCT_TYPES)}
)

products_app = typer.Typer(
    help="Data product resolution and download commands.",
    no_args_is_help=True,
)


def filter_products_by_qa0(*args, **kwargs):
    from ..services.archive.qa0 import filter_products_by_qa0 as _filter_products_by_qa0

    return _filter_products_by_qa0(*args, **kwargs)


def _report_stage_failures(stage_label: str, failures: list[TaskFailure]) -> None:
    report_failures(stage_label, failures, unit="UID(s)", echo=lambda t: typer.echo(t, err=True))


def _exit_if_failures(stage_label: str, failures: list[TaskFailure]) -> None:
    """Print a failure summary and exit non-zero once every UID has been attempted."""
    if not failures:
        return
    _report_stage_failures(stage_label, failures)
    raise typer.Exit(code=1)


def _download_products_with_progress(
    products: list[Any],
    destination: Path,
    **kwargs: Any,
):
    total_known_bytes = sum(max(int(product.content_length), 0) for product in products)
    total_files = len(products)
    progress_total = total_known_bytes if total_known_bytes > 0 else max(total_files, 1)
    progress_unit = "B" if total_known_bytes > 0 else "file"
    progress_kwargs = {"unit_scale": True, "unit_divisor": 1000} if total_known_bytes > 0 else {}
    progress_lock = Lock()
    previous_bytes: dict[str, int] = {}
    previous_states: dict[str, str] = {}
    completed_files = 0

    with tqdm(
        total=progress_total,
        desc="Downloading products",
        unit=progress_unit,
        leave=True,
        **progress_kwargs,
    ) as progress_bar:
        progress_bar.set_postfix_str(f"files 0/{total_files}")

        def update_callback(file_status: Any) -> None:
            nonlocal completed_files

            key = f"{file_status.access_url}|{file_status.filename}"
            with progress_lock:
                current_bytes = max(int(file_status.bytes_downloaded), 0)
                previous = previous_bytes.get(key, 0)
                if total_known_bytes > 0 and current_bytes > previous:
                    progress_bar.update(current_bytes - previous)
                previous_bytes[key] = max(previous, current_bytes)

                status = str(file_status.status)
                previous_status = previous_states.get(key)
                if status in {"completed", "failed", "cancelled"} and previous_status not in {
                    "completed",
                    "failed",
                    "cancelled",
                }:
                    completed_files += 1
                    if total_known_bytes <= 0:
                        progress_bar.update(1)
                previous_states[key] = status
                progress_bar.set_postfix_str(f"files {completed_files}/{total_files}")

        return download_products(
            products,
            destination,
            update_callback=update_callback,
            **kwargs,
        )


def _read_member_uids_from_metadata(
    metadata_csv: Path,
    member_limit: int | None,
) -> list[str]:
    import pandas as pd

    metadata = pd.read_csv(metadata_csv.expanduser().resolve())
    if "member_ous_uid" not in metadata.columns:
        usage_error(f"Metadata CSV does not contain member_ous_uid: {metadata_csv}")
    series = metadata["member_ous_uid"].dropna().astype(str)
    if member_limit is not None:
        series = series.head(member_limit)
    return dedupe_keep_order(series.tolist())


def _parse_uid_options(uids: list[str] | None) -> list[str]:
    """UIDs from repeated or comma-separated options, in order, without duplicates."""
    return dedupe_keep_order(split_csv_values(uids) or [])


def _resolve_products_from_inputs(
    *,
    products_csv: Path | None,
    metadata_csv: Path | None,
    member_ous_uid: list[str] | None,
    member_limit: int | None,
    save_products_csv_path: Path | None,
) -> list[Any]:
    if products_csv is not None:
        loaded = load_products_csv(products_csv)
        typer.echo(f"Loaded products CSV: {products_csv.expanduser().resolve()}")
        return loaded

    member_uids = _parse_uid_options(member_ous_uid)
    if metadata_csv is not None:
        member_uids.extend(_read_member_uids_from_metadata(metadata_csv, member_limit))
    member_uids = dedupe_keep_order([uid for uid in member_uids if uid])
    if not member_uids:
        usage_error("Provide --products-csv, --metadata-csv, or at least one --member-ous-uid.")

    typer.echo(f"Resolving DataLink products for {len(member_uids)} member OUS UID(s)...")
    typer.echo(
        "Using ALMA DataLink services: "
        "ESO (almascience.eso.org), NRAO (almascience.nrao.edu), "
        "NAOJ (almascience.nao.ac.jp)"
    )

    resolved = []
    with typer.progressbar(member_uids, label="Resolving member OUS UIDs") as progress:
        for uid in progress:
            resolved.extend(resolve_products([uid]))

    typer.echo(f"Resolved DataLink rows: {len(resolved)}")
    if not resolved:
        fail("No products were resolved for the requested member_ous_uid values.")

    if save_products_csv_path is not None:
        saved = save_products_csv(resolved, save_products_csv_path)
        typer.echo(f"Saved resolved products CSV: {saved}")
    return resolved


@products_app.command("resolve")
def products_resolve(
    metadata_csv: Path | None = typer.Option(
        None,
        "--metadata-csv",
        help="Metadata CSV containing member_ous_uid rows.",
    ),
    member_ous_uid: list[str] | None = typer.Option(
        None,
        "--member-ous-uid",
        help="Direct member_ous_uid values. Repeat or pass comma-separated values.",
    ),
    member_limit: int | None = typer.Option(
        None,
        "--member-limit",
        min=1,
        help="Max metadata member_ous_uid rows to read (default: unlimited).",
    ),
    save_member_ous_uid_list: Path | None = typer.Option(
        None,
        "--save-member-ous-uid-list",
        help="Optional text file to write extracted member_ous_uid list (one per line).",
    ),
    save_products_csv_path: Path = typer.Option(
        default_output_path("resolved_products.csv"),
        "--save-products-csv",
        help="Destination CSV for resolved DataLink products.",
    ),
) -> None:
    """Extract member_ous_uid values and resolve ALMA DataLink products."""
    member_uids = _parse_uid_options(member_ous_uid)
    if metadata_csv is not None:
        member_uids.extend(_read_member_uids_from_metadata(metadata_csv, member_limit))
    member_uids = dedupe_keep_order([uid for uid in member_uids if uid])
    if not member_uids:
        usage_error(
            "Provide --metadata-csv and/or --member-ous-uid to extract member_ous_uid values."
        )

    typer.echo(f"Extracted member_ous_uid values: {len(member_uids)}")
    if save_member_ous_uid_list is not None:
        uid_path = save_member_ous_uid_list.expanduser().resolve()
        uid_path.parent.mkdir(parents=True, exist_ok=True)
        uid_path.write_text("\n".join(member_uids) + "\n", encoding="utf-8")
        typer.echo(f"Saved member_ous_uid list: {uid_path}")

    products = _resolve_products_from_inputs(
        products_csv=None,
        metadata_csv=None,
        member_ous_uid=member_uids,
        member_limit=None,
        save_products_csv_path=save_products_csv_path,
    )
    typer.echo(f"Resolved products: {len(products)}")


def _update_manifest_measurement_sets(
    manifest_path: str | None, raw_mss: list[str], calibrated_mss: list[str]
) -> None:
    """Record post-processing outputs in the download manifest written earlier."""
    if not manifest_path:
        return
    import json

    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(manifest, dict):
        return
    manifest["raw_measurement_sets"] = list(raw_mss)
    manifest["calibrated_measurement_sets"] = list(calibrated_mss)
    try:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except OSError:
        typer.echo(f"Could not update manifest {path}", err=True)


def _organize_delivery(delivery_root: Path, calibrated_ms_root: Path, raw_ms_root: Path) -> None:
    """Link the calibrated MSs into their member folders and index the tree; never fails the run."""
    from ..services.archive.delivery import organize_delivery

    try:
        organize_delivery(delivery_root, calibrated_ms_root, raw_ms_root, echo=typer.echo)
    except OSError as exc:
        typer.echo(f"Could not organize the delivery tree under {delivery_root}: {exc}", err=True)


_ORGANIZE_HELP = (
    "Link each calibrated MS into its member folder of the ALMA delivery tree "
    "(<project>/science_goal.*/group.*/member.*/calibrated/) and write ous_index.csv "
    "at the root of the tree."
)


def _cleanup_after_download_postprocess(
    *,
    summary: Any,
    destination: Path,
    archive_root: Path,
    calibrated_mss: list[str],
) -> None:
    """Apply --clean-intermediate-files once every UID has been calibrated successfully."""
    from ..services.archive import cleanup_intermediate_calibration_data
    from ..services.download import _cleanup_download_inputs

    calibrated_paths = [Path(path) for path in calibrated_mss]
    cleanup_intermediate_calibration_data(
        archive_root / "raw_ms",
        archive_root / "calibrated_ms",
        calibrated_paths,
        logger_fn=typer.echo,
    )
    _cleanup_download_inputs(
        destination,
        getattr(summary, "files", []) or [],
        getattr(summary, "extracted_files", []) or [],
        protected_roots=[archive_root, *calibrated_paths],
    )


@dataclass(frozen=True)
class DownloadOptions:
    """Every option of ``products download`` (MADIV-28 T3.8)."""

    products_csv: Path | None
    metadata_csv: Path | None
    member_ous_uid: list[str] | None
    member_limit: int | None
    product_filter: str
    skip_qa0_semipass: bool
    qa0_report_root: Path | None
    save_products_csv_path: Path | None
    destination: Path
    max_parallel: int
    extract_tar: bool
    unpack_ms: bool
    generate_calibrated_visibilities: bool
    archive_output_root: Path | None
    casa_data_root: Path | None
    skip_casa_data_update: bool
    clean_intermediate_files: bool
    postprocess_backend: PostprocessBackend
    slurm_queue: SlurmQueue
    slurm_project: SlurmProject
    slurm_walltime: SlurmWalltime
    slurm_cores: SlurmCores
    slurm_memory: SlurmMemory
    slurm_workers: int
    overwrite_archive_outputs: bool
    continue_on_error: bool
    keep_working_copies: bool
    yes: bool


@products_app.command("download")
def products_download(
    products_csv: Path | None = typer.Option(
        None,
        "--products-csv",
        help="Previously resolved products CSV.",
        rich_help_panel="Selection",
    ),
    metadata_csv: Path | None = typer.Option(
        None,
        "--metadata-csv",
        help="Metadata CSV containing member_ous_uid rows to resolve first.",
        rich_help_panel="Selection",
    ),
    member_ous_uid: list[str] | None = typer.Option(
        None,
        "--member-ous-uid",
        help="Direct member_ous_uid values. Repeat or pass comma-separated values.",
        rich_help_panel="Selection",
    ),
    member_limit: int | None = typer.Option(
        None,
        "--member-limit",
        min=1,
        help="Max metadata member_ous_uid rows to read (default: unlimited).",
        rich_help_panel="Selection",
    ),
    product_filter: ProductFilter = typer.Option(
        ProductFilter("all"),
        "--product-filter",
        help="Subset of resolved products to download.",
        case_sensitive=False,
        rich_help_panel="Selection",
    ),
    skip_qa0_semipass: bool = typer.Option(
        False,
        "--skip-qa0-semipass",
        help=(
            "Skip raw ASDMs whose execution block is QA0 SemiPass, read from the "
            "<eb>.qa0_report.pdf in the member's extracted auxiliary delivery. ALMA never "
            "delivers calibration for these, so they can never be calibrated. Download and "
            "extract the auxiliary products first; a raw product without a readable report "
            "is kept."
        ),
        rich_help_panel="Selection",
    ),
    qa0_report_root: Path | None = typer.Option(
        None,
        "--qa0-report-root",
        help=(
            "Root holding the extracted auxiliary deliveries "
            "(<project>/science_goal.*/group.*/member.*/qa/). Defaults to --destination."
        ),
        rich_help_panel="Selection",
    ),
    save_products_csv_path: Path | None = typer.Option(
        default_output_path("resolved_products.csv"),
        "--save-products-csv",
        help="Save resolved products CSV before download.",
        rich_help_panel="Download",
    ),
    destination: Path = typer.Option(
        default_output_path("downloads"),
        "--destination",
        help="Directory for downloaded files.",
        rich_help_panel="Download",
    ),
    max_parallel: int = typer.Option(
        3,
        "--max-parallel",
        min=1,
        help=(
            "Max concurrent downloads across ALMA mirrors; capped at "
            f"{MAX_PARALLEL_TOTAL} ({MAX_PARALLEL_PER_MIRROR} per mirror)."
        ),
        rich_help_panel="Download",
    ),
    extract_tar: bool = typer.Option(
        False,
        "--extract-tar",
        help="Extract downloaded tar/tgz archives.",
        rich_help_panel="Post-processing",
    ),
    unpack_ms: bool = typer.Option(
        False,
        "--unpack-ms",
        help="Import extracted ASDMs into raw MeasurementSets.",
        rich_help_panel="Post-processing",
    ),
    generate_calibrated_visibilities: bool = typer.Option(
        False,
        "--generate-calibrated-visibilities",
        help="Apply delivered calibrations and write calibrated MeasurementSets.",
        rich_help_panel="Post-processing",
    ),
    archive_output_root: Path | None = typer.Option(
        None,
        "--archive-output-root",
        help="Root directory for archive_ms raw_ms/calibrated_ms products.",
        rich_help_panel="Post-processing",
    ),
    casa_data_root: Path | None = typer.Option(
        None,
        "--casa-data-root",
        help="Optional CASA runtime data directory.",
        rich_help_panel="CASA",
    ),
    skip_casa_data_update: bool = typer.Option(
        False,
        "--skip-casa-data-update",
        help="Do not auto-download CASA runtime data if missing.",
        rich_help_panel="CASA",
    ),
    clean_intermediate_files: bool = typer.Option(
        False,
        "--clean-intermediate-files",
        help="Clean downloaded/intermediate raw files after calibrated outputs are created.",
        rich_help_panel="Post-processing",
    ),
    postprocess_backend: PostprocessBackend = Backend.sync,
    slurm_queue: SlurmQueue = "normal",
    slurm_project: SlurmProject = None,
    slurm_walltime: SlurmWalltime = "02:00:00",
    slurm_cores: SlurmCores = 1,
    slurm_memory: SlurmMemory = "4GB",
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=0,
        help="Number of Slurm workers for post-processing. Pass 0 to spawn one worker per UID.",
        rich_help_panel="Cluster",
    ),
    overwrite_archive_outputs: bool = typer.Option(
        False,
        "--overwrite-archive-outputs",
        help="Overwrite existing raw/calibrated MS outputs.",
        rich_help_panel="Post-processing",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help=(
            "Skip UIDs whose processing fails and carry on with the rest, then exit "
            "non-zero with a summary (default). --fail-fast aborts on the first failure."
        ),
        rich_help_panel="Post-processing",
    ),
    keep_working_copies: bool = typer.Option(
        False,
        "--keep-working-copies",
        help=(
            "Keep each UID's working directory (raw MS copy plus caltables, about 1.8x the "
            "raw MS) after its calibrated output is written. By default it is removed."
        ),
        rich_help_panel="Post-processing",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt and start download immediately.",
        rich_help_panel="Download",
    ),
) -> None:
    """Download ALMA products and optionally unpack/calibrate archive data."""
    options = DownloadOptions(**locals())
    selected = _select_download_products(options)
    if options.unpack_ms or options.generate_calibrated_visibilities:
        _download_and_postprocess(options, selected)
    else:
        summary = _download_products_with_progress(
            selected,
            options.destination,
            **_download_kwargs(options, postprocess_in_process=True),
        )
        _print_download_summary(
            summary, summary.raw_measurement_sets, summary.calibrated_measurement_sets
        )


def _select_download_products(o: DownloadOptions) -> list[Any]:
    """Validate the options and return the products to download (exits when there are none)."""
    product_filter = o.product_filter.lower()  # click checked the choice (MADIV-28 S2.2)
    if o.max_parallel > MAX_PARALLEL_TOTAL:
        typer.echo(
            "Requested --max-parallel="
            f"{o.max_parallel} exceeds cap; clamping to {MAX_PARALLEL_TOTAL}."
        )
    products = _resolve_products_from_inputs(
        products_csv=o.products_csv,
        metadata_csv=o.metadata_csv,
        member_ous_uid=o.member_ous_uid,
        member_limit=o.member_limit,
        save_products_csv_path=o.save_products_csv_path,
    )
    filtered = filter_products(products, product_filter)
    if not filtered:
        fail(f"No products matched --product-filter={product_filter}")

    typer.echo(f"Resolved products: {len(products)}")
    if o.skip_qa0_semipass:
        report_root = (
            (o.qa0_report_root if o.qa0_report_root is not None else o.destination)
            .expanduser()
            .resolve()
        )
        qa0_result = filter_products_by_qa0(filtered, report_root)
        typer.echo(
            f"QA0 filter: skipped {len(qa0_result.skipped)} SemiPass raw product(s) "
            f"({format_bytes(qa0_result.bytes_skipped)}); "
            f"{len(qa0_result.unknown)} raw product(s) without a readable QA0 report "
            f"under {report_root} kept."
        )
        filtered = qa0_result.kept
        if not filtered:
            typer.echo("Nothing left to download after the QA0 filter.")
            raise typer.Exit(code=0)

    total_bytes = sum(product.content_length for product in filtered)
    typer.echo(f"Selected for download: {len(filtered)} ({format_bytes(total_bytes)})")
    if not o.yes:
        unknown_sizes = sum(1 for product in filtered if product.content_length <= 0)
        message = (
            f"About to download {len(filtered)} product(s), total size {format_bytes(total_bytes)}"
        )
        if unknown_sizes:
            message += f" ({unknown_sizes} item(s) with unknown size)"
        message += ". Continue?"
        if not typer.confirm(message, default=True):
            typer.echo("Download cancelled.")
            raise typer.Exit(code=0)
    if o.generate_calibrated_visibilities and not o.unpack_ms:
        typer.echo(
            "--generate-calibrated-visibilities without --unpack-ms "
            "expects existing raw_ms outputs.",
            err=True,
        )
    return filtered


def _archive_root(o: DownloadOptions) -> Path:
    if o.archive_output_root is not None:
        return o.archive_output_root.expanduser().resolve()
    return o.destination.expanduser().resolve() / "archive_ms"


def _download_kwargs(o: DownloadOptions, *, postprocess_in_process: bool) -> dict[str, Any]:
    """``download_products`` arguments; the archive stages run in the download only when asked."""
    return {
        "max_parallel": min(o.max_parallel, MAX_PARALLEL_TOTAL),
        "extract_tar": o.extract_tar,
        "unpack_ms": o.unpack_ms and postprocess_in_process,
        "generate_calibrated_visibilities": (
            o.generate_calibrated_visibilities and postprocess_in_process
        ),
        "clean_intermediate_files": o.clean_intermediate_files and postprocess_in_process,
        "archive_output_root": _archive_root(o),
        "casa_data_root": o.casa_data_root,
        "skip_casa_data_update": o.skip_casa_data_update,
        "logger_fn": typer.echo,
    }


def _download_and_postprocess(o: DownloadOptions, selected: list[Any]) -> None:
    """Download, then unpack/calibrate one UID per subprocess (sync) or per Slurm task.

    A CASA crash on one execution block then never aborts the run.
    """
    archive_root = _archive_root(o)
    summary = _download_products_with_progress(
        selected, o.destination, **_download_kwargs(o, postprocess_in_process=False)
    )
    backend = o.postprocess_backend.lower()
    stage_failures: list[TaskFailure] = []
    with _stage_errors():
        raw_mss, calibrated_mss = _run_parallel_archive_jobs(
            download_root=Path(summary.destination),
            archive_output_root=archive_root,
            unpack_ms=o.unpack_ms,
            generate_calibrated_visibilities=o.generate_calibrated_visibilities,
            postprocess_backend=backend,
            postprocess_backend_kwargs=(
                SlurmOptions(
                    queue=o.slurm_queue,
                    project=o.slurm_project,
                    walltime=o.slurm_walltime,
                    memory=o.slurm_memory,
                    n_workers=o.slurm_workers,
                    cores=o.slurm_cores,
                ).backend_kwargs()
                if backend == "slurm"
                else {}
            ),
            casa_data_root=o.casa_data_root,
            skip_casa_data_update=o.skip_casa_data_update,
            overwrite_archive_outputs=o.overwrite_archive_outputs,
            continue_on_error=o.continue_on_error,
            failures=stage_failures,
            keep_working_copies=o.keep_working_copies,
            echo=typer.echo,
        )
    _update_manifest_measurement_sets(summary.manifest_path, raw_mss, calibrated_mss)
    # Before the cleanup, which can remove the qa/ reports that place each EB in its member.
    _organize_delivery(
        Path(summary.destination), archive_root / "calibrated_ms", archive_root / "raw_ms"
    )
    _print_download_summary(summary, raw_mss, calibrated_mss, show_extraction_failures=True)
    if o.clean_intermediate_files and o.generate_calibrated_visibilities:
        if stage_failures:
            typer.echo(
                "Skipping --clean-intermediate-files because "
                f"{len(stage_failures)} UID(s) failed; inputs are kept for a re-run.",
                err=True,
            )
        else:
            _cleanup_after_download_postprocess(
                summary=summary,
                destination=Path(summary.destination),
                archive_root=archive_root,
                calibrated_mss=calibrated_mss,
            )
    _exit_if_failures("Archive post-processing", stage_failures)


def _print_download_summary(
    summary: Any,
    raw_mss: Sequence[str],
    calibrated_mss: Sequence[str],
    *,
    show_extraction_failures: bool = False,
) -> None:
    typer.echo(f"Destination: {summary.destination}")
    typer.echo(f"Completed: {summary.files_completed}")
    typer.echo(f"Failed: {summary.files_failed}")
    extraction_failed = getattr(summary, "extraction_failed", []) or []
    if show_extraction_failures and extraction_failed:
        typer.echo(
            f"Extraction failed: {len(extraction_failed)} (tar kept on disk; "
            "re-run `almasim products extract` on it)",
            err=True,
        )
        for tar_path in extraction_failed:
            typer.echo(f"  {tar_path}", err=True)
    if summary.manifest_path:
        typer.echo(f"Manifest: {summary.manifest_path}")
    if raw_mss:
        typer.echo("Raw MS products:")
        for raw_ms in raw_mss:
            typer.echo(f"  {raw_ms}")
    if calibrated_mss:
        typer.echo("Calibrated MS products:")
        for calibrated_ms in calibrated_mss:
            typer.echo(f"  {calibrated_ms}")


@products_app.command("extract", hidden=True)
def products_extract(
    source_root: Path = typer.Option(
        default_output_path("downloads"),
        "--source-root",
        help="Directory containing downloaded archive files.",
        rich_help_panel="Inputs",
    ),
    destination: Path | None = typer.Option(
        None,
        "--destination",
        help="Extraction destination (defaults to --source-root).",
        rich_help_panel="Outputs",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Recursively search --source-root for .tar/.tgz archives.",
        rich_help_panel="Inputs",
    ),
    delete_archives: bool = typer.Option(
        False,
        "--delete-archives",
        help="Delete each archive after successful extraction.",
        rich_help_panel="Outputs",
    ),
    postprocess_backend: PostprocessBackend = Backend.sync,
    slurm_queue: SlurmQueue = "normal",
    slurm_project: SlurmProject = None,
    slurm_walltime: SlurmWalltime = "01:00:00",
    slurm_cores: SlurmCores = 1,
    slurm_memory: SlurmMemory = "4GB",
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=0,
        help="Number of Slurm workers. Pass 0 to spawn one worker per archive.",
        rich_help_panel="Cluster",
    ),
    slurm_scheduler_host: SchedulerHost = None,
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip archives that have already been extracted (detected via a .done marker file).",
        rich_help_panel="Outputs",
    ),
) -> None:
    """Extract ALMA archive tarballs as a standalone step."""
    backend_normalized = postprocess_backend.lower()

    source = source_root.expanduser().resolve()
    if not source.exists() or not source.is_dir():
        usage_error(f"--source-root is not a directory: {source}")

    target = destination.expanduser().resolve() if destination is not None else source
    target.mkdir(parents=True, exist_ok=True)

    archives = _find_archives(source, recursive)
    if not archives:
        fail(f"No .tar/.tgz archives found under {source}")

    typer.echo(f"Found {len(archives)} archive(s) to extract.")
    with _stage_errors():
        extracted_files, failed_archives = _run_extract_jobs(
            source=source,
            target=target,
            archives=archives,
            postprocess_backend=backend_normalized,
            postprocess_backend_kwargs=SlurmOptions(
                queue=slurm_queue,
                project=slurm_project,
                walltime=slurm_walltime,
                memory=slurm_memory,
                n_workers=slurm_workers,
                cores=slurm_cores,
                scheduler_host=slurm_scheduler_host,
            ).backend_kwargs(),
            delete_archives=delete_archives,
            skip_existing=skip_existing,
            echo=typer.echo,
        )

    typer.echo(f"Extracted files: {len(extracted_files)}")
    typer.echo(f"Failed archives: {len(failed_archives)}")
    if failed_archives:
        raise typer.Exit(code=1)


@products_app.command("unpack", hidden=True)
def products_unpack(
    input_root: Path = typer.Option(
        default_output_path("downloads"),
        "--input-root",
        help="Directory containing extracted ASDM directories.",
        rich_help_panel="Inputs",
    ),
    output_root: Path = typer.Option(
        default_output_path("downloads") / "archive_ms" / "raw_ms",
        "--output-dir",
        "--output-root",
        help="Directory where raw MeasurementSets are written.",
        rich_help_panel="Outputs",
    ),
    asdm_uid: list[str] | None = typer.Option(
        None,
        "--asdm-uid",
        help="Optional ASDM UID(s) to process. Repeat or pass comma-separated values.",
        rich_help_panel="Inputs",
    ),
    casa_data_root: Path | None = typer.Option(
        None,
        "--casa-data-root",
        help="Optional CASA runtime data directory.",
        rich_help_panel="CASA",
    ),
    skip_casa_data_update: bool = typer.Option(
        False,
        "--skip-casa-data-update",
        help="Do not auto-download CASA runtime data if missing.",
        rich_help_panel="CASA",
    ),
    postprocess_backend: PostprocessBackend = Backend.sync,
    slurm_queue: SlurmQueue = "normal",
    slurm_project: SlurmProject = None,
    slurm_walltime: SlurmWalltime = "02:00:00",
    slurm_cores: SlurmCores = 1,
    slurm_memory: SlurmMemory = "4GB",
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=0,
        help="Number of Slurm workers for post-processing. Pass 0 to spawn one worker per UID.",
        rich_help_panel="Cluster",
    ),
    slurm_scheduler_host: SchedulerHost = None,
    overwrite_outputs: bool = typer.Option(
        False,
        "--overwrite-outputs",
        help="Overwrite existing raw MS outputs.",
        rich_help_panel="Outputs",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help=(
            "Skip UIDs whose processing fails and carry on with the rest, then exit "
            "non-zero with a summary (default). --fail-fast aborts on the first failure."
        ),
        rich_help_panel="Outputs",
    ),
) -> None:
    """Import ASDM directories into raw MeasurementSets as a standalone step."""
    backend_normalized = postprocess_backend.lower()

    parsed_uids = _parse_uid_options(asdm_uid)
    stage_failures: list[TaskFailure] = []
    with _stage_errors():
        raw_outputs = _run_unpack_jobs(
            input_root=input_root.expanduser().resolve(),
            output_root=output_root.expanduser().resolve(),
            asdm_uids=parsed_uids,
            postprocess_backend=backend_normalized,
            postprocess_backend_kwargs=SlurmOptions(
                queue=slurm_queue,
                project=slurm_project,
                walltime=slurm_walltime,
                memory=slurm_memory,
                n_workers=slurm_workers,
                cores=slurm_cores,
                scheduler_host=slurm_scheduler_host,
            ).backend_kwargs(),
            casa_data_root=casa_data_root,
            skip_casa_data_update=skip_casa_data_update,
            overwrite_outputs=overwrite_outputs,
            continue_on_error=continue_on_error,
            failures=stage_failures,
            echo=typer.echo,
        )

    typer.echo(f"Raw MS products: {len(raw_outputs)}")
    for raw_ms in raw_outputs:
        typer.echo(f"  {raw_ms}")
    _exit_if_failures("Unpack", stage_failures)


@products_app.command("calibrate", hidden=True)
def products_calibrate(
    input_root: Path = typer.Option(
        default_output_path("downloads"),
        "--input-root",
        help="ALMA delivery root containing calibration products.",
        rich_help_panel="Inputs",
    ),
    raw_ms_root: Path = typer.Option(
        default_output_path("downloads") / "archive_ms" / "raw_ms",
        "--raw-ms-root",
        help="Directory containing raw MeasurementSets.",
        rich_help_panel="Inputs",
    ),
    output_root: Path = typer.Option(
        default_output_path("downloads") / "archive_ms" / "calibrated_ms",
        "--output-dir",
        "--output-root",
        help="Directory where calibrated MeasurementSets are written.",
        rich_help_panel="Outputs",
    ),
    asdm_uid: list[str] | None = typer.Option(
        None,
        "--asdm-uid",
        help="Optional UID(s) to calibrate. Repeat or pass comma-separated values.",
        rich_help_panel="Inputs",
    ),
    casa_data_root: Path | None = typer.Option(
        None,
        "--casa-data-root",
        help="Optional CASA runtime data directory.",
        rich_help_panel="CASA",
    ),
    skip_casa_data_update: bool = typer.Option(
        False,
        "--skip-casa-data-update",
        help="Do not auto-download CASA runtime data if missing.",
        rich_help_panel="CASA",
    ),
    postprocess_backend: PostprocessBackend = Backend.sync,
    slurm_queue: SlurmQueue = "normal",
    slurm_project: SlurmProject = None,
    slurm_walltime: SlurmWalltime = "02:00:00",
    slurm_cores: SlurmCores = 1,
    slurm_memory: SlurmMemory = "4GB",
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=0,
        help="Number of Slurm workers for post-processing. Pass 0 to spawn one worker per UID.",
        rich_help_panel="Cluster",
    ),
    slurm_scheduler_host: SchedulerHost = None,
    overwrite_outputs: bool = typer.Option(
        False,
        "--overwrite-outputs",
        help="Overwrite existing calibrated MS outputs.",
        rich_help_panel="Outputs",
    ),
    clean_intermediate_files: bool = typer.Option(
        False,
        "--clean-intermediate-files",
        help="Remove intermediate raw and working files after successful calibration.",
        rich_help_panel="Outputs",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help=(
            "Skip UIDs whose processing fails and carry on with the rest, then exit "
            "non-zero with a summary (default). --fail-fast aborts on the first failure."
        ),
        rich_help_panel="Outputs",
    ),
    keep_working_copies: bool = typer.Option(
        False,
        "--keep-working-copies",
        help=(
            "Keep each UID's working directory (raw MS copy plus caltables, about 1.8x the "
            "raw MS) after its calibrated output is written. By default it is removed."
        ),
        rich_help_panel="Outputs",
    ),
    organize_delivery: bool = typer.Option(
        True,
        "--organize-delivery/--no-organize-delivery",
        help=_ORGANIZE_HELP + " The tree is --input-root.",
        rich_help_panel="Outputs",
    ),
) -> None:
    """Create calibrated MeasurementSets as a standalone step."""
    backend_normalized = postprocess_backend.lower()

    parsed_uids = _parse_uid_options(asdm_uid)
    stage_failures: list[TaskFailure] = []
    with _stage_errors():
        calibrated_outputs = _run_calibrate_jobs(
            input_root=input_root.expanduser().resolve(),
            raw_ms_root=raw_ms_root.expanduser().resolve(),
            output_root=output_root.expanduser().resolve(),
            asdm_uids=parsed_uids,
            postprocess_backend=backend_normalized,
            postprocess_backend_kwargs=SlurmOptions(
                queue=slurm_queue,
                project=slurm_project,
                walltime=slurm_walltime,
                memory=slurm_memory,
                n_workers=slurm_workers,
                cores=slurm_cores,
                scheduler_host=slurm_scheduler_host,
            ).backend_kwargs(),
            casa_data_root=casa_data_root,
            skip_casa_data_update=skip_casa_data_update,
            overwrite_outputs=overwrite_outputs,
            clean_intermediate=clean_intermediate_files,
            continue_on_error=continue_on_error,
            failures=stage_failures,
            keep_working_copies=keep_working_copies,
            echo=typer.echo,
        )

    typer.echo(f"Calibrated MS products: {len(calibrated_outputs)}")
    for calibrated_ms in calibrated_outputs:
        typer.echo(f"  {calibrated_ms}")
    if organize_delivery:
        _organize_delivery(
            input_root.expanduser().resolve(),
            output_root.expanduser().resolve(),
            raw_ms_root.expanduser().resolve(),
        )
    _exit_if_failures("Calibrate", stage_failures)


@products_app.command("organize", hidden=True)
def products_organize(
    delivery_root: Path = typer.Option(
        ...,
        "--delivery-root",
        help="Root of the ALMA delivery tree (<project>/science_goal.*/group.*/member.*/).",
    ),
    calibrated_ms_root: Path = typer.Option(
        ..., "--calibrated-ms-root", help="Folder of the calibrated MSs (<eb>.ms.split.cal)."
    ),
    raw_ms_root: Path | None = typer.Option(
        None, "--raw-ms-root", help="Folder of the raw MSs (<eb>.ms), recorded in the index."
    ),
    index_csv: Path | None = typer.Option(
        None,
        "--index-csv",
        help="Where to write the index (default: <delivery-root>/ous_index.csv).",
    ),
) -> None:
    """Link the calibrated MSs into their member folders and index the delivery tree."""
    from ..services.archive.delivery import organize_delivery

    root = delivery_root.expanduser().resolve()
    if not root.is_dir():
        usage_error(f"--delivery-root {root} is not a folder.")
    organize_delivery(
        root,
        calibrated_ms_root.expanduser().resolve(),
        raw_ms_root.expanduser().resolve() if raw_ms_root else None,
        index_path=index_csv.expanduser().resolve() if index_csv else None,
        echo=typer.echo,
    )


@products_app.command("executions")
def products_executions(
    metadata_path: Path = typer.Option(
        ...,
        "--metadata-path",
        help="Metadata CSV (from 'almasim metadata') to add n_executions to.",
    ),
    save_csv: Path | None = typer.Option(
        None,
        "--save-csv",
        help="Destination CSV; defaults to overwriting --metadata-path.",
    ),
) -> None:
    """Add how many execution blocks observed each row's MOUS (n_executions).

    A row's Int.Time is its MOUS's total time on source; `simulation run`
    observes it in n_executions blocks (without the column: blocks of at most
    3000 s).
    """
    import pandas as pd

    from ..services.metadata.storage import save_dataframe_csv
    from .metadata import _with_execution_counts

    metadata = _with_execution_counts(pd.read_csv(metadata_path))
    output_csv = (save_csv or metadata_path).expanduser().resolve()
    save_dataframe_csv(metadata, output_csv)
    typer.echo(f"Saved metadata CSV: {output_csv}")
