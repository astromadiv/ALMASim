"""Metadata query commands for the ALMASim CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from ..services.metadata.storage import save_dataframe_csv
from ..services.metadata.tap import (
    ALL_COLUMNS,
    ExclusionFilters,
    InclusionFilters,
    query_metadata_by_science,
)
from ._shared import (
    default_output_path,
    fail,
    split_csv_values,
    usage_error,
    validate_date_range,
    validate_range,
)

metadata_app = typer.Typer(
    help="Metadata query commands.",
    no_args_is_help=True,
)


def _build_query_summary(
    include: Any,
    exclude: Any,
    visible_columns: list[str] | None,
    limit: int | None,
) -> str:
    summary: dict[str, Any] = {
        "source_name": include.source_name,
        "science_keyword": include.science_keyword,
        "scientific_category": include.scientific_category,
        "band": include.band,
        "antenna_arrays": include.antenna_arrays,
        "array_type": include.array_type,
        "array_configuration": include.array_configuration,
        "angular_resolution_range": include.angular_resolution_range,
        "observation_date_range": include.observation_date_range,
        "qa2_status": include.qa2_status,
        "obs_type": include.obs_type,
        "fov_range": include.fov_range,
        "time_resolution_range": include.time_resolution_range,
        "frequency_range": include.frequency_range,
        "proposal_id_prefix": include.proposal_id_prefix,
        "proposal_id": include.proposal_id,
        "public_only": include.public_only,
        "science_only": include.science_only,
        "exclude_mosaic": include.exclude_mosaic,
        "exclude_science_keyword": exclude.science_keyword,
        "exclude_scientific_category": exclude.scientific_category,
        "exclude_source_name": exclude.source_name,
        "exclude_obs_type": exclude.obs_type,
        "exclude_solar": exclude.solar,
        "visible_columns": visible_columns,
        "limit": limit,
    }
    active = {key: value for key, value in summary.items() if value not in (None, [], False)}
    return ", ".join(f"{key}={value}" for key, value in active.items())


@dataclass(frozen=True)
class MetadataQueryOptions:
    """Every option of ``metadata`` (MADIV-28 T3.8)."""

    source_name: str | None
    science_keyword: list[str] | None
    scientific_category: list[str] | None
    band: list[int] | None
    antenna_arrays: str | None
    array_type: list[str] | None
    array_configuration: list[str] | None
    angular_resolution_min: float | None
    angular_resolution_max: float | None
    observation_date_start: str | None
    observation_date_end: str | None
    qa2_status: list[str] | None
    obs_type: list[str] | None
    fov_min: float | None
    fov_max: float | None
    time_resolution_min: float | None
    time_resolution_max: float | None
    frequency_min: float | None
    frequency_max: float | None
    exclude_science_keyword: list[str] | None
    exclude_scientific_category: list[str] | None
    exclude_source_name: list[str] | None
    exclude_obs_type: list[str] | None
    exclude_solar: bool
    cycles: list[int] | None
    project_codes: list[str] | None
    public_only: bool
    science_only: bool
    exclude_mosaic: bool
    visible_columns: list[str] | None
    limit: int | None
    save_csv: Path | None
    save_products_csv: Path | None
    product_member_limit: int | None
    count_executions: bool
    yes: bool


@metadata_app.command("query")
def metadata_query(
    source_name: str | None = typer.Option(
        None,
        "--source-name",
        help="Source-name substring filter.",
        rich_help_panel="Target",
    ),
    science_keyword: list[str] | None = typer.Option(
        None,
        "--science-keyword",
        help="Science keyword(s) to include. Repeat or pass comma-separated values.",
        rich_help_panel="Science",
    ),
    scientific_category: list[str] | None = typer.Option(
        None,
        "--scientific-category",
        help="Scientific category(ies) to include. Repeat or pass comma-separated values.",
        rich_help_panel="Science",
    ),
    band: list[int] | None = typer.Option(
        None,
        "--band",
        help="ALMA band filter. Repeat for multiple values.",
        rich_help_panel="Spectral setup",
    ),
    antenna_arrays: str | None = typer.Option(
        None,
        "--antenna-arrays",
        help="Raw antenna_arrays substring filter.",
        rich_help_panel="Array",
    ),
    array_type: list[str] | None = typer.Option(
        None,
        "--array-type",
        help="Array type(s): 12m, 7m, TP. Repeat or pass comma-separated values.",
        rich_help_panel="Array",
    ),
    array_configuration: list[str] | None = typer.Option(
        None,
        "--array-configuration",
        help="Array configuration(s) like C-1, C-2. Repeat or pass comma-separated values.",
        rich_help_panel="Array",
    ),
    angular_resolution_min: float | None = typer.Option(
        None,
        "--angular-resolution-min",
        help="Minimum angular resolution (arcsec).",
        rich_help_panel="Array",
    ),
    angular_resolution_max: float | None = typer.Option(
        None,
        "--angular-resolution-max",
        help="Maximum angular resolution (arcsec).",
        rich_help_panel="Array",
    ),
    observation_date_start: str | None = typer.Option(
        None,
        "--observation-date-start",
        help="Observation date start (ISO date).",
        rich_help_panel="Observation",
    ),
    observation_date_end: str | None = typer.Option(
        None,
        "--observation-date-end",
        help="Observation date end (ISO date).",
        rich_help_panel="Observation",
    ),
    qa2_status: list[str] | None = typer.Option(
        None,
        "--qa2-status",
        help="QA2 status values: Pass, Fail, SemiPass, T, F, X.",
        rich_help_panel="Observation",
    ),
    obs_type: list[str] | None = typer.Option(
        None,
        "--obs-type",
        help="Observation type(s) to include. Repeat or pass comma-separated values.",
        rich_help_panel="Observation",
    ),
    fov_min: float | None = typer.Option(
        None,
        "--fov-min",
        help="Minimum FOV.",
        rich_help_panel="Array",
    ),
    fov_max: float | None = typer.Option(
        None,
        "--fov-max",
        help="Maximum FOV.",
        rich_help_panel="Array",
    ),
    time_resolution_min: float | None = typer.Option(
        None,
        "--time-resolution-min",
        help="Minimum time resolution.",
        rich_help_panel="Observation",
    ),
    time_resolution_max: float | None = typer.Option(
        None,
        "--time-resolution-max",
        help="Maximum time resolution.",
        rich_help_panel="Observation",
    ),
    frequency_min: float | None = typer.Option(
        None,
        "--frequency-min",
        help="Minimum frequency.",
        rich_help_panel="Spectral setup",
    ),
    frequency_max: float | None = typer.Option(
        None,
        "--frequency-max",
        help="Maximum frequency.",
        rich_help_panel="Spectral setup",
    ),
    exclude_science_keyword: list[str] | None = typer.Option(
        None,
        "--exclude-science-keyword",
        help="Science keyword(s) to exclude. Repeat or pass comma-separated values.",
        rich_help_panel="Exclusions",
    ),
    exclude_scientific_category: list[str] | None = typer.Option(
        None,
        "--exclude-scientific-category",
        help="Scientific category(ies) to exclude. Repeat or pass comma-separated values.",
        rich_help_panel="Exclusions",
    ),
    exclude_source_name: list[str] | None = typer.Option(
        None,
        "--exclude-source-name",
        help="Source-name substring(s) to exclude. Repeat or pass comma-separated values.",
        rich_help_panel="Exclusions",
    ),
    exclude_obs_type: list[str] | None = typer.Option(
        None,
        "--exclude-obs-type",
        help="Observation type substring(s) to exclude. Repeat or pass comma-separated values.",
        rich_help_panel="Exclusions",
    ),
    exclude_solar: bool = typer.Option(
        False,
        "--exclude-solar",
        help="Exclude solar observations.",
        rich_help_panel="Exclusions",
    ),
    cycles: list[int] | None = typer.Option(
        None,
        "--cycle",
        help="ALMA cycle number(s). Cycle N maps to proposal_id prefix {2012+N}.",
        rich_help_panel="Target",
    ),
    project_codes: list[str] | None = typer.Option(
        None,
        "--project-code",
        help=(
            "Exact ALMA project code(s). Repeat for multiple or pass comma-separated values. "
            "Example: --project-code 2019.1.00001.S --project-code 2022.1.00333.S"
        ),
        rich_help_panel="Target",
    ),
    public_only: bool = typer.Option(
        True,
        "--public-only/--include-proprietary",
        help="Restrict to public data (default: public only).",
        rich_help_panel="Science",
    ),
    science_only: bool = typer.Option(
        True,
        "--science-only/--include-non-science",
        help="Restrict to science observations (default: science only).",
        rich_help_panel="Science",
    ),
    exclude_mosaic: bool = typer.Option(
        True,
        "--exclude-mosaic/--include-mosaic",
        help="Exclude mosaic observations (default: excluded).",
        rich_help_panel="Observation",
    ),
    visible_columns: list[str] | None = typer.Option(
        None,
        "--visible-column",
        help="Column(s) to include in output (ordered). Repeat this option per column.",
        rich_help_panel="Output",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Maximum number of metadata rows to keep. Defaults to unlimited.",
        rich_help_panel="Output",
    ),
    save_csv: Path | None = typer.Option(
        None,
        "--save-csv",
        help="Destination CSV for normalized metadata rows.",
        rich_help_panel="Output",
    ),
    save_products_csv: Path | None = typer.Option(
        None,
        "--save-products-csv",
        help=(
            "Optional destination CSV for the DataLink product rows of the queried "
            "member OUS UIDs, ready for `products download --products-csv`."
        ),
        rich_help_panel="Output",
    ),
    product_member_limit: int | None = typer.Option(
        None,
        "--product-member-limit",
        min=1,
        help=(
            "How many queried member OUS UIDs to resolve into product rows. Defaults to unlimited."
        ),
        rich_help_panel="Output",
    ),
    count_executions: bool = typer.Option(
        False,
        "--count-executions",
        help=(
            "Add n_executions: how many execution blocks observed each row's MOUS "
            "(its raw ASDMs in DataLink; one request per MOUS). Int.Time is the "
            "MOUS's total, which `simulation run` observes in that many blocks."
        ),
        rich_help_panel="Output",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt and start the query immediately.",
        rich_help_panel="Output",
    ),
) -> None:
    """Query ALMA metadata from TAP and save normalized CSV results."""
    options = MetadataQueryOptions(**locals())
    include, exclude, output_columns = _query_filters(options)
    if not options.yes:
        message = (
            "About to start ALMA TAP metadata query with "
            + _build_query_summary(include, exclude, output_columns, options.limit)
            + ". Continue?"
        )
        if not typer.confirm(message, default=True):
            typer.echo("Query cancelled.")
            raise typer.Exit(code=0)
    typer.echo("Starting ALMA TAP metadata query...")
    metadata = query_metadata_by_science(
        include=include,
        exclude=exclude,
        visible_columns=output_columns,
    )
    if metadata.empty:
        fail("No metadata rows matched the requested filters")
    if options.limit is not None:
        metadata = metadata.head(options.limit).reset_index(drop=True)
    _save_and_report_metadata(options, metadata)


def _query_filters(
    o: MetadataQueryOptions,
) -> tuple[InclusionFilters, ExclusionFilters, list[str] | None]:
    """The query's inclusion and exclusion filters and output columns, validated."""
    array_types = split_csv_values(o.array_type)
    output_columns = split_csv_values(o.visible_columns)
    invalid_array_types = [
        value for value in (array_types or []) if value not in {"12m", "7m", "TP"}
    ]
    if invalid_array_types:
        usage_error(
            "Invalid --array-type values: "
            + ", ".join(invalid_array_types)
            + ". Allowed values: 12m, 7m, TP."
        )
    invalid_columns = [column for column in (output_columns or []) if column not in ALL_COLUMNS]
    if invalid_columns:
        typer.echo(
            "Invalid --visible-column values: " + ", ".join(invalid_columns) + ".",
            err=True,
        )
        usage_error("Allowed columns: " + ", ".join(ALL_COLUMNS))

    include = InclusionFilters(
        science_keyword=split_csv_values(o.science_keyword),
        scientific_category=split_csv_values(o.scientific_category),
        band=o.band,
        source_name=o.source_name,
        antenna_arrays=o.antenna_arrays,
        array_type=array_types,
        array_configuration=split_csv_values(o.array_configuration),
        angular_resolution_range=validate_range(
            min_value=o.angular_resolution_min,
            max_value=o.angular_resolution_max,
            label="angular resolution range",
        ),
        observation_date_range=validate_date_range(
            start_date=o.observation_date_start,
            end_date=o.observation_date_end,
        ),
        qa2_status=split_csv_values(o.qa2_status),
        obs_type=split_csv_values(o.obs_type),
        fov_range=validate_range(min_value=o.fov_min, max_value=o.fov_max, label="FOV range"),
        time_resolution_range=validate_range(
            min_value=o.time_resolution_min,
            max_value=o.time_resolution_max,
            label="time resolution range",
        ),
        frequency_range=validate_range(
            min_value=o.frequency_min,
            max_value=o.frequency_max,
            label="frequency range",
        ),
        proposal_id_prefix=[f"{2012 + cycle}." for cycle in o.cycles] if o.cycles else None,
        proposal_id=split_csv_values(o.project_codes),
        public_only=o.public_only,
        science_only=o.science_only,
        exclude_mosaic=o.exclude_mosaic,
    )
    exclude = ExclusionFilters(
        science_keyword=split_csv_values(o.exclude_science_keyword),
        scientific_category=split_csv_values(o.exclude_scientific_category),
        source_name=split_csv_values(o.exclude_source_name),
        obs_type=split_csv_values(o.exclude_obs_type),
        solar=o.exclude_solar,
    )
    return include, exclude, output_columns


def _save_and_report_metadata(o: MetadataQueryOptions, metadata: Any) -> None:
    """Save the rows, print a preview and, when asked, resolve their DataLink products."""
    output_csv = (
        (o.save_csv or default_output_path("metadata_query_results.csv")).expanduser().resolve()
    )
    if o.count_executions:
        metadata = _with_execution_counts(metadata)
    save_dataframe_csv(metadata, output_csv)
    typer.echo(f"Saved metadata CSV: {output_csv}")
    typer.echo(f"Metadata rows: {len(metadata)}")
    display_columns = [
        col
        for col in ("ALMA_source_name", "Band", "Freq", "member_ous_uid")
        if col in metadata.columns
    ]
    if display_columns:
        typer.echo(metadata[display_columns].head(min(5, len(metadata))).to_string(index=False))
    if o.save_products_csv is None:
        return
    if "member_ous_uid" not in metadata.columns:
        fail("Queried metadata does not include member_ous_uid values")
    member_series = metadata["member_ous_uid"].dropna().astype(str)
    if o.product_member_limit is not None:
        member_series = member_series.head(o.product_member_limit)
    member_uids = member_series.tolist()
    if not member_uids:
        fail("Queried metadata does not include member_ous_uid values")
    # DataLink rows, the format `products download --products-csv` reads
    # (the TAP product rows saved here before could not be downloaded).
    from . import products as cli_products

    cli_products._resolve_products_from_inputs(
        products_csv=None,
        metadata_csv=None,
        member_ous_uid=member_uids,
        member_limit=None,
        save_products_csv_path=o.save_products_csv,
    )


def _with_execution_counts(metadata: Any) -> Any:
    """``metadata`` with each row's MOUS's execution blocks (n_executions)."""
    from ..services.metadata.executions import COLUMN, add_execution_counts

    if "member_ous_uid" not in metadata.columns:
        fail("Metadata does not include member_ous_uid values")
    typer.echo(f"Counting execution blocks of {metadata['member_ous_uid'].nunique()} MOUS(s)...")
    counted = add_execution_counts(metadata)
    missing = int(counted[COLUMN].isna().sum())
    if missing:
        typer.echo(f"No raw ASDMs found for {missing} row(s): n_executions left empty.", err=True)
    return counted
