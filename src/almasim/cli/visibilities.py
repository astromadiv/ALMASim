"""Visibility processing commands for the ALMASim CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from ..services.instrument.constants import DEFAULT_MAX_WEIGHT
from . import image as cli_image
from . import products as cli_products
from ._shared import fail

visibilities_app = typer.Typer(
    help="Visibility processing commands.",
    no_args_is_help=True,
)

# One implementation each: the products commands (whose names stay as hidden
# aliases) and the image MS overview.
visibilities_app.command("extract")(cli_products.products_extract)
visibilities_app.command("unpack")(cli_products.products_unpack)
visibilities_app.command("calibrate")(cli_products.products_calibrate)
visibilities_app.command("spw-overview")(cli_image.derive_parameters)


@visibilities_app.command("flag-summary")
def flag_summary(  # noqa: C901
    inputs: list[Path] = typer.Argument(
        ...,
        help=(
            "MeasurementSet(s) to summarise, or folders searched with --pattern. "
            "Several may be given."
        ),
    ),
    output_csv: Path = typer.Option(
        Path("flag_summary.csv"),
        "--output",
        "-o",
        help="CSV to write, one row per (MS, field, SPW).",
    ),
    pattern: str = typer.Option(
        "*.ms.split.cal", "--pattern", help="Glob for the MeasurementSets inside a folder."
    ),
    max_weight: float = typer.Option(
        DEFAULT_MAX_WEIGHT,
        "--max-weight",
        help="Unflagged rows with a WEIGHT above this count as high_weight_rows.",
    ),
    require_done_marker: bool = typer.Option(
        True,
        "--require-done-marker/--no-require-done-marker",
        help=(
            "In a folder, skip an MS whose <ms>.done calibration marker is missing. Only "
            "applies when the folder holds at least one such marker. MSs named directly "
            "are always summarised."
        ),
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Abort on the first MS that cannot be read (default: record it and continue).",
    ),
) -> None:
    """Summarise flagging and visibility properties per MS, field and spectral window.

    For each (field, SPW): intents, channels, correlations, scans, antennas,
    time range and time on source, rows and visibilities flagged, median and
    largest unflagged weight, and unflagged rows above --max-weight. Reads only
    FLAG, FLAG_ROW, WEIGHT and the scalar columns, never DATA. An MS that cannot
    be read is listed in ``<output>.failed.tsv`` and the exit status is 1.
    """
    import pandas as pd
    from tqdm import tqdm

    from ..services.archive.flag_summary import flag_summary as summarise

    mss: list[Path] = []
    for item in inputs:
        if (item / "table.dat").is_file():
            mss.append(item)
            continue
        if not item.is_dir():
            fail(f"Not a MeasurementSet or folder: {item}")
        found = sorted(p for p in item.glob(pattern) if p.is_dir())
        if require_done_marker:
            with_marker = [ms for ms in found if ms.with_name(ms.name + ".done").is_file()]
            if with_marker and len(with_marker) < len(found):
                typer.echo(
                    f"Skipping {len(found) - len(with_marker)} MS(s) in {item} without a "
                    ".done marker."
                )
                found = with_marker
        mss.extend(found)
    if not mss:
        typer.echo(f"Cannot find any MS matching {pattern} in {', '.join(map(str, inputs))}")
        raise typer.Exit(code=1)

    frames: list[pd.DataFrame] = []
    failures: list[tuple[Path, str]] = []
    for ms in tqdm(mss, disable=len(mss) == 1):
        try:
            frames.append(summarise(ms, max_weight=max_weight))
        except Exception as exc:
            if fail_fast:
                raise
            failures.append((ms, f"{type(exc).__name__}: {exc}"))
            typer.echo(f"FAILED {ms.name}: {type(exc).__name__}: {exc}", err=True)

    if frames:
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(output_csv, index=False)
        rows, flagged = int(table["rows"].sum()), int(table["flagged_rows"].sum())
        high = int(table["high_weight_rows"].sum())
        with_high = int((table.groupby("uid")["high_weight_rows"].sum() > 0).sum())
        typer.echo(
            f"Wrote {len(table)} (MS, field, SPW) row(s) for {len(frames)} MS(s) to {output_csv}"
        )
        typer.echo(
            f"Rows: {rows:,}; flagged {flagged:,} ({100 * flagged / max(rows, 1):.2f} %); "
            f"unflagged with WEIGHT > {max_weight:g}: {high:,} in {with_high} MS(s)"
        )
        is_target = table["intents"].str.contains("OBSERVE_TARGET", na=False)
        for label, part in (("target", table[is_target]), ("other", table[~is_target])):
            if len(part):
                share = 100 * part["flagged_rows"].sum() / max(part["rows"].sum(), 1)
                typer.echo(f"  {label} fields: {share:.2f} % of rows flagged")
    if failures:
        failed_path = output_csv.with_name(output_csv.name + ".failed.tsv")
        failed_path.write_text("".join(f"{ms}\t{error}\n" for ms, error in failures))
        fail(f"{len(failures)} MS(s) could not be read; see {failed_path}")
