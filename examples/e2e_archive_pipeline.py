"""End-to-end ALMA archive pipeline: query → download → unpack → calibrate.

Run with:
    marimo run examples/e2e_archive_pipeline.py
    marimo edit examples/e2e_archive_pipeline.py   # interactive editing

Requirements:
    uv sync --group dev
    pip install "almasim[casa]"   # casatools + casatasks required for Steps 4-5
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", app_title="ALMASim Archive Pipeline")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


@app.cell
def _header():
    import marimo as mo

    mo.md(
        """
        # ALMASim End-to-End Archive Pipeline

        Steps covered:

        1. **Query** ALMA metadata via TAP
        2. **Resolve** DataLink product URLs for selected observations
        3. **Download** raw data products (parallel, multi-mirror)
        4. **Unpack** ASDM archives into raw MeasurementSets *(requires CASA, Linux)*
        5. **Calibrate** raw MSs with the delivered calibration *(requires CASA, Linux)*

        Query filter presets can be **saved** as JSON files and **reloaded** in future sessions.
        """
    )
    return (mo,)


# ---------------------------------------------------------------------------
# Presets directory
# ---------------------------------------------------------------------------


@app.cell
def _presets_setup(mo):
    from pathlib import Path

    presets_dir_input = mo.ui.text(
        value="examples/output/query_presets",
        label="Query presets directory",
        placeholder="Path where .query.json files are stored",
    )
    mo.vstack([mo.md("### Query Presets Storage"), presets_dir_input])
    return Path, presets_dir_input


@app.cell
def _presets_dir(Path, presets_dir_input):
    presets_dir = Path(presets_dir_input.value).expanduser().resolve()
    presets_dir.mkdir(parents=True, exist_ok=True)
    return (presets_dir,)


# ---------------------------------------------------------------------------
# Load preset (optional — runs before config so defaults are applied)
# ---------------------------------------------------------------------------


@app.cell
def _load_preset_section(mo, presets_dir):
    from almasim.services.metadata import list_presets

    available = list_presets(presets_dir)
    if not available:
        mo.md("> No saved presets yet. Run a query and save it below to create one.")
        return None, None

    options = {"(none — use manual config)": None}
    options.update({f"{p.name}  [{p.result_count} rows]": p for p in available})
    preset_selector = mo.ui.dropdown(options=options, label="Load a saved query preset")
    mo.vstack([mo.md("---\n### Load Saved Preset"), preset_selector])
    return None, preset_selector


@app.cell
def _apply_preset(mo, preset_selector):
    loaded_preset = None
    if preset_selector is not None and preset_selector.value is not None:
        loaded_preset = preset_selector.value
        f = loaded_preset.filters
        mo.callout(
            mo.md(
                f"**Preset loaded: `{loaded_preset.name}`**  \n"
                f"{loaded_preset.description}  \n\n"
                "| Filter | Value |\n|---|---|\n"
                + "\n".join(f"| `{k}` | `{v}` |" for k, v in f.items())
            ),
            kind="info",
        )
    return (loaded_preset,)


# ---------------------------------------------------------------------------
# Configuration (defaults from loaded preset when available)
# ---------------------------------------------------------------------------


@app.cell
def _config_ui(loaded_preset, mo):
    f = loaded_preset.filters if loaded_preset else {}

    band_map = {"Band 3": 3, "Band 4": 4, "Band 6": 6, "Band 7": 7, "Band 9": 9}

    science_keyword = mo.ui.text(
        value=f.get("science_keyword", "Galaxies"),
        label="Science keyword",
        placeholder="e.g. Galaxies, ISM, Star formation",
    )
    band = mo.ui.dropdown(
        options=band_map,
        # A preset stores the band number (band.value); the dropdown takes its label.
        value=f"Band {f.get('band', 6)}",
        label="ALMA band",
    )
    row_limit = mo.ui.slider(
        1, 50, value=int(f.get("row_limit", 10)), label="Metadata rows to fetch"
    )
    member_limit = mo.ui.slider(
        1, 5, value=int(f.get("member_limit", 1)), label="Members to resolve & download"
    )
    output_dir = mo.ui.text(
        value=f.get("output_dir", "examples/output/archive_pipeline"),
        label="Output directory",
    )
    max_parallel = mo.ui.slider(
        1, 6, value=int(f.get("max_parallel", 3)), label="Max parallel downloads"
    )

    mo.vstack(
        [
            mo.md("---\n## Configuration"),
            mo.md("### Query filters"),
            mo.hstack([science_keyword, band, row_limit]),
            mo.md("### Download settings"),
            mo.hstack([member_limit, max_parallel, output_dir]),
        ]
    )
    return band, max_parallel, member_limit, output_dir, row_limit, science_keyword


# ---------------------------------------------------------------------------
# Step 1 — Query metadata
# ---------------------------------------------------------------------------


@app.cell
def _step1_header(mo):
    mo.md("---\n## Step 1 — Query ALMA Metadata")
    return


@app.cell
def _query(band, mo, row_limit, science_keyword):
    from almasim.services.metadata.tap import InclusionFilters, query_metadata_by_science

    with mo.status.spinner(title="Querying ALMA TAP service…"):
        include = InclusionFilters(
            science_keyword=[kw.strip() for kw in science_keyword.value.split(",")],
            band=[band.value],
        )
        metadata = query_metadata_by_science(include=include)

    if metadata.empty:
        mo.stop(True, mo.callout(mo.md("No metadata rows matched the filters."), kind="warn"))

    metadata = metadata.head(row_limit.value).reset_index(drop=True)
    mo.md(f"**Fetched {len(metadata)} metadata rows.**")
    return (metadata,)


@app.cell
def _query_table(metadata, mo):
    display_cols = [
        c
        for c in ("ALMA_source_name", "Band", "Freq", "spatial_resolution", "member_ous_uid")
        if c in metadata.columns
    ]
    mo.ui.table(metadata[display_cols], label="Metadata results", selection="single")
    return


@app.cell
def _save_metadata(Path, metadata, mo, output_dir):
    out = Path(output_dir.value).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    metadata_csv = out / "metadata.csv"
    metadata.to_csv(metadata_csv, index=False)
    mo.callout(mo.md(f"Metadata CSV saved: `{metadata_csv}`"), kind="success")
    return metadata_csv, out


# ---------------------------------------------------------------------------
# Save preset (after a successful query)
# ---------------------------------------------------------------------------


@app.cell
def _save_preset_ui(mo):
    preset_name = mo.ui.text(
        value="",
        label="Preset name",
        placeholder="e.g. galaxies_band6",
    )
    preset_desc = mo.ui.text(
        value="",
        label="Description (optional)",
        placeholder="e.g. Public Band 6 galaxy sample",
    )
    save_btn = mo.ui.run_button(label="Save Query Preset")
    mo.vstack(
        [
            mo.md("---\n### Save Current Query as Preset"),
            mo.hstack([preset_name, preset_desc, save_btn]),
        ]
    )
    return preset_desc, preset_name, save_btn


@app.cell
def _do_save_preset(
    band,
    max_parallel,
    member_limit,
    metadata,
    mo,
    output_dir,
    preset_desc,
    preset_name,
    presets_dir,
    row_limit,
    save_btn,
    science_keyword,
):
    from almasim.services.metadata import QueryPreset, save_preset

    mo.stop(not save_btn.value, mo.md("*Fill in a preset name and click Save.*"))

    name = preset_name.value.strip()
    if not name:
        mo.stop(True, mo.callout(mo.md("Preset name cannot be empty."), kind="warn"))

    preset = QueryPreset(
        name=name,
        description=preset_desc.value.strip(),
        result_count=len(metadata),
        filters={
            "science_keyword": science_keyword.value,
            "band": band.value,
            "row_limit": row_limit.value,
            "member_limit": member_limit.value,
            "output_dir": output_dir.value,
            "max_parallel": max_parallel.value,
        },
    )
    path = save_preset(preset, presets_dir)
    mo.callout(mo.md(f"Preset **`{name}`** saved to `{path}`"), kind="success")
    return


# ---------------------------------------------------------------------------
# Step 2 — Resolve DataLink products
# ---------------------------------------------------------------------------


@app.cell
def _step2_header(mo):
    mo.md("---\n## Step 2 — Resolve DataLink Products")
    return


@app.cell
def _resolve(member_limit, metadata, mo, out):
    from almasim.services.download import resolve_products, save_products_csv

    member_uids = metadata["member_ous_uid"].dropna().astype(str).head(member_limit.value).tolist()
    if not member_uids:
        mo.stop(True, mo.callout(mo.md("No `member_ous_uid` values in metadata."), kind="warn"))

    with mo.status.spinner(title=f"Resolving products for {len(member_uids)} member(s)…"):
        products = resolve_products(member_uids)

    if not products:
        mo.stop(True, mo.callout(mo.md("No products resolved."), kind="warn"))

    products_csv_path = out / "resolved_products.csv"
    save_products_csv(products, products_csv_path)

    mo.md(
        f"Resolved **{len(products)} products** across {len(member_uids)} member OUS(s).  \n"
        f"Saved: `{products_csv_path}`"
    )
    return products, products_csv_path


@app.cell
def _products_table(mo, products):
    import pandas as pd

    rows = [
        {
            "uid": p.uid,
            "type": p.product_type,
            "size_mb": round(p.size_mb, 2),
            "filename": p.filename,
        }
        for p in products
    ]
    mo.ui.table(pd.DataFrame(rows), label="Resolved products")
    return


# ---------------------------------------------------------------------------
# Step 3 — Download
# ---------------------------------------------------------------------------


@app.cell
def _step3_header(mo):
    mo.md("---\n## Step 3 — Download Products")
    return


@app.cell
def _product_filter_ui(mo):
    from almasim.services.download import PRODUCT_TYPES

    product_filter = mo.ui.dropdown(
        options=sorted(PRODUCT_TYPES),
        value="all",
        label="Product type filter",
    )
    extract_tar = mo.ui.checkbox(value=True, label="Extract tar archives after download")
    mo.hstack([product_filter, extract_tar])
    return extract_tar, product_filter


@app.cell
def _download(mo, product_filter, products):
    from almasim.services.download import filter_products, format_bytes

    filtered = filter_products(products, product_filter.value)
    if not filtered:
        msg = f"No products match filter `{product_filter.value}`."
        mo.stop(True, mo.callout(mo.md(msg), kind="warn"))

    total = sum(p.content_length for p in filtered)
    mo.md(f"Downloading **{len(filtered)} files** ({format_bytes(total)})…")
    return (filtered,)


@app.cell
def _run_download(extract_tar, filtered, max_parallel, mo, out):
    from almasim.services.download import download_products

    download_dest = out / "downloads"
    with mo.status.spinner(title="Downloading…"):
        summary = download_products(
            filtered,
            download_dest,
            max_parallel=max_parallel.value,
            extract_tar=extract_tar.value,
            unpack_ms=False,
            generate_calibrated_visibilities=False,
            logger_fn=print,
        )

    mo.callout(
        mo.md(
            f"**Download complete.**  \n"
            f"Completed: {summary.files_completed}  \n"
            f"Failed: {summary.files_failed}  \n"
            f"Destination: `{summary.destination}`"
        ),
        kind="success" if summary.files_failed == 0 else "warn",
    )
    return download_dest, summary


# ---------------------------------------------------------------------------
# Step 4 — Unpack ASDMs → raw MeasurementSets
# ---------------------------------------------------------------------------


@app.cell
def _step4_header(mo):
    mo.md(
        "---\n## Step 4 — Unpack ASDMs into Raw MeasurementSets\n\n"
        '> Requires `casatools` + `casatasks` (`pip install "almasim[casa]"`). '
        "Linux x86-64 only."
    )
    return


@app.cell
def _casa_check(mo):
    try:
        import casatasks  # noqa: F401
        import casatools  # noqa: F401

        casa_available = True
        mo.callout(mo.md("CASA tools detected — Steps 4 and 5 will run."), kind="success")
    except Exception as e:  # noqa: BLE001 - CASA may be installed but broken
        casa_available = False
        mo.callout(
            mo.md(
                f"**CASA tools not available** (`{e}`).  \n"
                'Install with `pip install "almasim[casa]"` on Linux x86-64.  \n'
                "Steps 4 and 5 are skipped."
            ),
            kind="warn",
        )
    return (casa_available,)


@app.cell
def _unpack(casa_available, download_dest, mo, out):
    if not casa_available:
        mo.stop(True)

    from almasim.services.archive import create_measurement_sets

    raw_ms_root = out / "raw_ms"
    raw_ms_root.mkdir(parents=True, exist_ok=True)

    with mo.status.spinner(title="Importing ASDMs into raw MeasurementSets…"):
        raw_mss = create_measurement_sets(
            input_root=download_dest,
            output_root=raw_ms_root,
            logger_fn=print,
        )

    if not raw_mss:
        mo.stop(
            True,
            mo.callout(
                mo.md("No raw MSs created. Check extracted ASDMs are present in download dir."),
                kind="warn",
            ),
        )

    mo.callout(
        mo.md(
            f"**{len(raw_mss)} raw MS(s) created.**\n\n" + "\n".join(f"- `{p}`" for p in raw_mss)
        ),
        kind="success",
    )
    return (raw_ms_root,)


# ---------------------------------------------------------------------------
# Step 5 — Apply delivered calibration
# ---------------------------------------------------------------------------


@app.cell
def _step5_header(mo):
    mo.md(
        "---\n## Step 5 — Apply Delivered Calibration\n\n"
        "Restores the delivered pipeline flags, replays the delivered `applycal` "
        "calls with `casatasks.applycal`, and splits the calibrated science SPWs "
        "into separate MSs."
    )
    return


@app.cell
def _calibrate(casa_available, download_dest, mo, out, raw_ms_root):
    if not casa_available:
        mo.stop(True)

    from almasim.services.archive import create_calibrated_measurement_sets

    calibrated_ms_root = out / "calibrated_ms"
    calibrated_ms_root.mkdir(parents=True, exist_ok=True)

    with mo.status.spinner(title="Applying calibration scripts…"):
        calibrated_mss = create_calibrated_measurement_sets(
            input_root=download_dest,
            raw_ms_root=raw_ms_root,
            output_root=calibrated_ms_root,
            logger_fn=print,
        )

    if not calibrated_mss:
        mo.stop(
            True,
            mo.callout(
                mo.md("No calibrated MSs produced. Check CASA logs for details."),
                kind="warn",
            ),
        )

    mo.callout(
        mo.md(
            f"**{len(calibrated_mss)} calibrated MS(s) ready.**\n\n"
            + "\n".join(f"- `{p}`" for p in calibrated_mss)
        ),
        kind="success",
    )
    return


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@app.cell
def _summary(metadata, metadata_csv, mo, out, products, products_csv_path, summary):
    table_rows = [
        ("Metadata rows", str(len(metadata))),
        ("Metadata CSV", str(metadata_csv)),
        ("Resolved products", str(len(products))),
        ("Products CSV", str(products_csv_path)),
        ("Downloaded files", str(summary.files_completed)),
        ("Failed downloads", str(summary.files_failed)),
        ("Output directory", str(out)),
    ]
    header = "| Item | Value |\n|---|---|"
    body = "\n".join(f"| {k} | `{v}` |" for k, v in table_rows)
    mo.md(f"---\n## Summary\n\n{header}\n{body}")
    return


if __name__ == "__main__":
    app.run()
