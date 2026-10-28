# Metadata

This page describes ALMASim's ALMA metadata workflows.

## Location

Metadata code lives in:

- [`src/almasim/services/metadata`](../src/almasim/services/metadata): the
  filter model (`filters.py`), the execution-block counts (`executions.py`),
  saved query presets (`saved_queries.py`) and CSV output (`storage.py`)
- [`src/almasim/services/metadata/tap`](../src/almasim/services/metadata/tap):
  the TAP connection, the ADQL builder and the high-level queries

## Capabilities

ALMASim can:

- query the ALMA archive's ObsCore table through TAP, trying the ESO, NAOJ and
  NRAO mirrors in turn and retrying failed searches
- filter by source, science keyword and category, band, array type and
  configuration, angular resolution, date, QA2 status, cycle, project code and
  more (`InclusionFilters`), and exclude science keywords, categories, sources,
  observation types or solar observations (`ExclusionFilters`); by default
  only public science data without mosaics
- normalize TAP columns into stable application fields (`ALL_COLUMNS`), one
  row per member OUS
- infer the array type (`12m`, `7m`, `TP` and their combinations) from the
  antenna list
- count the execution blocks behind each row (`n_executions`)
- export metadata rows to CSV
- feed metadata rows directly into simulations

## Query interfaces

- **Command line**: `almasim metadata` takes the filters as options and
  writes the normalized rows with `--save-csv`. `--save-products-csv` also
  resolves the rows' data products for
  [`almasim products download`](downloads.md), and `--count-executions` adds
  `n_executions`. See [`almasim metadata`](cli.md#almasim-metadata).
- **Python**: `almasim.services.metadata.tap` exports
  `query_metadata_by_science(include=..., exclude=..., save_to=...)`,
  `query_metadata_by_targets`, `query_products` and `load_metadata`.
- **Web console**: the [*Archive* page](console.md#archive) runs the same
  query as a job.
- **Examples**: [`examples/query_metadata_cli.py`](../examples/query_metadata_cli.py)
  and [`examples/query_metadata_notebook.ipynb`](../examples/query_metadata_notebook.ipynb).

## Execution blocks

A row's `Int.Time` is the member OUS's total time on source, summed over its
execution blocks; ObsCore does not say how many blocks there were. DataLink
lists one raw ASDM per block, so `almasim metadata --count-executions` or
`almasim products executions --metadata-path <csv>` adds `n_executions` to
each row (one DataLink request per member OUS). `almasim simulation` observes
`Int.Time` in that many blocks; without the column it uses blocks of at most
3000 s.

## Metadata-driven simulation

`SimulationParams.from_metadata_row()` takes from a metadata row:

- target name and coordinates
- ALMA band, frequency, bandwidth and frequency support
- observation date and PWV
- integration time (and `n_executions`, when present)
- field of view
- angular and velocity resolution
- continuum and line sensitivity
- antenna arrays, which the observation plan splits into 12 m, 7 m and TP
  configurations

The metadata row is the main source of truth: `almasim simulation` and the
console's *Simulate* page take the observation from it, and their own options
set only what the row does not (the sky model, cube size, noise model and so
on).
