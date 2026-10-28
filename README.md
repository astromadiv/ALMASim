# ALMASim

[![PyPI version](https://badge.fury.io/py/almasim.svg)](https://pypi.org/project/almasim/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Documentation](https://github.com/astromadiv/ALMASim/actions/workflows/docs.yml/badge.svg)](https://astromadiv.github.io/ALMASim/)
[![CI](https://github.com/astromadiv/ALMASim/actions/workflows/lint_and_test.yml/badge.svg)](https://github.com/astromadiv/ALMASim/actions/workflows/lint_and_test.yml)
[![codecov](https://codecov.io/gh/astromadiv/ALMASim/branch/main/graph/badge.svg)](https://codecov.io/gh/astromadiv/ALMASim)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

ALMASim is a **library-first** Python environment for simulating ALMA observations, exploring ALMA metadata, downloading science products, and building ML-ready radio/mm-wave datasets.

It provides reusable services in [`src/almasim`](src/almasim) that can be driven by the `almasim` command line, the web console, Jupyter notebooks, or direct Python code — all through the same staged API.

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Staged Simulation API](#staged-simulation-api)
- [Skymodels](#skymodels)
- [Compute Backends](#compute-backends)
- [Metadata and Downloads](#metadata-and-downloads)
- [Web Console](#web-console)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Key Capabilities

**Simulation**
- Build clean sky cubes from point, Gaussian, extended (Python API only), molecular-cloud, diffuse, Galaxy Zoo, and Hubble-100 sky models, or from your own FITS cube or image
- Simulate single-pointing ALMA interferometric observations with multi-configuration support (12m, 7m, TP)
- PWV-aware per-channel thermal noise, plus released noise models measured on real calibrated ALMA data (`--noise-model alma`, `almasim noise`)
- Additive astrophysical background sky (Python API) — faint dusty galaxies, diffuse emission, or combined
- Optional serendipitous source injection
- Wiener-filter preview reconstruction and TP+INT feather-style image combination

**Imaging**
- WSClean imaging of calibrated MeasurementSets, one task per spectral window, on Slurm: continuum images and cubes, continuum subtraction, self-calibration and primary-beam correction
- Clean-quality measurements per window (`almasim image clean-quality`)

**Data Products**
- Dirty cube, dirty visibilities, beam cube, UV mask cube, U/V coordinate cubes
- Interferometric, total-power, and combined TP+INT image cubes
- ML-ready HDF5 shards (clean cube + dirty cube + dirty visibilities + UV mask + metadata)
- Native MeasurementSet (`.ms`) export via CASA tools or python-casacore

**Metadata and Archive**
- Query ALMA observations via TAP with rich inclusion/exclusion filters
- Normalise TAP columns into stable application fields
- Resolve DataLink products, download ALMA data products with parallel support
- Unpack raw ASDMs into MeasurementSets
- Apply delivered calibration to produce calibrated science MeasurementSets

**Compute**
- Synchronous, local multiprocess, Dask, Slurm, and Kubernetes backends
- Backend-agnostic simulation service layer
- Every `almasim` command can run as a Slurm job (`almasim submit`, `almasim jobs`, the web console)

---

## Architecture

```
src/almasim/          ← installable library  (pip install almasim)
  services/
    simulation/       ← staged pipeline entry points
    interferometry/   ← UV sampling, baselines, noise, TP
    noise_model/      ← released noise models and their components
    noise/            ← measuring the noise of real ALMA data
    imaging/          ← WSClean batch imaging, quality, Wiener preview, TP+INT combination
    metadata/         ← TAP queries, normalisation
    download/         ← DataLink resolution, product downloads
    archive/          ← ASDM unpack, calibration apply
    ms/               ← MeasurementSet reading and writing
    products/         ← MS export, HDF5 shards, cube export
    compute/          ← backend abstraction
    astro/            ← spectral lines, redshift, parameters
  skymodels/          ← sky model implementations
  jobs/               ← runs of `almasim` commands as Slurm jobs (run folders, profiles)
  cli/                ← the `almasim` command line
  web/                ← the web console (`almasim serve`, API v2)
frontend/             ← the console's pages (SvelteKit, built into the wheel)
ood/almasim/          ← Open OnDemand interactive app for the console
examples/             ← CLI scripts and Jupyter notebooks
```

The library layer owns all domain logic. The CLI wraps the library services, and the web console runs CLI commands as jobs. Scripts and notebooks call the same staged services directly.

---

## Installation

ALMASim needs Python 3.12.

### From PyPI

The package installs the library and the `almasim` command:

```bash
pip install almasim
almasim --help
```

Optional extras: `[web]` for the web console (`almasim serve`), `[casa]` and `[ms-casacore]` for MeasurementSet work (below).

### From source

To work from a local clone, install the project into a virtual environment with [uv](https://docs.astral.sh/uv/) and run the CLI through it:

```bash
git clone https://github.com/astromadiv/ALMASim.git
cd ALMASim
pip install uv
uv sync --group dev
uv run almasim --help
```

For an editable install that puts `almasim` on your shell path, run `pip install -e .` in the clone instead.

### The `almasim` command

Once installed, the CLI exposes the main workflows:

```bash
almasim metadata --help      # query ALMA metadata (TAP)
almasim products --help      # resolve and download archive products
almasim visibilities --help  # extract, unpack and calibrate MeasurementSets; flag summaries
almasim image --help         # imaging parameters, WSClean imaging, quality, primary-beam correction
almasim predict --help       # predict model visibilities into MeasurementSets
almasim simulation --help    # simulate ALMA observations
almasim preview --help       # quick-look FITS previews: integrated maps, Wiener reconstructions
almasim noise --help         # released noise models; measuring real data's noise
almasim clean --help         # run WSClean with extra arguments passed through
almasim submit --help        # run any almasim command as a Slurm job
almasim jobs --help          # list, follow and control those runs
almasim serve --help         # the web console
almasim doctor --help        # check what jobs need (Python, CASA, WSClean, Slurm)
```

Every command and option is listed in the [command-line reference](https://astromadiv.github.io/ALMASim/cli.html) (`docs/cli.md`), generated from
the command definitions (`python scripts/generate_cli_reference.py`).

### With CASA tools (Linux x86-64 only)

`casatools` and `casatasks` wheels are Linux-only. Install the optional `[casa]` extra on a supported Linux system:

```bash
pip install "almasim[casa]"
```

The `[casa]` extra enables:
- Native MeasurementSet export via `casatools`
- ASDM-to-MS conversion via `casatasks.importasdm`
- Calibration application via `casatasks.applycal`
- Self-calibration and uv-plane continuum subtraction in `almasim image image-from-ms` (`--selfcal`, `--uv-contsub`)

Without `[casa]`, simulation, metadata and downloads still work. WSClean imaging (`almasim image`) reads the MeasurementSets with `python-casacore`, and the MS export path falls back to it when CASA is missing:

```bash
pip install "almasim[ms-casacore]"
```

### Web console with Docker Compose (laptop)

On a cluster, install the package and run `almasim serve` ([Web Console](#web-console)). On a laptop without Slurm, Docker Compose runs the console and its pages, with every command running in the console's container:

```bash
git clone https://github.com/astromadiv/ALMASim.git
cd ALMASim
docker compose up
```

`docker compose up` reads `docker-compose.yml` and the development
`docker-compose.override.yml` (hot reload for the console and the frontend).
For the production builds, layer `docker-compose.prod.yml` on the base file
instead: `make start-prod`, or
`docker compose -f docker-compose.yml -f docker-compose.prod.yml up`. Then open http://localhost:3000.
`make help` lists the other targets, including `make test`, `make lint` and
`make typecheck` for local checks.

The console image is available pre-built from GHCR:

```bash
docker pull ghcr.io/astromadiv/almasim-console:latest
```

---

## Quick Start

### From the command line

```bash
# Band 6 galaxy observations from the ALMA archive, saved as a metadata CSV
almasim metadata --science-keyword Galaxies --band 6 --limit 10 \
  --save-csv output/metadata.csv --yes

# Simulate the first row (a point source by default) into an ML-ready HDF5 shard;
# add --save-mode fits --persist-standard-outputs to keep the FITS cubes as well
almasim simulation --metadata-path output/metadata.csv --row-idx 0 \
  --project-name demo --output-dir output/simulation --ml-shard-path output/demo.h5
```

### Query ALMA metadata

```python
from almasim.services.metadata.tap import InclusionFilters, query_metadata_by_science

df = query_metadata_by_science(include=InclusionFilters(science_keyword=["Galaxies"], band=[6]))
print(df[["ALMA_source_name", "Band", "Ang.res."]].head())
```

`query_metadata_by_science` returns the normalised columns a simulation reads; `query_by_science_type` (in `almasim.services.metadata.tap.service`) returns the raw TAP columns.

### Run a simulation from a metadata row

```python
from pathlib import Path

import almasim
from almasim import SimulationParams, run_simulation

out = Path("output")
params = SimulationParams.from_metadata_row(
    df.iloc[0],  # a row of the metadata query above
    idx=0,
    main_dir=Path(almasim.__file__).parent,  # the package's data files
    output_dir=out,
    tng_dir=out / "tng",
    galaxy_zoo_dir=out / "galaxy_zoo",
    hubble_dir=out / "hubble",
    project_name="my_project",
)

result = run_simulation(params)
```

### Use the staged API

```python
from almasim import (
    SimulationParams,
    generate_clean_cube,
    simulate_observation,
    image_products,
    export_results,
)

params = SimulationParams.from_metadata_row(row, idx=0, ...)

clean_cube_stage   = generate_clean_cube(params)
simulation_results = simulate_observation(clean_cube_stage)
simulation_results = image_products(simulation_results)
exported           = export_results(params, clean_cube_stage, simulation_results)
```

---

## Staged Simulation API

The pipeline is split into four composable stages:

| Stage | Function | What it does |
|---|---|---|
| 1 | `generate_clean_cube()` | Build the sky cube from the sky model, add the background |
| 2 | `simulate_observation()` | Run the interferometric + TP simulation, return the dirty products |
| 3 | `image_products()` | Build the image cubes: Wiener reconstruction, INT+TP combination |
| 4 | `export_results()` | Write cubes, ML shards and parameter summaries to disk |

`run_simulation()` orchestrates all four in sequence.

`write_ml_dataset_shard()` exports an HDF5 shard (clean cube + dirty cube + dirty visibilities + UV mask + metadata) independently of the main export path.

`estimate_simulation_footprint()` (in `almasim.services.simulation`) returns the resolved pixel count, channel count, cell size, beam size, and raw output size in GiB — useful for pre-run capacity checks.

Full reference: [Simulation docs](https://astromadiv.github.io/ALMASim/simulation.html)

---

## Skymodels

| Source type | Description |
|---|---|
| `point` | Point source — PSF and CLEAN validation |
| `gaussian` | 2-D Gaussian — compact extended source |
| `extended` | TNG-backed realistic extended emission (Python API only, not in the CLI) |
| `galaxy-zoo` | Galaxy Zoo image morphology prior |
| `hubble-100` | Hubble Top-100 image morphology prior |
| `molecular` | Molecular cloud structured emission |
| `diffuse` | Correlated diffuse emission field |
| `external-fits-cube`, `external-fits-image` | Your own FITS cube or image (`almasim simulation --input`), resampled to the run's grid and given a header built from the metadata row |

The `point` and `gaussian` sky models accept `source_offset_x_arcsec` / `source_offset_y_arcsec` (Python API only) to shift the science target from the phase center.

**Additive background sky** (independent of the main source; set `background_mode` and `background_level` in `SimulationParams` — the CLI's `--background-level` does not add one yet):

| Mode | Effect |
|---|---|
| `blank_field_dsfg` | Faint dusty star-forming galaxies |
| `dusty_diffuse` | Correlated low-spatial-frequency dusty background |
| `combined` | Both of the above |

Full reference: [Skymodels docs](https://astromadiv.github.io/ALMASim/skymodels.html)

---

## Compute Backends

Create a backend with `almasim.services.compute.create_backend(...)` and pass it as `compute_backend=` to `run_simulation()` or the stage functions (without one, the channels are imaged in-process). On the command line, `almasim simulation --backend sync|local` chooses between the first two.

| Backend | Use case |
|---|---|
| `sync` | Notebooks, examples, debugging |
| `local` | Local CPU parallelism |
| `dask` | Distributed execution, cluster scheduling |
| `slurm` | HPC job submission |
| `kubernetes` | Cluster-native environments |

Full reference: [Compute docs](https://astromadiv.github.io/ALMASim/compute.html)

---

## Metadata and Downloads

### Query metadata via TAP

```python
from almasim.services.metadata.tap import (
    query_metadata_by_science,
    InclusionFilters,
    ExclusionFilters,
)

df = query_metadata_by_science(
    include=InclusionFilters(
        science_keyword=["Galaxies"],
        band=[6, 7],
        public_only=True,
        science_only=True,
    ),
    exclude=ExclusionFilters(solar=True),
)
```

### Download products

```python
from pathlib import Path

from almasim.services.download import download_products, resolve_products

products = resolve_products(df["member_ous_uid"].tolist())
summary = download_products(products, destination=Path("downloads"), extract_tar=True)
```

Full reference: [Metadata docs](https://astromadiv.github.io/ALMASim/metadata.html) · [Downloads docs](https://astromadiv.github.io/ALMASim/downloads.html)

---

## Web Console

`almasim serve` runs the web console on a cluster's login node: one place to query the ALMA archive, start simulations, image MeasurementSets, run any `almasim` command as a Slurm job, follow jobs live and look at the products. Each user runs their own console and reaches it through an SSH tunnel:

```bash
# on the cluster's login node (in tmux or screen, so it outlives the SSH session)
pip install "almasim[web]"     # or the site's shared install
almasim doctor --slurm         # checks, on a compute node, what jobs need
almasim serve                  # prints an ssh -L line and a link with your token

# on your laptop: the printed tunnel, then open the printed link
ssh -N -L 8765:/home/you/.almasim/serve.sock you@cluster.login
```

The console listens on a socket only you can open and asks for your token. `almasim serve --status` prints the link again and `almasim serve --stop` stops it; your jobs carry on either way.

On a site with Open OnDemand, the console can instead start from the portal as an interactive app, with no tunnel: see [`ood/almasim`](ood/almasim/README.md).

![The web console](https://raw.githubusercontent.com/astromadiv/ALMASim/main/docs/images/console/home.png)

The console serves its pages and API v2 (`/api/v2`, OpenAPI docs at `/api/docs`) together, so a cluster needs no Node and no database: the runs live in run folders a team can share.

Full reference: [Web console docs](https://astromadiv.github.io/ALMASim/console.html)

---

## Examples

The example scripts use the `sync` compute backend and need no running scheduler; the `almasim` commands below show the same workflows, some of them on Slurm.

| Script | Description |
|---|---|
| [`examples/query_metadata_cli.py`](examples/query_metadata_cli.py) | Query TAP, export metadata and product CSVs |
| [`examples/download_products_cli.py`](examples/download_products_cli.py) | Resolve and download ALMA products |
| [`examples/archive_ms_cli.py`](examples/archive_ms_cli.py) | Unpack ASDMs and apply calibration |
| [`examples/staged_pipeline_cli.py`](examples/staged_pipeline_cli.py) | Full pipeline: query → simulate → ML shard |

```bash
# Query metadata for Band 6 galaxy observations
almasim metadata \
  --science-keyword Galaxies --band 6 \
  --save-csv examples/output/metadata.csv

# Extract member_ous_uid values and resolve DataLink products
almasim products resolve \
  --metadata-csv examples/output/metadata.csv \
  --save-member-ous-uid-list examples/output/member_ous_uids.txt \
  --save-products-csv examples/output/resolved_products.csv

# Download selected product types and extract archives
almasim products download \
  --products-csv examples/output/resolved_products.csv \
  --product-filter all \
  --destination examples/output/downloads \
  --extract-tar

# Two-pass download that skips execution blocks ALMA never calibrated: the
# auxiliary tar carries each EB's QA0 report, and a QA0 "SemiPass" EB ships a raw
# ASDM but no calibration products, so it can never be calibrated (see docs/downloads.md)
almasim products download \
  --products-csv examples/output/resolved_products.csv \
  --product-filter auxiliary \
  --destination examples/output/downloads \
  --extract-tar --yes
almasim products download \
  --products-csv examples/output/resolved_products.csv \
  --product-filter raw \
  --destination examples/output/downloads \
  --extract-tar --skip-qa0-semipass --yes

# Split archive stages so download/extract/unpack/calibrate can run independently
# (`almasim products extract/unpack/calibrate` remain as hidden aliases)
almasim products download \
  --products-csv examples/output/resolved_products.csv \
  --destination examples/output/downloads

almasim visibilities extract \
  --source-root examples/output/downloads

almasim visibilities unpack \
  --input-root examples/output/downloads \
  --output-dir examples/output/archive_ms/raw_ms \
  --backend slurm \
  --slurm-workers 8

almasim visibilities calibrate \
  --input-root examples/output/downloads \
  --raw-ms-root examples/output/archive_ms/raw_ms \
  --output-dir examples/output/archive_ms/calibrated_ms \
  --backend slurm \
  --slurm-workers 8

# Unpack/calibrate run every UID in its own subprocess (sync and slurm alike), so a
# CASA crash on one execution block is skipped and the rest of the batch continues.
# Skipped UIDs are listed at the end and the command exits with status 1; pass
# --fail-fast to abort on the first failure instead.
#
# Every finished calibration gets a marker <output-root>/<uid>.ms.split.cal.done; on a
# re-run UIDs with a marker are skipped automatically (--overwrite-outputs redoes them),
# while partial outputs or working copies left by a crash are replaced. To trust an
# output produced before markers existed, create its .done file by hand (touch).
# The per-UID working copy (raw MS + caltables, ~1.8x the raw MS) is deleted as soon as
# the split succeeds; pass --keep-working-copies to retain it.
#
# Following progress: each UID writes <output-root>/logs/<uid>.calibrate.log
# (or <uid>.unpack.log) with the full CASA output, so on Slurm you can
#   tail -f examples/output/archive_ms/calibrated_ms/logs/*.calibrate.log
# from the submit node. The submit node also prints the last log line of every
# running UID once a minute. Dask worker stdout/stderr lands in ~/dask-worker-logs.

# Download + unpack ASDM + calibrate using Slurm-backed parallel post-processing
almasim products download \
  --products-csv examples/output/resolved_products.csv \
  --destination examples/output/downloads \
  --extract-tar \
  --unpack-ms \
  --generate-calibrated-visibilities \
  --backend slurm \
  --slurm-queue normal \
  --slurm-workers 8

# Image every calibrated MS, one WSClean per science spectral window, on Slurm.
# Same contract as unpack/calibrate: SPW-<n>.done / .failed / .log next to each
# output directory, failed tasks never stop the batch, reruns resume from the markers
# (see docs/imaging.md)
almasim image compute-parameters examples/output/archive_ms/calibrated_ms imaging_parameters.csv
almasim image image-from-ms imaging_parameters.csv examples/output/images \
  --num-cores 10 --slurm-queue normal --slurm-workers 8

# Run WSClean through ALMASim (all WSClean flags are forwarded unchanged)
almasim clean -- \
  -name examples/output/imaging/demo \
  -size 1024 1024 \
  -scale 0.1asec \
  -niter 20000 \
  examples/output/archive_ms/calibrated_ms/*.ms.split.cal

# Run a staged simulation from the first row of the metadata CSV
almasim simulation \
  --metadata-path examples/output/metadata.csv \
  --row-idx 0 \
  --project-name demo \
  --ml-shard-path examples/output/demo.h5

# The same with the example scripts: query metadata for Band 6 galaxy observations
python examples/query_metadata_cli.py \
  --science-keyword Galaxies --band 6 \
  --save-csv examples/output/metadata.csv

# ...then run a staged simulation from the first metadata row
python examples/staged_pipeline_cli.py \
  --metadata-mode csv --metadata-path examples/output/metadata.csv \
  --row-idx 0 --project-name demo \
  --ml-shard-path examples/output/demo.h5
```

Notebook equivalents: [`staged_pipeline_notebook.ipynb`](examples/staged_pipeline_notebook.ipynb) · [`query_metadata_notebook.ipynb`](examples/query_metadata_notebook.ipynb) · [`download_products_notebook.ipynb`](examples/download_products_notebook.ipynb)

### End-to-end archive pipeline (Marimo)

[`examples/e2e_archive_pipeline.py`](examples/e2e_archive_pipeline.py) is a reactive [Marimo](https://marimo.io) notebook that covers the full archive workflow interactively: query ALMA metadata → resolve DataLink products → download → unpack ASDMs → apply calibration.

```bash
# Install dev dependencies (includes marimo)
uv sync --group dev

# Interactive editing mode — cells re-run automatically as you edit
marimo edit examples/e2e_archive_pipeline.py

# Read-only app mode — run the pipeline step-by-step via the UI
marimo run examples/e2e_archive_pipeline.py
```

Steps 4 (unpack) and 5 (calibrate) require CASA tools (Linux x86-64 only):

```bash
pip install "almasim[casa]"
```

The notebook saves query filter presets as `.query.json` files so they can be reloaded across sessions.

---

## Documentation

Full documentation: **[astromadiv.github.io/ALMASim](https://astromadiv.github.io/ALMASim/)**

| Section | Topics |
|---|---|
| [Quick Start](https://astromadiv.github.io/ALMASim/quickstart.html) | Installation, first simulation |
| [Command-line reference](https://astromadiv.github.io/ALMASim/cli.html) | Every `almasim` command and option |
| [Web Console](https://astromadiv.github.io/ALMASim/console.html) | `almasim serve`: pages, jobs, shared runs |
| [Simulation](https://astromadiv.github.io/ALMASim/simulation.html) | Staged API, SimulationParams, outputs |
| [Interferometry](https://astromadiv.github.io/ALMASim/interferometry.html) | UV sampling, baselines, multi-config |
| [Noise](https://astromadiv.github.io/ALMASim/noise.html) | PWV-aware thermal noise and the noise models, in brief |
| [Noise Models](https://astromadiv.github.io/ALMASim/noise_models.html) | The noise models in depth: measuring, building and releasing one |
| [Background Sky](https://astromadiv.github.io/ALMASim/background.html) | Additive astrophysical background |
| [Skymodels](https://astromadiv.github.io/ALMASim/skymodels.html) | Sky models reference |
| [Imaging](https://astromadiv.github.io/ALMASim/imaging.html) | WSClean batch imaging, TP+INT combination |
| [Metadata](https://astromadiv.github.io/ALMASim/metadata.html) | TAP queries, filters |
| [Downloads](https://astromadiv.github.io/ALMASim/downloads.html) | Product download workflow |
| [Compute Backends](https://astromadiv.github.io/ALMASim/compute.html) | Sync, Dask, Slurm, Kubernetes |
| [Slurm Scheduler](https://astromadiv.github.io/ALMASim/slurm_scheduler.html) | The Slurm-backed Dask cluster behind `--backend slurm` |

Build docs locally:

```bash
uv sync --group dev
uv run sphinx-build -b html docs/source docs/build/html
```

---

## Contributing

```bash
git clone https://github.com/astromadiv/ALMASim.git
cd ALMASim
uv sync --group dev
make check        # ruff format --check, ruff check, mypy and pytest, as CI runs them
make format       # apply ruff's formatting and automatic fixes
```

A release is published automatically when a version tag is pushed:

```bash
# 1. Bump the version in pyproject.toml (the only place it is written)
# 2. Commit and tag
git tag vX.Y.Z
git push origin vX.Y.Z
```

The release pipeline then:
1. Validates that the tag matches `pyproject.toml`
2. Runs the full lint + test suite
3. Builds the web app into the wheel and sdist, and publishes them to PyPI via OIDC trusted publisher
4. Creates a GitHub Release with auto-generated changelog and attached artifacts
5. Builds and pushes the console and frontend Docker images to GHCR

> **One-time PyPI setup:** register a [trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) on PyPI with owner `astromadiv`, repo `ALMASim`, workflow `release.yml`, environment `pypi`.

---

## License

ALMASim is released under the [GNU General Public License v3](LICENSE.md).
