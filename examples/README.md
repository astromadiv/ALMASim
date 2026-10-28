# ALMASim examples

This directory holds runnable examples of the ALMASim Python API and CLI:

- `query_metadata_cli.py`
  Queries ALMA metadata from TAP, saves it as CSV, and optionally saves the DataLink product rows of the first queried member OUSs (`--product-member-limit`, default 1).
- `query_metadata_notebook.ipynb`
  Notebook version of the same query: preview the results and export the metadata and product rows as CSV for later simulation or download.
- `download_products_cli.py`
  Resolves DataLink products from `member_ous_uid` values (a metadata CSV or `--member-ous-uid`) or loads a saved products CSV, then downloads them. It can also extract the tarballs, import the ASDMs into raw MeasurementSets and write calibrated MeasurementSets.
- `download_products_notebook.ipynb`
  Notebook that resolves products from queried metadata, saves them as CSV and selects the files to download (the download call itself is left commented out).
- `archive_ms_cli.py`
  Imports raw ALMA ASDMs into MeasurementSets, then applies the delivered calibration and writes calibrated science MeasurementSets.
- `e2e_archive_pipeline.py`
  [Marimo](https://marimo.io) notebook for the whole archive workflow: query, resolve, download, unpack and calibrate, with query presets you can save and reload. The unpack and calibrate steps need CASA (`pip install "almasim[casa]"`).
- `staged_pipeline_cli.py`
  Queries metadata or loads a CSV, then runs clean-cube generation, the observation simulation and the ML (HDF5) shard export from the command line.
- `staged_pipeline_notebook.ipynb`
  Notebook version of the same pipeline, starting from a live metadata query, for interactive exploration and DDRM training data.
- `slurm_cluster_submit_demo.py`
  Submits a shell command through ALMASim's Slurm Dask cluster and prints `squeue`, `sacct` and `scontrol` details to debug the workers.
- `ImagingWithDucc0.py`
  Experimental Marimo notebook that grids a Stokes I dirty image and PSF from an MSv4 dataset with `ducc0`, writes them as `wsclean_like_image.fits` and `psf.fits`, and deconvolves with Radler. It needs `radler`, `xarray`, `holoviews` and `matplotlib`, which ALMASim does not install.
- `clean.py`
  Runs Radler deconvolution on the `wsclean_like_image.fits` and `psf.fits` that `ImagingWithDucc0.py` writes in the current directory, and saves the model and residual images as FITS.
- `site-profiles/`
  Site profiles for the web console and job submission (`ALMASIM_SITE_PROFILE`): `laptop.toml` for a machine without Slurm or `docker compose up`, `zwicky.toml` for the zwicky cluster.

The staged pipeline examples use the in-process `sync` compute backend, so they need no Dask cluster, process pool or scheduler.

The same archive workflow with the `almasim` CLI:

```bash
almasim metadata \
  --science-keyword Galaxies \
  --band 6 \
  --save-csv examples/output/metadata_query_results.csv

almasim products resolve \
  --metadata-csv examples/output/metadata_query_results.csv \
  --save-products-csv examples/output/resolved_products.csv

almasim products download \
  --products-csv examples/output/resolved_products.csv \
  --destination examples/output/downloads \
  --extract-tar

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

almasim clean -- \
  -name examples/output/imaging/demo \
  -size 1024 1024 \
  -scale 0.1asec \
  -niter 20000 \
  examples/output/archive_ms/calibrated_ms/*.ms.split.cal
```

`almasim visibilities extract` is only needed for archives downloaded without `--extract-tar`. See [docs/cli.md](../docs/cli.md) for every command and option.

Metadata query script usage:

```bash
python examples/query_metadata_cli.py \
  --science-keyword Galaxies \
  --band 6 \
  --save-csv examples/output/metadata_query_results.csv \
  --save-products-csv examples/output/metadata_products.csv
```

Download script usage:

```bash
python examples/download_products_cli.py \
  --metadata-csv examples/output/metadata_query_results.csv \
  --member-limit 1 \
  --product-filter all \
  --save-products-csv examples/output/resolved_products.csv \
  --destination examples/output/downloads \
  --extract-tar
```

Download, unpack raw MeasurementSets and write calibrated visibilities:

```bash
python examples/download_products_cli.py \
  --products-csv examples/output/resolved_products.csv \
  --destination examples/output/downloads \
  --extract-tar \
  --unpack-ms \
  --generate-calibrated-visibilities \
  --archive-output-root examples/output/archive_ms
```

Archive MeasurementSet script usage:

```bash
python examples/archive_ms_cli.py \
  --input-root /path/to/2023.1.01196.S \
  --output-root examples/output/archive_ms
```

Archive pipeline notebook:

```bash
marimo edit examples/e2e_archive_pipeline.py   # edit interactively
marimo run examples/e2e_archive_pipeline.py    # run as an app
```

Staged pipeline script usage:

```bash
python examples/staged_pipeline_cli.py \
  --science-keyword Galaxies \
  --band 6 \
  --row-idx 0 \
  --project-name ddrm_demo \
  --save-metadata-csv examples/output/staged_pipeline_metadata.csv \
  --ml-shard-path examples/output/ddrm_training_sample.h5
```

Slurm submit demo usage (from a machine that can submit Slurm jobs):

```bash
python examples/slurm_cluster_submit_demo.py \
  --command "echo hello" \
  --cores 2 \
  --queue normal
```
