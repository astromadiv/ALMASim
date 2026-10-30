# Downloads

This page describes ALMASim's data-download workflow.

## Location

The download service lives in
[`src/almasim/services/download`](../src/almasim/services/download):
`datalink.py` (resolve products, list the files inside their tars, filter
and select them), `catalog_io.py` (products CSV),
`transfer.py` (single files, retries, mirrors) and `orchestrate.py`
(`download_products`). The QA0 filter and the unpack and calibrate stages live
in [`src/almasim/services/archive`](../src/almasim/services/archive).

## Capabilities

ALMASim can:

- resolve the data products of member OUS UIDs through ALMA DataLink
- save resolved products to CSV and reload them
- filter them by type (`raw`, `calibration`, `auxiliary`, `qa_reports`,
  `scripts`, `weblog`, `cubes`, `continuum`, `fits`, `other`, or `all`)
- list the files inside each member's product and auxiliary tars, and pick
  single files by name, for example only the cleaned `*.pbcor.fits` images
  ([below](#choosing-single-files))
- skip the raw ASDMs of execution blocks ALMA never calibrated (below)
- download them across the ESO, NRAO and NAOJ mirrors, at most 10 transfers
  per mirror (`--max-parallel`, default 3, capped at 30)
- extract the tar archives, import the ASDMs into raw MeasurementSets and
  apply the delivered calibration, in this process or on Slurm

## Download model

1. query metadata ([Metadata](metadata.md))
2. resolve the products of the chosen observations
3. inspect or save the product list
4. download the selected products, then optionally extract, unpack and
   calibrate them

```bash
# resolve the products of a metadata CSV
almasim products resolve --metadata-csv metadata.csv \
  --save-products-csv resolved.csv

# download every product, extract the archives, and write calibrated
# MeasurementSets with 4 Slurm workers
almasim products download --products-csv resolved.csv \
  --product-filter all --destination downloads --extract-tar \
  --unpack-ms --generate-calibrated-visibilities \
  --backend slurm --slurm-workers 4 --yes
```

`almasim metadata --save-products-csv` resolves the products in the same run
as the query. A file already on disk at its full size is not downloaded again,
and an interrupted transfer resumes from its `.part` file. Each run writes
`download_manifest.json` in the destination folder. Raw and calibrated
MeasurementSets go under `--archive-output-root` (default
`<destination>/archive_ms`). Each stage can also run on its own: `almasim
visibilities extract`, `unpack` and `calibrate`. See
[`almasim products download`](cli.md#almasim-products-download) for every
option.

## Choosing single files

DataLink offers each member OUS as separate tars:

| Product type | DataLink semantics | File | Contents |
|---|---|---|---|
| `fits` | `#this` | `<project>_<member>_001_of_001.tar` (a large delivery is split into `_001_of_002.tar` and so on) | `product/`: the pipeline's images, each as `.pbcor.fits` (primary-beam corrected), `.pb.fits.gz` (primary beam) and `.mask.fits.gz` (clean mask), per spw cube, mfs and continuum image, for the science targets (`<field>_sci`) and the calibrators (for example `<field>_bp`, `<field>_ph`) |
| `auxiliary` | `#auxiliary` | `<project>_<member>_auxiliary.tar` | `calibration/`, `log/`, `qa/` and `script/` |
| `auxiliary` | `#documentation` | `member.<member>.README.txt` | the delivery's README |
| `raw` | `#progenitor`, `#package` | `<project>_<eb>.asdm.sdm.tar` | one execution block's raw ASDM |

`--product-filter` picks whole tars. `--expand-archives` replaces each
selected product and auxiliary tar by the files inside it, one row per file,
which keeps the tar's product type; ALMA lists those files through a DataLink
of the tar's own, one request per tar. `--include` and `--exclude` then pick
files by name with shell globs (`*`, `?`, `[seq]`, `[!seq]`, matched
case-sensitively against the whole filename). Repeat them or pass
comma-separated globs: a file is kept when it matches any `--include` (every
file, when there is none) and no `--exclude`.

```bash
# the science targets' cleaned, primary-beam-corrected cubes and continuum
# images, without the per-spw mfs images
almasim products download --products-csv resolved.csv \
  --product-filter fits --expand-archives \
  --include '*_sci.*.pbcor.fits' --exclude '*.mfs.*' \
  --destination images --yes
```

`almasim products resolve --expand-archives` saves the files instead of the
tars in the products CSV, so the list shows each file's size before anything is
downloaded.

The files go where the tars would put them (`--layout delivery`, the default):

```text
<destination>/<project>/science_goal.<uid>/group.<uid>/member.<uid>/
    member.<uid>.README.txt
    product/        the images
    calibration/    the auxiliary tar's calibration tables and calapply files
    qa/             the QA0 and QA2 reports and the weblog
    script/         the pipeline scripts
    log/            the pipeline's casa_commands.log
```

Each listed file's folder is the type its DataLink description names
(`product`, `calibration`, `qa`, `script`), and the log goes in `log/`. The
project, science goal and group are read from the first header of one tar
per member: a short read that stops after 16 KB. When that read fails, the
member's files go in a `member.<uid>` folder in the destination. The products
CSV records each file's folder in its `path` column. `--layout flat` puts
every file in `--destination` itself; each name starts with
`member.<member uid>.`. The member's tars, the raw ASDM tars and any file
that is not listed go in `--destination`, and `--extract-tar` unpacks a tar
into the same tree. The QA0 filter below reads the reports from the `qa/`
folders under `--qa0-report-root`, so the delivery layout serves it whether
the auxiliary tar was extracted or listed file by file.

## Skipping execution blocks ALMA never calibrated

Every execution block ships a `<eb>.qa0_report.pdf` in its member OUS `qa/`
directory, inside the *auxiliary* tar. Its **QA0 Status** decides whether ALMA
delivers calibration at all: `Pass` blocks always carry calibration products,
`SemiPass` blocks never do. ALMA still ships the raw ASDM of a SemiPass block,
so downloading raw products blindly pulls execution blocks (about 12 % in
Cycle 11) that unpack fine and then can never be calibrated.

`almasim products download --skip-qa0-semipass` reads those reports and drops
the SemiPass raw ASDMs before the transfer. The reports have to be on disk
first, so download in two passes:

```bash
# 1. auxiliary products first: small, and they carry the QA0 reports
almasim products download --products-csv resolved.csv \
  --product-filter auxiliary --destination downloads --extract-tar --yes

# 2. raw ASDMs, minus the SemiPass execution blocks
almasim products download --products-csv resolved.csv \
  --product-filter raw --destination downloads --extract-tar \
  --skip-qa0-semipass --yes
```

The reports are looked up under `--qa0-report-root` (default: `--destination`)
in the standard `<project>/science_goal.*/group.*/member.*/qa/` layout. A raw
product whose report is missing or unreadable is **kept**, never dropped, and
counted in the summary line, so a wrong root is visible immediately. Do not
use `ExecBlock Status` for this decision; it does not discriminate.

The library entry point is `almasim.services.archive.filter_products_by_qa0`.

Raw ASDMs downloaded without this filter are still safe: calibration skips an
execution block whose member OUS has a `calibration/` directory but no
`<eb>.ms.calapply.txt` for it, and reports it as delivered without
calibration rather than as a failure, so the run still succeeds. Its raw MS
is unpacked as usual. A member with no `calibration/` directory at all (the
auxiliary products were not downloaded) still fails.

## Python

```python
from almasim.services.download import (
    download_products,
    expand_archives,
    filter_products,
    resolve_products,
    save_products_csv,
    select_files,
)

products = resolve_products(metadata["member_ous_uid"].tolist())
save_products_csv(products, "resolved.csv")
summary = download_products(filter_products(products, "auxiliary"), "downloads", extract_tar=True)

# only the primary-beam-corrected images inside the product tars
images = select_files(expand_archives(filter_products(products, "fits")), ["*.pbcor.fits"])
download_products(images, "images")
```

## Web console

The console's [*Archive* page](console.md#archive) resolves and downloads
products by running the same `almasim products resolve` and `almasim products
download` commands as jobs.

## Example scripts

- [`examples/download_products_cli.py`](../examples/download_products_cli.py)
- [`examples/download_products_notebook.ipynb`](../examples/download_products_notebook.ipynb)
