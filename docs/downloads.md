# Downloads

This page describes ALMASim's data-download workflow.

## Location

The download service lives in
[`src/almasim/services/download`](../src/almasim/services/download):
`datalink.py` (resolve and filter products), `catalog_io.py` (products CSV),
`transfer.py` (single files, retries, mirrors) and `orchestrate.py`
(`download_products`). The QA0 filter and the unpack and calibrate stages live
in [`src/almasim/services/archive`](../src/almasim/services/archive).

## Capabilities

ALMASim can:

- resolve the data products of member OUS UIDs through ALMA DataLink
- save resolved products to CSV and reload them
- filter them by type (`raw`, `calibration`, `auxiliary`, `qa_reports`,
  `scripts`, `weblog`, `cubes`, `continuum`, `fits`, `other`, or `all`)
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

## Python

```python
from almasim.services.download import (
    download_products,
    filter_products,
    resolve_products,
    save_products_csv,
)

products = resolve_products(metadata["member_ous_uid"].tolist())
save_products_csv(products, "resolved.csv")
summary = download_products(
    filter_products(products, "auxiliary"), "downloads", extract_tar=True
)
```

## Web console

The console's [*Archive* page](console.md#archive) resolves and downloads
products by running the same `almasim products resolve` and `almasim products
download` commands as jobs.

## Example scripts

- [`examples/download_products_cli.py`](../examples/download_products_cli.py)
- [`examples/download_products_notebook.ipynb`](../examples/download_products_notebook.ipynb)
