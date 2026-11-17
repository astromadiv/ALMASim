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
- with each picked `*.pbcor.fits` image, download its primary beam and write
  the image before primary-beam correction
  ([below](#primary-beams-and-the-images-before-correction))
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
  --product-filter all --destination data --extract-tar \
  --unpack-ms --generate-calibrated-visibilities \
  --backend slurm --slurm-workers 4 --yes
```

`almasim metadata --save-products-csv` resolves the products in the same run
as the query. A file already on disk at its full size is not downloaded again,
and an interrupted transfer resumes from its `.part` file. Each run writes
`download_manifest.json` in the destination folder. The destination is the
data folder, `~/almasim_outputs/data` unless `--destination` names another
(a shared folder, say), and `visibilities extract`, `unpack` and `calibrate`
read the same folder by default. Each raw and calibrated
MeasurementSet goes into its member folder of the delivery tree in the
destination (see [The delivery tree](#the-delivery-tree));
`--archive-output-root` puts them in `raw_ms/` and `calibrated_ms/` of a folder
of their own instead. Each stage can also run on its own: `almasim
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

### Primary beams and the images before correction

With each `<name>.pbcor.fits` image that `--include` and `--exclude` pick,
`products download` also downloads its primary beam, `<name>.pb.fits.gz` from
the same product tar, unless an `--exclude` glob names it. After the
download it writes `<name>.image.fits` next to them: the corrected image
times the primary beam, the image before primary-beam correction, whose
noise is flat across the field. Pixels the pipeline blanked below its
primary-beam limit stay blank. Each image keeps the corrected image's header
and its `BEAMS` table, with a `HISTORY` card naming both inputs. The product
is taken one channel at a time, so a cube never has to fit in memory. An
image newer than both its inputs is left as it is, so a second run over the
same destination writes only the missing ones; the summary counts the images
written, those already there and the corrected images without a primary
beam. Pbcor images that `--extract-tar` unpacks from whole product tars get
their images too. `--no-pb` downloads only the picked files.

```bash
# the science targets' continuum images, their primary beams, and the
# continuum images before primary-beam correction (*.cont.I.image.fits)
almasim products download --products-csv resolved.csv \
  --product-filter fits --expand-archives \
  --include '*.cont.I.pbcor.fits' --destination images --yes
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

## The delivery tree

Every image and MeasurementSet ALMASim downloads or makes lives in the ALMA
delivery tree, in the folder of its member OUS, below its project, science
goal and group:

```text
<root>/ous_index.csv
<root>/<project>/science_goal.<uid>/group.<uid>/member.<uid>/
    product/  calibration/  qa/  script/  log/          as ALMA delivers them
    raw/<eb>.asdm.sdm                                   the ASDMs, as ALMA delivers them
    raw/<eb>.ms  raw/<eb>.ms.flagversions               the raw MSs, next to their ASDMs
    calibrated/<eb>.ms.split.cal                        the calibrated MSs
    images/SPW-<n>_<field>/wsclean-*.fits               image-from-ms --by-member
    images/<eb>.ms.split/SPW-<n>_FIELD-<f>/             image-from-ms of one EB's MS
```

The extracted tars carry the tree in their paths, and `--layout delivery`
puts single files at the same paths. Each member's `qa/` folder
holds one `<eb>.qa0_report.pdf` per execution block, its `calibration/`
folder one `<eb>.ms.calapply.txt` per calibrated block, and its `raw/`
folder each downloaded ASDM, so the tree says which member each execution
block belongs to. `visibilities unpack` writes each raw MS next to its ASDM,
and `visibilities calibrate` each calibrated MS into the member's
`calibrated/` folder, where ALMA's scriptForPI puts it, with the stage's
`.done` or `.failed` marker beside it; `products download` does both. An
execution block no member lists goes to `<root>/working/<eb>.ms` (raw) or
`<root>/<eb>.ms.split.cal` (calibrated). `--output-dir` (and `--raw-ms-root`)
give a stage a folder of its own instead, with every MS in it; the CI
regression chain runs that way.

`visibilities calibrate` and `products download` then rewrite
`ous_index.csv` at the root of the tree. The index has one row per
execution block: `eb_uid`, `project_code`, `science_goal_uid`,
`group_ous_uid`, `member_ous_uid`, `member_folder` (relative to the index),
`alma_calibrated` (`yes` when the member has the block's calapply file) and
the paths of its `raw_ms` and `calibrated_ms`, those in its member folder
first and else those in the stages' own folders. A calibrated MS in such a
folder whose execution block no member lists is reported.

An image-from-ms output directory that is the root of the tree (it holds
`ous_index.csv`) puts each imaging task in its member's `images/` folder
(see [Imaging a member](imaging.md#imaging-a-member-from-all-its-execution-blocks)),
and `image compute-parameters` given the root reads the members'
`calibrated/` folders.

The library entry point is `almasim.services.archive.write_ous_index`, and
`read_ous_index` reads the index back as `{eb_uid: row}`. `member_status`
reads it member by member. A member is complete when every execution block
ALMA calibrated for it has a calibrated MS. Only a complete member is imaged
as a whole
(see [Imaging a member](imaging.md#imaging-a-member-from-all-its-execution-blocks)).
`image image-from-ms` takes the OUSs of every FITS header it writes from it
(see [Provenance in the FITS headers](imaging.md#provenance-in-the-fits-headers)).

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
  --product-filter auxiliary --destination data --extract-tar --yes

# 2. raw ASDMs, minus the SemiPass execution blocks
almasim products download --products-csv resolved.csv \
  --product-filter raw --destination data --extract-tar \
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
    with_primary_beams,
    write_uncorrected_images,
)

products = resolve_products(metadata["member_ous_uid"].tolist())
save_products_csv(products, "resolved.csv")
summary = download_products(filter_products(products, "auxiliary"), "downloads", extract_tar=True)

# only the primary-beam-corrected images inside the product tars, with their
# primary beams, then the images before primary-beam correction
listing = expand_archives(filter_products(products, "fits"))
images = with_primary_beams(select_files(listing, ["*.pbcor.fits"]), listing)
summary = download_products(images, "images")
write_uncorrected_images("images", [f.filename for f in summary.files if f.status == "completed"])
```

## Web console

The console's [*Archive* page](console.md#archive) resolves and downloads
products by running the same `almasim products resolve` and `almasim products
download` commands as jobs.

## Example scripts

- [`examples/download_products_cli.py`](../examples/download_products_cli.py)
- [`examples/download_products_notebook.ipynb`](../examples/download_products_notebook.ipynb)
