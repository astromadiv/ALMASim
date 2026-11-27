# Imaging and Combination

This page describes ALMASim's image-domain products and its imaging of
MeasurementSets with WSClean.

## Location

Imaging code lives in [`src/almasim/services/imaging`](../src/almasim/services/imaging):

| module | contents |
|---|---|
| `reconstruction/` | the in-memory image products of a simulation: Wiener deconvolution and preview, TP/INT merging, cube regridding |
| `ms_geometry.py` | `compute_imaging_parameters`: the WSClean geometry of every (spectral window, field) of an MS |
| `wsclean_plan.py`, `wsclean/` | WSClean command lines, the cube channel plan, the per-task run |
| `member_plan.py` | a member's parameter rows merged per window and field, for `--by-member` |
| `batch_runner.py`, `task.py`, `markers.py` | running the tasks on `sync` or `slurm`, their `.done`/`.failed`/`.log` files |
| `ms/` | the single-window MS each task images, and a member's combined one |
| `uv_contsub.py`, `selfcal/`, `primary_beam.py` | visibility-plane continuum subtraction, self-calibration, primary-beam correction |
| `cont_dat.py` | the ALMA pipeline's line channels (`cont.dat`) for `--pipeline-line-free` |
| `provenance.py` | the OUSs, execution blocks and WSClean commands in every product's FITS header |
| `quality/` | the cleaning-quality statistics and moment maps of `image clean-quality` |

## Simulation image products

`build_image_products()` in
[`reconstruction/`](../src/almasim/services/imaging/reconstruction) turns a
simulation's dirty and beam cubes into:

- the interferometric image cube (Wiener-deconvolved, one per array
  configuration too)
- the total-power image cube
- the TP+INT combined cube (feather-style merge; the INT or TP cube alone
  when only one exists)

The same package holds the deterministic tools behind them:

- `wiener_deconvolve_cube`: Wiener-style deconvolution per channel
- `feather_merge_cube` and `match_tp_to_int_flux_scale`: TP/INT merging
- `regrid_cube_to_match` and `convolve_cube_with_beam`: per-channel cube
  operations
- `clean_beam_cube`: a Gaussian clean beam per channel, fitted to the central
  lobe of each PSF

These in-memory products stay available for machine-learning work. The
reference imaging of a simulation is WSClean on its MeasurementSet (below).

## Wiener preview

`wiener_preview()` is the quick look behind `almasim preview wiener` and the
web console's *Products* page. For each spectral channel it returns:

- `model`: the Wiener model (`wiener_deconvolve_cube`, regularised by
  `epsilon`, default `1e-3`)
- `residual`: the dirty cube minus the model convolved with the
  peak-normalised dirty beam
- `restored`: the model convolved with the clean beam, plus the residual
- `clean_beam`: the Gaussian clean beam of each channel (`clean_beam_cube`)

`almasim preview wiener <dirty> <beam> <output_dir>` writes
`wiener-model.fits`, `wiener-restored.fits` and `wiener-residual.fits`.

The iterative Högbom `clean_deconvolve_cube` was retired (MADIV-28 T5.12).

## Reference imaging of simulations

WSClean on the simulated MeasurementSet is the reference imaging for
simulations. The MS writer records the `OBSERVE_TARGET#ON_SOURCE` intent, each
antenna's dish diameter, and two hands built from Stokes I (XX = I + d and
YY = I − d, with d independent noise at each visibility's own level, so
(XX + YY) / 2 is still I), so the archive commands below (`compute-parameters`, `image-from-ms`, `clean-quality`) run on a
simulated MS as they do on archive data.

## Imaging archive MeasurementSets with WSClean

`almasim image compute-parameters` and `almasim image image-from-ms` turn a
folder of calibrated MeasurementSets into one WSClean image per science
spectral window and target field. They follow the same contract as the unpack
and calibrate stages: every task leaves a marker, a failed task never stops
the batch, and a rerun resumes from the markers.

```bash
# 1. one CSV row per (MS, science spectral window, target field)
almasim image compute-parameters /data/ms-calibrated imaging_parameters.csv

# 2. image them on Slurm, 10 cores per WSClean, 8 workers
almasim image image-from-ms imaging_parameters.csv /data/images \
  --wsclean-bin /opt/spack/.../bin/wsclean \
  --num-cores 10 --max-cores-per-node 95 \
  --slurm-queue normal --slurm-workers 8 --slurm-walltime 1-00:00:00 \
  --task-timeout 7200
```

**compute-parameters** reads each MS (`--pattern`, default `*.cal`, matched in
the members' `calibrated/` folders when the folder is a delivery tree's root) with
python-casacore and writes the WSClean geometry of each spectral window:
field of view (the primary beam's half-power width for the smallest dish),
synthesized beam and reference frequency, with the window's name
(`spectral_window_name`). It writes one row per spectral window and field, and only for (window, field) pairs that hold visibilities:
an ALMA MS lists every correlator window, WVR ones included, but after
`split` only the science windows have rows. By default the fields are those
observed with the `OBSERVE_TARGET` intent, recorded in `target_field_ids`, so
imaging leaves the calibrator scans out. `--intent` selects other fields (for
example `CALIBRATE_PHASE#ON_SOURCE`); `--all-spws` lists every field,
calibrators included. An MS whose `<ms>.done` calibration marker is missing is
skipped when the folder holds such markers, and an MS that cannot be read is
listed in `<csv>.failed.tsv` while the scan continues (`--fail-fast` aborts
instead).

**image-from-ms** runs one WSClean per CSV row, each in its own subprocess on
a worker, writing to `<output>/<ms stem>/SPW-<n>_FIELD-<f>/wsclean-*.fits`
(`SPW-<n>/` for a row without a single field). When `<output>` is the root of
an ALMA delivery tree (it holds `ous_index.csv`), the task goes into the
member folder of the MS's execution block instead,
`<member>/images/<ms stem>/SPW-<n>_FIELD-<f>/`, and an MS the index does not
list keeps `<output>/<ms stem>/...`
([The delivery tree](downloads.md#the-delivery-tree)). Next to that directory it
leaves:

| file | meaning |
|---|---|
| `<task>.done` | finished; JSON with the images written, the WSClean command(s) and the run time |
| `<task>.failed` | failed; JSON with the error (exit code, the `what():` line of a crash, a timeout, a missing binary, or "exited 0 but wrote no image") and the log path |
| `<task>.log` | the full WSClean output, written live on the shared filesystem: `tail -f` it from the submit node |

A rerun redoes every task (`--overwrite-outputs`, the default). With
`--skip-done`, tasks with a `.done` marker are skipped and everything else is
retried, which resumes an interrupted run, and `--skip-existing` also
trusts a `wsclean-image.fits` produced before markers existed (and writes its
marker). Failed tasks are listed at the end and the command exits 1 once every
task has been attempted; `--fail-fast` stops at the first failure instead. If
a worker dies with its task (walltime, node failure), the driver writes the
`.failed` marker itself, so the accounting stays complete: the work remaining
is always *CSV rows − `.done` markers*.

By default each task first extracts its spectral window and target fields
into a per-task MS on the worker and images that with `-no-reorder`
(`--single-window-ms`): WSClean 3.7's reordering crashes on multi-window ALMA
sets. `--no-single-window-ms` images the original MS with `-spws`/`-field`
instead. Each WSClean gets `-j <num-cores>`, `-field <target_field_ids>` when
the CSV has that column, and `-mem` set to its share of the node
(`--task-memory-gb` gives an absolute `-abs-mem` limit instead). It runs with
`OPENBLAS_NUM_THREADS=1`, since WSClean refuses to start otherwise when linked
against a threaded OpenBLAS. `--backend sync` runs the tasks one at a time in
the current process, which is handy for a smoke test on one MS.

### Imaging on the grid of the pipeline's images

`compute-parameters` derives each window's grid from the MeasurementSet. The
images the ALMA pipeline delivered carry the grid tclean imaged them on.
`image fetch-headers` saves each delivered image's header and writes its
imaging parameters beside it, inside the image's own member, so that
`image-from-ms` makes images on the delivered ones' grid:

```bash
# 1. a header and a parameter CSV per delivered image, in its member's folder
almasim image fetch-headers /data/deliveries

# 2. image from them: the tree stands for the parameter CSV
almasim image image-from-ms /data/deliveries /data/deliveries --by-member --cube
```

The delivery tree is the only folder to name. Each image of a member's
`product/` folder gets two files in the member's `images/headers/` folder,
next to the images ALMASim makes:

```text
<member>/product/<image>.pbcor.fits            the pipeline's image
<member>/images/headers/<image>.pbcor.fits.hdr its FITS header
<member>/images/headers/<image>.pbcor.fits.csv its imaging parameters
```

**fetch-headers** finds, for each row `compute-parameters` would write, the
member's `mfs` and `cube` science images of that window (`--kinds`). With
`--kinds mfs,cube,cont` it also saves the header of the image of the field's
windows together (`...spw25_27_29_31.cont...`), whose beam an aggregate
continuum is restored with (`--header-beam`, the default). It
needs the delivery tree's `ous_index.csv`, which names each MS's member
(`--ous-index`, or the one at `--delivery-root`, or the nearest above the
MSs). The images are those in the member's `product/` folder (`--source
local`), those of the archive (`archive`), or the archive's where the folder
has none (`auto`, the default). Only the header is read, a 2880-byte block at
a time up to its `END` card, a few tens of kilobytes of a cube of gigabytes.
The archive is asked once per member for its list of images.

- An image is of a window when its name lists it (`spw25`, or
  `spw13_15_17_19` for an image of several windows), and of a field when its
  `OBJECT` card is the field's name: the file name truncates long names. An
  archive image is matched on its name alone, since its header is not on disk.
- Of several images of one window and field, the self-calibrated one
  (`.selfcal.`) is taken before the `.regcal.` one, and a Stokes I image
  before any other. Its data are self-calibrated: `--alma-selfcal` (on by
  default with `--by-member`) images from data self-calibrated as they were
  ([the pipeline's self-calibration](#the-alma-pipelines-self-calibration)).
- An image's CSV has one row per (MS, window, field) the image is of: an
  image of a member with three execution blocks has three rows, and an image
  of several windows a row for each. A row holds the columns of
  `compute-parameters`, those of the header, the MS-derived size and cell
  (`ms_n_pixels`, `ms_cell_arcsec`), and in `image_check` where the image's
  size or cell is more than 2 % from them.
- A header already saved is not read again (`--overwrite` reads it again),
  and a rerun skips an MS its member's CSVs already list (`--no-resume`
  writes its rows again).
- A row with no image, or an MS that cannot be read, is listed in
  `headers.failed.tsv` at the tree's root and the command exits 1 after the
  scan (`--fail-fast` stops at the first unreadable MS instead).
- `--output <folder>` writes to `<folder>/<member>/` instead of the members'
  own folders. MSs outside a delivery tree need it.

**image-from-ms** given a folder in place of a parameter CSV reads the
per-image CSVs under it: every member's in a delivery tree, one member's in
a member folder, or those of an `--output` folder. Each (MS, window, field)
takes the row of its cube image, or of its `mfs` image when the member has no
cube. `pb-correct` reads a folder the same way.

A cube's row carries the clean limits tclean gave one channel of the cube,
whose threshold is tens of times the continuum's. The row therefore also
holds the limits of the window's `mfs` image, as `continuum_niter`,
`continuum_threshold_jy`, `continuum_weighting` and `continuum_robust`. The
continuum image is cleaned to those and the cube to the `image_*` ones. A
window with no `mfs` image has them empty, and its continuum image takes the
`image_*` limits. A parameter CSV may carry the `continuum_*` columns too.

**parse-headers** reads the saved headers again and rewrites the image
columns and the check of each CSV, without opening an MS or the archive:
`almasim image parse-headers /data/deliveries`. It serves after a change of
`--fov-fraction`, `--beam-sampling` or `--min-beams-across`, which set the
MS-derived size and cell the check compares with.

| column | from | `image-from-ms` uses it as |
|---|---|---|
| `image_n_pixels` | `NAXIS1`, `NAXIS2` (the longer side) | `-size` |
| `image_cell_arcsec` | `CDELT2` | `-scale` |
| `image_weighting`, `image_robust` | tclean's `weighting`, `robust` | `-weight` |
| `image_niter` | tclean's `niter` | `-niter` |
| `image_threshold_jy` | tclean's `threshold` | `-abs-threshold` |
| `continuum_weighting`, `continuum_robust` | the `mfs` image's `weighting`, `robust` | `-weight` of the continuum image |
| `continuum_niter` | the `mfs` image's `niter` | `-niter` of the continuum image |
| `continuum_threshold_jy` | the `mfs` image's `threshold` | `-abs-threshold` of the continuum image |
| `image_bmaj_arcsec`, `image_bmin_arcsec`, `image_bpa_deg` | `BMAJ`, `BMIN`, `BPA` | `-beam-shape`, with `--header-beam` |
| `continuum_bmaj_arcsec`, `continuum_bmin_arcsec`, `continuum_bpa_deg` | the `mfs` image's `BMAJ`, `BMIN`, `BPA` | `-beam-shape` of the continuum image, with `--header-beam` |
| `aggregate_bmaj_arcsec`, `aggregate_bmin_arcsec`, `aggregate_bpa_deg` | the `cont` image's `BMAJ`, `BMIN`, `BPA` | `-beam-shape` of the aggregate continuum image, with `--header-beam` |
| `phasecenter_ra_deg`, `phasecenter_dec_deg` | tclean's `phasecenter` (the `CRVAL` cards without one) | `-shift` |
| `ms_phasecenter_ra_deg`, `ms_phasecenter_dec_deg` | the field's `PHASE_DIR` in the MS | `-shift` |
| `image_n_channels`, `image_specmode` | `NAXIS3`, tclean's `specmode` | `-channels-out` of a cube |
| `image_start_frequency_hz`, `image_channel_width_hz`, `image_spectral_frame` | the first channel's frequency, `CDELT3`, `SPECSYS` | the cube's channels, with `--header-channels` |

**The delivered beam.** WSClean restores an image with the Gaussian it fits
to its own PSF, which differs from tclean's by a few percent, and with it
the flux per beam of everything that was cleaned. `--header-beam` (the
default; `--fitted-beam` keeps WSClean's) restores with the delivered image's
beam instead (WSClean `-beam-shape`), so the two
images compare pixel by pixel: a cube with its cube's beam, a window's
continuum with its `mfs` image's, and an aggregate continuum with that of
the image of the windows together (its windows' first `mfs` beam where the
member has no such image, or its header was not fetched). A row without a
beam keeps the fitted one: a cube tclean restored with a beam per channel
has no `BMAJ` card. A cube's single beam restores every channel.

The other `image_*` columns (`image_stokes`, `image_deconvolver`,
`image_imsize`, `image_cell`, `image_phasecenter`, `image_threshold`,
`image_reference_frequency`) record tclean's values and the header's spectral
axis; `image-from-ms` does not read them.

**image-from-ms** takes each of these columns in place of the MS-derived
value where the row has it, and the MS-derived value where it is empty, so a
window with no delivered image is imaged as from a `compute-parameters` CSV:

- **Weighting.** tclean's `briggs` and `briggsbwtaper` are WSClean's
  `briggs <robust>`; `natural` and `uniform` are WSClean's own. Any other
  scheme images with `briggs 0.5`.
- **Iterations.** An image the pipeline found nothing to clean in has
  `niter = 0`. That is not a limit on the clean, so `-niter` is then 100000,
  as it is without a header.
- **Phase centre.** `-shift` is added when the image's centre is more than
  1 arcsec from the field's in the MS (a mosaic, an offset centre). A moving
  target's image (`phasecenter = 'TRACKFIELD'`) has no fixed centre and is
  not shifted.
- **Cube channels.** tclean regrids a window to LSRK, so its cube often has a
  few channels fewer than the window; WSClean images the window's own
  channels. The cube takes tclean's channel averaging, not its count: the
  window's channels over the whole number of them per tclean channel (a
  1917-channel cube of a 1920-channel window gives 1920; a 958-channel one
  gives 960). This replaces `--spectral-sampling` and
  `--velocity-resolution`; `--max-cube-gb` caps the count as it does
  without a header.
- **The delivered channels.** With `--by-member` and `--cube`,
  `--header-channels` makes each cube on the channels of its delivered cube:
  the execution blocks are regridded onto the delivered cube's first channel,
  channel width and count (its LSRK axis), so channel *n* of the cube is
  channel *n* of the delivered one, whatever `--spectral-sampling` and
  `--max-cube-gb` say. A delivered channel the execution blocks do not all
  cover, edge to edge, is left out, and the task's log names the channels
  made. A row whose cube is not in LSRK, or whose header lacks the frequency
  axis, takes the window's own channels, as `--window-channels` (the
  default) does for every row.

`--fov-fraction`, `--beam-sampling` and `--min-beams-across` do not change a
row that has `image_n_pixels` and `image_cell_arcsec`.

### Cubes, self-calibration and primary-beam correction

`image-from-ms` makes the continuum image of every window by default. Other
products are opt-in; [`almasim image image-from-ms`](cli.md#almasim-image-image-from-ms)
lists every option.

- `--cube` also images each window as a spectral cube
  (`wsclean-cube-<product>.fits`). The channel count follows
  `--spectral-sampling`, `--velocity-resolution` and `--max-cube-gb`;
  `--plan-only` prints the channel plan and the projected volume without
  imaging. `--cube-products` chooses which of image, residual, psf, model and
  dirty to keep. The cube run lets WSClean reorder the visibilities, which
  keeps three files open per channel; `almasim` raises its soft open-file
  limit to the hard one at start-up, and every job it submits inherits that.
- `--uv-contsub` subtracts the continuum from the visibilities (CASA
  `uvcontsub`) before imaging. It is on by default for every cube made from
  single-window copies; `--no-uv-contsub` keeps the continuum in the cube,
  and `--contsub` writes an image-plane continuum-subtracted cube instead. It finds the line channels on the
  source spectrum (every row's visibilities averaged per channel) and fits
  the continuum over the others; the task log's "lines in channels" lists
  them.
- `--pipeline-line-free` (off by default) also leaves out of the line-free
  channels of `--uv-contsub` and `--selfcal` the channels the ALMA pipeline
  calls line: a channel is line-free only when both ALMASim and the
  pipeline's `cont.dat` call it continuum. The `cont.dat` comes from the
  member folder's `calibration/*auxproducts.tgz`, read without extracting
  it; the member is the execution block's in the `ous_index.csv`
  (`--ous-index`, by default the one nearest above the MS). Its spectral
  window ids are the pipeline MS's, which can differ from a split MS's, so
  the window is matched by field name and by the largest overlap in
  frequency, among the blocks with at least half their ranges inside the
  window (a neighbouring window can touch its edge). The MS's channel
  frequencies are converted to `cont.dat`'s frame (LSRK) at the field's
  direction and the median time of its rows. A window marked `ALL` is all
  continuum and one marked `NONE` all line. The task log says how many line
  channels `cont.dat` adds, and its "lines in channels" lists both sets
  together. When `cont.dat` cannot be used (no index, member, tarball,
  `cont.dat` or matching window) the log says why and the task goes on with
  ALMASim's own line channels.
- `--selfcal` self-calibrates each window with CASA `gaincal` before imaging
  (its own solutions; `--alma-selfcal` applies the ALMA pipeline's instead,
  [below](#the-alma-pipelines-self-calibration)),
  one round per solution interval (e.g. `inf,30s,ap:inf`). A round is kept
  only if the continuum S/N rises. With `--noise-products` (the default) it
  also keeps what self-calibration removed and the gain tables.
- `--pbcor` writes the primary beam (`wsclean[-cube]-pb.fits`, CASA's ALMA
  Airy model) and the corrected images (`wsclean[-cube]-image.pbcor.fits`).
  `almasim image pb-correct` adds them to an existing run.

### One continuum image of all the windows

A task images one window, so a field observed in four windows has four
continuum images. `--aggregate-continuum` adds one task per field that
images the continuum of all its windows together, as the ALMA pipeline's
aggregate continuum does:

```bash
almasim image image-from-ms imaging_parameters.csv /data/deliveries --by-member --aggregate-continuum
```

The task writes to `CONT[_<field>]` beside the windows' folders
(`<member>/images/CONT_<field>/` beside `<member>/images/SPW-<n>_<field>/`
with `--by-member`, `<ms>/CONT_FIELD-<id>/` beside `<ms>/SPW-<n>_FIELD-<id>/`
for one execution block), with the products' names of a window task
(`wsclean-image.fits`, `wsclean-residual.fits`, `wsclean-model.fits`,
`wsclean-psf.fits`, and with `--pbcor` `wsclean-pb.fits` and
`wsclean-image.pbcor.fits`) and its own `CONT[_<field>].done`, `.failed` and
`.log`. A field with a single window gets none: that window's image is its
continuum.

- **The data.** The task builds each window's single-window MS as the
  window's own task does (with `--by-member`, from all the execution
  blocks), and removes them when it ends. With `--selfcal` each window is
  self-calibrated on its own continuum first. Each window's continuum is
  then fitted to its visibilities over its line-free channels (a polynomial
  of order `--contsub-order` per visibility, the fit `--uv-contsub` makes
  for a window's cube; with `--pipeline-line-free` the pipeline's `cont.dat`
  line channels are left out too), and the image is made from the fitted
  continua, so the windows' lines stay out of it. A window with too few
  line-free channels for the fit is left out, and the log names it.
  `--aggregate-all-channels` images every channel of every window instead,
  lines included, as a window's own continuum image without `--uv-contsub`
  is.
- **The image.** WSClean takes all the MSs in one run. It makes one output
  channel per window (`-channels-out`, split in the middle of the gaps
  between the windows with `-channel-division-frequencies`; windows that
  overlap share a channel), finds the clean components on their sum
  (`-join-channels`) and fits each with a two-term polynomial over
  frequency (`-fit-spectral-pol 2`), so a source's spectral slope across
  the windows does not stay in the residual. The image WSClean makes of all
  the channels together is kept; the per-channel ones are removed.
- **The grid.** The finest cell of the windows', and as many pixels as the
  widest of their images needs on it.
- **The beam.** The Gaussian WSClean fits to the joint PSF, or with
  `--header-beam` the beam of the delivered image of the windows together
  (`image fetch-headers --kinds mfs,cube,cont`).
- **The header.** `SPW` is left out (the image is of several windows); the
  `HISTORY` names the windows imaged and any left out.

`--aggregate-continuum` needs the continuum (not `--no-continuum`) and
`--single-window-ms`. `clean-quality` assesses a `CONT` folder like any
task's: its `quality.json` and its row of `clean-quality.csv` hold the
continuum statistics. `pb-correct` and the noise commands work on the
windows' tasks and do not read the `CONT` folders; `--pbcor` corrects the
aggregate image in its own task.

### Imaging a member from all its execution blocks

`--by-member` images each member OUS as ALMA does, from every execution block
ALMA calibrated for it, in one task per window and field:

```bash
almasim image image-from-ms imaging_parameters.csv /data/images --by-member --cube
```

The CSV must hold the rows of all the member's execution blocks.
The members come from the delivery tree's `ous_index.csv` (`--ous-index`, by
default the one nearest above the MSs; see
[The delivery tree](downloads.md#the-delivery-tree)). Only a complete member is
imaged: every execution block ALMA calibrated for it (`alma_calibrated`) has a
calibrated MS and rows in the CSV. Every other member is listed with the
execution blocks it lacks. The windows of different execution blocks are
matched by name (`spectral_window_name`): each EB numbers its windows its own
way. Fields are matched by name too. A mosaic, whose pointings share one name,
is listed and left out.

A member task writes to `<output>/member.<uid>/SPW-<n>_<field>/`, where
`<n>` is the window's id in the earliest execution block. With the delivery
tree's root as `<output>` it writes into the member's own folder,
`<member>/images/SPW-<n>_<field>/`:

```bash
almasim image image-from-ms imaging_parameters.csv /data/deliveries --by-member --cube
```

`pb-correct`, `clean-quality` and the noise commands given the same root find
the tasks there. Its single-window
MS combines all the execution blocks (`build_member_window_ms`). The window
and field of each block are extracted as for any task. CASA `mstransform` then
regrids each copy onto one LSRK channel grid, and CASA `concat` joins them
into one MS of one window and one field. The execution blocks' channels are
topocentric, so one window starts a few MHz apart from one day to the next:
channel *n* of one block is not channel *n* of another. `mstransform`
regrids only onto channels a block covers over its whole time range, so the
grid is where all the blocks' windows overlap in LSRK, a channel in from each
end, with the widest block's channel width (with `--header-channels`, the
delivered cube's channels instead; see
[the delivered channels](#imaging-on-the-grid-of-the-pipelines-images)). It has a few channels fewer than
one block's window, and every channel holds every block. A cube keeps the
planned number of native channels per cube channel, so its `-channels-out`
follows the grid. The task's log gives the grid, the cube's channel count and
each block's own LSRK start. The cell and image size follow the longest
baseline of the member's execution blocks. Self-calibration, continuum
subtraction, the cube and the primary-beam correction then run on the
combined MS as on any other. The regridding needs CASA and its runtime data
(`--casa-data-root`).

### The ALMA pipeline's self-calibration

The ALMA imaging pipeline self-calibrates a bright target (`hif_selfcal`) and
delivers the images made from the self-calibrated data (`.selfcal.` in their
names). The delivery holds the execution blocks' calibration tables
(`calibration/*.caltables.tgz`) but not the self-calibration's gain tables, and
ALMA's restore script rebuilds the calibrated data without them. Its record is
the member's `log/*casa_commands.log`: every CASA call the pipeline made, with
the parameters it computed (thresholds, solution intervals, reference
antennas), and which rounds it kept, since a rejected round is followed by
the restore of its flags and no `applycal`.

`image alma-selfcal` replays those calls on the member's calibrated MSs and
keeps the tables the pipeline applied to its targets:

```bash
# 1. replay each self-calibrated member of the tree
almasim image alma-selfcal /data/deliveries --casa-data-root /data/casa-data

# 2. image the members from their self-calibrated execution blocks
#    (--alma-selfcal, --uv-contsub and --header-beam are the defaults)
almasim image image-from-ms /data/deliveries /data/deliveries --by-member --cube
```

The replay follows the pipeline:

1. `hif_mstransform`: each execution block's targets at full resolution, from
   the calibrated MS's `DATA` (the pipeline's `CORRECTED_DATA`).
2. `hifa_flagtargets`: the commands of each block's target flagging template,
   which the member's weblog keeps (`<eb>.flagtargetstemplate.txt`).
3. `hif_selfcal`: the line channels flagged, a channel-averaged copy of each
   block's targets in a folder of its own, and on it every round's `tclean`,
   `gaincal` and `applycal` with the pipeline's arguments (`tclean` runs
   serially; an argument this CASA does not take is left out, and the log
   names it; a copy of an image product the serial `tclean` does not write,
   such as `.gridwt`, is skipped). The pipeline's final application of the tables to its
   full-resolution data, through a calibration library the delivery does not
   hold, ends the replay: the last `applycal` of each block and field before
   it says which tables were applied, and how.

Each member's replay is in its `images/alma-selfcal/` folder (`--output
<folder>` writes `<folder>/member.<uid>/alma-selfcal/` instead): the gain
tables, `apply.json` (each block's tables, `interp`, `spwmap` and `applymode`
per field) and `alma-selfcal.log`. A member already replayed is skipped
(`--overwrite` replays it again), and one whose log has no `hif_selfcal`
stage has nothing to replay. A block the pipeline did not apply tables to is
listed in `apply.json`'s `not_applied`.

With `--by-member`, `image-from-ms` applies a member's tables
(`--alma-selfcal`, the default; `--no-alma-selfcal` leaves them out)
to each block's window, once the window is extracted and before it is
regridded: `applycal` with each table's `spwmap` taken to the extracted window
(a table solved over the windows together maps every window to the one it was
solved in). The task's log says which blocks were self-calibrated. A member
without a replay is imaged from its calibrated data, and `image-from-ms`
names it.

### Provenance in the FITS headers

Every FITS a task writes says what it was made from, with the keywords of
ALMA's own pipeline images where ALMA has one:

| keyword | value |
|---|---|
| `PROPCODE` | the project code |
| `SGOAL`, `GROUP`, `MEMBER` | the science goal, group and member OUS UIDs |
| `SPW`, `FIELD` | the spectral window (its id in the MS) and the field imaged |
| `NEB`, `EBUID01` ... | the execution blocks imaged (all of a member's with `--by-member`) |
| `NMS`, `MSNAM01` ... | the MeasurementSets imaged |
| `ALMASVER` | the ALMASim version |
| `HISTORY` | after WSClean's own record of its command line, each step of the task in the order it ran (`ALMASim: ...`): the window's extraction (a member's combination of its execution blocks), the continuum subtraction from the visibilities (with `--pipeline-line-free`, what `cont.dat` added to its line channels), and every WSClean command (self-calibration rounds included), with the task folder's paths written relative to it |

The OUS UIDs come from the delivery tree's `ous_index.csv`
([The delivery tree](downloads.md#the-delivery-tree)), looked up by the
execution block in the MS's name: `--ous-index` names it, and by default each
MS uses the `ous_index.csv` nearest above it, which finds the index of
everything `products download` fetched and calibrated. An MS that no index
lists keeps its execution block, MS, window and field. The images made from
another one (the continuum-subtracted cube, the primary-beam correction, the
self-calibration products and the `clean-quality` maps) copy these cards and
add a `HISTORY` line of their own.

### Cleaning quality

`almasim image clean-quality <output_root>` measures how much source signal
the clean left in the residuals of each task (in a delivery tree's root, the
tasks in the members' `images/` folders). It writes `quality.json` per
task (continuum and cube statistics), moment maps of the line and of the
residual for cubes (`quality/cube-*.fits`), and a summary CSV
(`clean-quality.csv`). `almasim.services.imaging.quality` documents each
statistic. `continuum.diverged` and `cube.diverged` count how many times
WSClean stopped a clean for divergence (from the task's log, over every
pass): a continuum that diverged holds the diverged model in its restored
image and residual.

## Web console

The [web console](console.md) uses the same commands:

- *Imaging* runs `compute-parameters`, `image-from-ms` and `clean-quality`
  on a folder of calibrated MeasurementSets and follows the windows as they
  finish.
- *Products* shows the images and cubes side by side with their cleaning
  quality, and runs the Wiener preview of a simulation's dirty and beam cubes.
