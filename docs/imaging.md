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
| `batch_runner.py`, `task.py`, `markers.py` | running the tasks on `sync` or `slurm`, their `.done`/`.failed`/`.log` files |
| `ms/` | the single-window MS each task images |
| `uv_contsub.py`, `selfcal/`, `primary_beam.py` | visibility-plane continuum subtraction, self-calibration, primary-beam correction |
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

**compute-parameters** reads each MS (`--pattern`, default `*.cal`) with
python-casacore and writes the WSClean geometry of each spectral window:
field of view (the primary beam's half-power width for the smallest dish),
synthesized beam and reference frequency. It writes one row per spectral
window and field, and only for (window, field) pairs that hold visibilities:
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
(`SPW-<n>/` for a row without a single field). Next to that directory it
leaves:

| file | meaning |
|---|---|
| `<task>.done` | finished; JSON with the images written, the WSClean command(s) and the run time |
| `<task>.failed` | failed; JSON with the error (exit code, the `what():` line of a crash, a timeout, a missing binary, or "exited 0 but wrote no image") and the log path |
| `<task>.log` | the full WSClean output, written live on the shared filesystem: `tail -f` it from the submit node |

Tasks with a `.done` marker are skipped on a rerun and everything else is
retried. `--overwrite-outputs` redoes all of them, and `--skip-existing` also
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
- `--contsub` writes an image-plane continuum-subtracted cube;
  `--uv-contsub` subtracts the continuum from the visibilities (CASA
  `uvcontsub`) before imaging instead.
- `--selfcal` self-calibrates each window with CASA `gaincal` before imaging,
  one round per solution interval (e.g. `inf,30s,ap:inf`). A round is kept
  only if the continuum S/N rises. With `--noise-products` (the default) it
  also keeps what self-calibration removed and the gain tables.
- `--pbcor` writes the primary beam (`wsclean[-cube]-pb.fits`, CASA's ALMA
  Airy model) and the corrected images (`wsclean[-cube]-image.pbcor.fits`).
  `almasim image pb-correct` adds them to an existing run.

### Provenance in the FITS headers

Every FITS a task writes says what it was made from, with the keywords of
ALMA's own pipeline images where ALMA has one:

| keyword | value |
|---|---|
| `PROPCODE` | the project code |
| `SGOAL`, `GROUP`, `MEMBER` | the science goal, group and member OUS UIDs |
| `SPW`, `FIELD` | the spectral window (its id in the MS) and the field imaged |
| `NEB`, `EBUID01` ... | the execution blocks imaged |
| `NMS`, `MSNAM01` ... | the MeasurementSets imaged |
| `ALMASVER` | the ALMASim version |
| `HISTORY` | after WSClean's own record of its command line, each step of the task in the order it ran (`ALMASim: ...`): the window's extraction, the continuum subtraction from the visibilities, and every WSClean command (self-calibration rounds included), with the task folder's paths written relative to it |

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
the clean left in the residuals of each task. It writes `quality.json` per
task (continuum and cube statistics), moment maps of the line and of the
residual for cubes (`quality/cube-*.fits`), and a summary CSV
(`clean-quality.csv`). `almasim.services.imaging.quality` documents each
statistic.

## Web console

The [web console](console.md) uses the same commands:

- *Imaging* runs `compute-parameters`, `image-from-ms` and `clean-quality`
  on a folder of calibrated MeasurementSets and follows the windows as they
  finish.
- *Products* shows the images and cubes side by side with their cleaning
  quality, and runs the Wiener preview of a simulation's dirty and beam cubes.
