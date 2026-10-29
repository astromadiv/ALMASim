# Simulation

This page describes the main ALMASim simulation workflow.

## Simulation Model

ALMASim simulates a single-pointing ALMA observation: it builds a model sky cube, observes it with the interferometer (and total power, when the observation has it), and reconstructs image-domain products from the result.

The central service is the package [`src/almasim/services/simulation/`](../src/almasim/services/simulation/):

| Module | Holds |
|---|---|
| `params.py` | `SimulationParams` and its coercion |
| `context.py` | `RunContext`: log, status and cancellation callbacks, and the sky's random generator |
| `rng.py` | `run_rng()`: one independent random stream per part of a run, all from `SimulationParams.seed` |
| `geometry.py` | `CubeGeometry`, `resolve_cube_geometry()`, `estimate_simulation_footprint()` |
| `spectral_setup.py` | redshift, rest frequency, continuum and line fluxes |
| `skymodel_stage.py` | the model cube: sky model, background, serendipitous sources |
| `background.py` | `generate_background_cube()` |
| `metadata.py` | the parameter record, run metadata and effective SNR |
| `planning.py` | the observing runs: one per execution block of each interferometric configuration, one per total-power configuration |
| `stages.py` | `generate_clean_cube()`, `simulate_observation()`, `image_products()` |
| `export.py` | `export_results()` |
| `pipeline.py` | `run_simulation()` |

The command line runs the same stages: `almasim simulation` (options in the [command-line reference](cli.md#almasim-simulation)), and the web console's [Simulate page](console.md#simulate) submits it as a job.

## Main Entry Points

- `SimulationParams`
- `generate_clean_cube()`
- `simulate_observation()`
- `image_products()`
- `export_results()`
- `run_simulation()`
- `write_ml_dataset_shard()`
- `estimate_simulation_footprint()`

All of them are exported by `almasim.services.simulation`; all but `estimate_simulation_footprint()` are also re-exported at package level from [`src/almasim/__init__.py`](../src/almasim/__init__.py).

## Staged Workflow

### 1. `generate_clean_cube()`

This stage:

- resolves the cube geometry from the metadata and any overrides: frequencies, channel width, beam, field of view, cell size and cube dimensions
- resolves the source spectrum: redshift, continuum and line fluxes
- builds the model cube from the chosen [sky model](skymodels.md), or loads an external one
- places the target at the phase center, shifted by optional source offsets in arcseconds
- adds an optional [background sky](background.md) and optional serendipitous sources
- resolves the effective SNR
- plans the observing runs and their per-channel noise
- records the parameter summary and run metadata

It returns a `CleanCubeStage`.

### 2. `simulate_observation()`

This stage:

- runs one interferometric simulation per [execution block](#execution-blocks) of each interferometric configuration (12 m, 7 m)
- runs one total-power simulation per TP configuration
- combines the runs into one interferometric and one total-power result, weighted by integration time
- writes each interferometric run's cubes when outputs are persisted
- returns the model cube, dirty cubes, model and dirty visibilities, beam cubes, sampling and UV mask cubes, the visibility table and the per-configuration results

The interferometer and total-power simulations are described in [Interferometry](interferometry.md); their noise in [Noise](noise.md).

### 3. `image_products()`

This stage builds deterministic image-domain products from the measurement-stage outputs:

- the interferometric image cube (`int_image_cube`): a Wiener deconvolution of the dirty cube
- the total-power image cube (`tp_image_cube`): the TP dirty cube
- the TP+INT combined cube (`tp_int_image_cube`): a feather merge of the two, or whichever of them exists

See [Imaging and Combination](imaging.md) for these products and for the reference WSClean imaging of simulated MeasurementSets.

### 4. `export_results()`

This stage:

- writes `antenna.cfg` (one per configuration, in `config_<n>/`, when there are several)
- writes the image-domain, TP and background cubes in the configured save mode
- writes the parameter summary `sim_params_<idx>.txt`
- writes the simulated visibilities as a MeasurementSet and/or an NPZ file, when requested
- writes the ML shard, when `ml_dataset_path` is set

## Full Run

`run_simulation()` runs the whole sequence, checking for cancellation between stages:

1. generate the clean cube
2. simulate the observation
3. reconstruct image products
4. export results

A cancelled run raises `SimulationCancelled`.

## Execution Blocks

A metadata row's `Int.Time` is the MOUS's total time on source. ALMASim observes it in execution blocks: the row's `n_executions` when the metadata has it (see `almasim products executions`), otherwise blocks of at most 3000 s. Each block is a track near transit on its own day, with its own draw of the MOUS's antennas. `SimulationParams.executions` (`--executions` on the command line) simulates only the first blocks; by default all of them are simulated.

## Simulation Parameters

`SimulationParams` includes:

- source and project metadata
- pointing coordinates
- ALMA band and frequency support
- continuum and line sensitivity metadata
- PWV and integration time
- execution blocks (`n_executions`, `executions`)
- source type, and the external model's path for the external source types
- optional manual cube-dimension overrides (`n_pix`, `n_channels`)
- the instrument geometry (`instrument_geometry`: `legacy` or `planner`)
- optional manual SNR override
- the noise model (`noise_model`; see [Noise models](noise_models.md#using-a-noise-model-in-a-simulation))
- background mode, level and seed
- explicit source offsets
- the random seed (`seed`)
- output, MeasurementSet export and compute settings

`SimulationParams.from_metadata_row()` builds the parameters from a metadata row; keyword arguments supply everything else and override the row's optional values.

## Auto-Derived Parameters

If you do not override them, ALMASim derives:

- the channel width from the archive's spectral resolution (ALMA's correlator channels are narrower than the resolution it reports)
- `n_channels` from the spectral window's bandwidth and that channel width
- the cell size: a fifth of the synthesized beam with the `legacy` geometry; an eighth with the `planner` geometry, which also takes the primary beam as the field of view instead of the metadata's
- `n_pix`: a power of two covering 1.5 times the field of view
- the effective SNR from the metadata sensitivities (`Line_sens_10kms_mJybeam` over `Cont_sens_mJybeam`, clipped to 1–100; 5 with only a continuum sensitivity, 1.3 with neither)

A given `n_pix` is rounded to a power of two and sets the cell size as the field of view over `n_pix`; a given `n_channels` sets the bandwidth. The command line always passes them (`--n-pix`, default 128; `--n-channels`, default 32); from Python, leave them `None` to derive them.

## Reproducibility

`SimulationParams.seed` seeds every random draw of a run: the sky model, the interferometric and total-power noise, the background, the execution blocks' antennas, the receiver ripple and the MeasurementSet's per-hand noise. Each part draws from its own stream, so a change in how many numbers one part draws never shifts another's. Without a seed every stream draws fresh entropy.

On the command line, `--seed` is the base seed: run `i` of an invocation uses `seed + i`.

## Preflight Estimate

`estimate_simulation_footprint()` computes, with the same geometry resolver as the run:

- resolved pixel count
- resolved channel count
- cell size
- beam size
- raw cube size in GiB (real and complex)
- a rough raw footprint of the standard outputs

The web console's Simulate page uses it for its pre-run estimate.

## Outputs

Depending on the run configuration, ALMASim can produce:

- model (clean) cube
- dirty cube
- model and dirty visibilities
- beam cube
- total sampling cube
- UV mask cube
- U/V coordinate cubes
- per-configuration interferometric results
- TP products
- interferometric, TP and TP+INT image cubes
- background cube
- simulated MeasurementSet
- ML HDF5 shard

On disk, each product is named `<product>_<idx>` with the save mode's extension, for example `dirty-cube_0.fits`, `int-image-cube_0.fits` and `background-cube_0.fits`. In FITS mode, complex visibility cubes are written as `<product>_real<idx>.fits` and `<product>_imag<idx>.fits`; in HDF5 mode the beam, sampling, UV mask and U/V cubes share `measurement-operator_<idx>.h5`.

### MeasurementSets

`export_results()` writes the run's visibility table according to `ms_save_mode`:

| `ms_save_mode` | Writes |
|---|---|
| `msv2` (the `SimulationParams` default) | a CASA MeasurementSet, `<project_name>_<idx>.ms` (or `ms_export_dir`) |
| `npz` | `<project_name>_<idx>_vis.npz` |
| `both` | both |
| `none` | nothing (unless `ms_export=True`, which writes an MS) |

Writing an MS needs `casatools` or `python-casacore`. The MS records the `OBSERVE_TARGET#ON_SOURCE` intent, each antenna's dish diameter, one observation per execution block, and Stokes I in both hands: XX and YY each carry their own noise, at each visibility's own sigma and with the window's channel correlation (as two real polarisations, each smoothed by the correlator on its own), and (XX + YY) / 2 is the simulated Stokes I. WEIGHT is `1 / mean(sigma^2)` over a row's channels, half of Stokes I's per hand, with each baseline's own noise scale when the noise model has `antenna_ripple`. With a noise model, its identity is recorded in the `HISTORY` table. `almasim simulation` does not export an MS.

## Persistence Modes

ALMASim supports:

- standard persisted file output (`save_mode` `fits`, `npz`, `h5` or `both`)
- pure Python mode with `persist=False`
- `save_mode="memory"` to keep every product in memory
- ML shard export even when standard outputs are skipped

In pure Python and memory modes, the products are returned in the result dictionary only; the MeasurementSet and the ML shard are still written when requested. The command line defaults to `--save-mode memory`, which writes only the ML shard (`--ml-shard-path`); `--save-mode fits` (or `npz`, `h5`) writes the standard outputs as well.
