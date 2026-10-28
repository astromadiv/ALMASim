# Interferometry

This page describes ALMASim's interferometric simulation layer.

## Location

The main code is in [`src/almasim/services/interferometry`](../src/almasim/services/interferometry).

## Main Modules

| Module | Holds |
|---|---|
| `core.py` | `Interferometer`: one interferometric run, from antenna positions to products |
| `simulator.py` | the channel-by-channel imaging the `Interferometer` scatters over the compute backend |
| `imaging.py` | per-channel uv gridding, primary beam, dirty beam and dirty image |
| `observing.py` | the hour-angle track: number of samples and scan time |
| `antenna.py` | antenna configurations, maximum baseline, beam size estimates |
| `baselines.py` | baseline preparation |
| `frequency.py` | the spectral window from the metadata's frequency support |
| `visibility.py` | the row-wise visibility table the MeasurementSet writer takes |
| `predict.py` | model visibilities of image planes with ducc0 (also used by `almasim predict`) |
| `multiconfig.py` | combining interferometric runs |
| `total_power.py` | total-power simulation and combination |
| `combine.py` | steps shared by the two combinations |
| `noise.py` | thermal noise: system temperature, SEFD, per-channel and per-visibility noise |
| `atmosphere.py` | the atmosphere's opacity and Tsys across a window, from CASA's ATM model |
| `corruptions.py` | what calibrated visibilities carry besides thermal noise (the noise model's engine) |
| `utils.py` | small helpers |

[`data/`](../src/almasim/services/interferometry/data) holds the ATM table for the ALMA site and the radiometer parameters. The noise modules are described in [Noise](noise.md) and [Noise models](noise_models.md).

## Current Scope

ALMASim simulates single-pointing observations.

It supports:

- interferometric simulation for ALMA antenna arrays (12 m and 7 m)
- multi-configuration single-pointing observation plans
- MOUSs observed in several execution blocks
- total-power simulation
- combined interferometric results
- UV mask generation
- dirty beam and dirty image products
- a visibility table for MeasurementSet export

Mosaics are not supported.

## Observation Planning

Single-pointing observation plans are built in [`src/almasim/services/observation_plan.py`](../src/almasim/services/observation_plan.py).

Plans can be:

- given directly (`SimulationParams.observation_configs`)
- inferred from metadata rows
- split automatically into `12m`, `7m` and `TP` configurations when the metadata's antenna list mixes antenna types

[`services/simulation/planning.py`](../src/almasim/services/simulation/planning.py) turns the plan into runs: one `Interferometer` run per execution block of each interferometric configuration, and one total-power run per TP configuration. Each execution block observes a track near transit on its own day, with its own draw of the configuration's antennas; see [Execution blocks](simulation.md#execution-blocks).

## The Interferometer

For each channel, the `Interferometer`:

- multiplies the model by the primary beam: CASA's Airy model of an ALMA antenna with its central blockage ([`instrument/primary_beam.py`](../src/almasim/services/instrument/primary_beam.py)), the same model the total-power simulation and `--pbcor` use
- samples the uv plane along the hour-angle track of the source
- adds thermal noise at each visibility, and the noise model's corruptions when one is set
- grids the visibilities with robust weighting (`robust`, `--robust`)
- forms the dirty beam and the dirty image

The cube, the uv sampling and the exported MeasurementSet share one channel grid, centered on the window's central frequency. The compute backend images the channels, one after another or in parallel (see [Compute Backends](compute.md)).

## Interferometric Products

The interferometry layer can produce:

- model cube (the primary-beam-weighted sky)
- dirty image cube
- model and dirty visibility cubes
- beam cube
- total sampling cube
- UV mask cube
- U cube
- V cube
- the visibility table, written as a MeasurementSet by `export_results()` (see [MeasurementSets](simulation.md#measurementsets))

These products serve both astronomy workflows and inverse-problem or ML workflows.

## Multi-Configuration Support

The multi-configuration combination is in [`multiconfig.py`](../src/almasim/services/interferometry/multiconfig.py).

It combines:

- the per-run interferometric results
- the per-run metadata
- the per-run dirty and beam products, weighted by integration time
- the uv sampling, summed into one UV mask
- the visibility tables, concatenated

## Total Power

Total-power simulation is in [`total_power.py`](../src/almasim/services/interferometry/total_power.py).

It provides:

- the TP beam: the primary beam's half-power width per channel
- the model cube smoothed by the primary beam
- TP dirty products, with per-channel thermal noise
- multi-configuration TP combination

## Relation to CASA `simalma`

ALMASim aims at `simalma`-like fidelity for single pointings while keeping its intermediate products accessible from Python for ML work. It shares CASA's ALMA primary-beam model, and its simulated MeasurementSets can be imaged with the same WSClean commands as archive data (see [Reference imaging of simulations](imaging.md#reference-imaging-of-simulations)).
