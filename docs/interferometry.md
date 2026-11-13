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
| `observing.py` | the hour-angle track: number of samples and their length, which together observe the time on source in full (`exact_samples`), each execution block's track (`Track`, `execution_track`: its scans, their hour angles and span) and when the source is at each hour angle |
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

The antennas sit on ALMA's pads at the positions in [`antenna_config/antenna_coordinates.csv`](../src/almasim/antenna_config/antenna_coordinates.csv) (metres, local tangent plane), the positions of CASA's `alma.all.cfg`, which match the MSs' `ANTENNA` tables to within half a metre. An archive row's `antenna_arrays` names them as `PAD:ANTENNA`.

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

[`services/simulation/planning.py`](../src/almasim/services/simulation/planning.py) turns the plan into runs: one `Interferometer` run per execution block of each interferometric configuration, and one total-power run per TP configuration. Each execution block observes a track of its own on its own day, with its own draw of the configuration's antennas; see [Execution blocks](simulation.md#execution-blocks).

## The Interferometer

For each channel, the `Interferometer`:

- multiplies the model by the primary beam: CASA's Airy model of an ALMA antenna with its central blockage ([`instrument/primary_beam.py`](../src/almasim/services/instrument/primary_beam.py)), the same model the total-power simulation and `--pbcor` use; in an array of two dish sizes each baseline sees the sky through its own dishes' beam ([below](#arrays-of-two-dish-sizes))
- samples the uv plane along the execution block's track: its scans at their hour angles on the block's day (one scan centred on transit for an `Interferometer` built without a track); the exported MS's times are when the source is at each sample's hour angle
- adds thermal noise at each visibility, and the noise model's corruptions when one is set
- grids the visibilities with robust weighting (`robust`, `--robust`)
- forms the dirty beam and the dirty image

The cube, the uv sampling and the exported MeasurementSet share one channel grid, centered on the window's central frequency. The compute backend images the channels, one after another or in parallel (see [Compute Backends](compute.md)).

## Arrays of two dish sizes

ALMA's 7 m array can correlate the 12 m total-power dishes, on their pads
T701-T704, with its 7 m dishes. An antenna on a total-power pad has a 12 m
dish in any interferometric configuration
([`instrument/array.py`](../src/almasim/services/instrument/array.py)
`pad_dish_diameter_m`), and the configuration's `antenna_diameter_m` is the
others'. A metadata row names those antennas PM, and the plan moves PM
antennas to a total-power configuration of their own; a configuration given
directly (`observation_configs`) keeps them in the array by listing their
pads without the name, as `noise validate` does for the MSs that correlate
them. In such an array:

- each baseline sees the sky through the product of its two dishes' voltage
  patterns (`baseline_beam_response`), a beam between the two dishes' own;
  the dirty image shows the model through the configuration's dish
- each baseline's noise is the geometric mean of its two antennas' SEFDs:
  the configuration's noise times the square root of the product of each
  antenna's effective-area ratio (the configuration's dish's over its own,
  from the noise model's aperture efficiencies), and its WEIGHT follows
- the MS lists each antenna's own dish diameter

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
