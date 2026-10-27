# Noise

This page describes ALMASim’s current noise model.

## Location

Noise handling lives primarily in [`src/almasim/services/interferometry/noise.py`](../src/almasim/services/interferometry/noise.py);
the noise models in [`src/almasim/services/noise_model/`](../src/almasim/services/noise_model/).

## Noise Models

What the visibilities carry besides white thermal noise is a noise model:
a named, versioned JSON file made of components, each with its own switch and
parameters. Released models ship with ALMASim in
[`services/noise_model/models/`](../src/almasim/services/noise_model/models/);
a released file is never edited, a change is its next version.

```
almasim simulation run ... --noise-model alma              # the newest alma
almasim simulation run ... --noise-model alma@1            # a fixed version
almasim simulation run ... --noise-model candidate.json    # any model file
almasim simulation run ... --noise-model alma --noise-without fast_phase,baseline_errors
almasim simulation run ... --noise-model alma --noise-only radiometer,channel_correlation
almasim noise show                  # the released models
almasim noise show alma             # one model, its components on or off
almasim noise components            # every component and its parameters
almasim noise schema -o model.schema.json
```

`--corruptions` is the option's earlier name and still works, as do the flat
JSON objects of `CorruptionModel` fields it took. A model file validates
against [`noise-model.schema.json`](noise-model.schema.json) (unknown
components or parameters, and values out of range, are refused), so a
candidate model can be used for a run before it is released.

| Component | What it adds | Parameters |
| --- | --- | --- |
| `radiometer` | each visibility's noise from the radiometer equation (below) | `receiver_temperature_k`, `aperture`, `quantisation_efficiency`, `correlator_efficiency` |
| `atmospheric_tsys` | the atmosphere's lines in Tsys across the window | |
| `receiver_ripple` | the receiver's smooth ripple of Tsys | `rms` |
| `channel_correlation` | correlated noise of neighbouring channels | `rho1`, `rho2`, `average` |
| `antenna_ripple` | each antenna's own ripple of its noise | `rms`, `scale_mhz` |
| `antenna_gains` | antenna gain errors per scan | `amp_rms`, `phase_rms_deg` |
| `slow_phase` | the per-scan atmospheric phase | `path_um_1km`, `slope` |
| `fast_phase` | the in-scan atmospheric phase | `path_um_1km`, `slope`, `coherence_s` |
| `baseline_errors` | per-baseline errors that do not close | `amp_rms`, `phase_rms_deg` |

ALMA's receiver temperatures, the dishes' Ruze aperture efficiencies and the
quantisation and correlator efficiencies are instrument data,
[`interferometry/data/alma_radiometer.json`](../src/almasim/services/interferometry/data/alma_radiometer.json),
which every run uses for Tsys. A model's `radiometer` component may set its
own (`alma@1` states the instrument's, so the model reproduces on its own),
and a candidate model can try other values without a code change.

A model's `settings` hold what components share: the scan length and the
screens' outer scale. A selection keeps the deselected components in the
model, switched off, so it is part of the model's identity: every product
of a run with a model records its label (`alma@1 without fast_phase`) and
the SHA-256 of its JSON, in the metadata (`noise_model`), the
`sim_params_<idx>.txt` file and the exported MS's HISTORY.

### Adding a component

1. Write its class in
   [`services/noise_model/components.py`](../src/almasim/services/noise_model/components.py):
   a `NoiseComponent` with its parameters (bounds and a description each)
   and `engine_fields`, the `CorruptionModel` fields it sets.
2. Add it as a field of `Components` there.
3. Make the simulator apply those fields
   ([`interferometry/corruptions.py`](../src/almasim/services/interferometry/corruptions.py)
   and where they are used).
4. Run `scripts/generate_noise_model_schema.py` and `scripts/generate_cli_reference.py`.

The schema, the validation of model files, `--noise-only` and
`--noise-without`, and `almasim noise show` and `components` follow from the
field; a test checks that every component sets only simulator fields, and
none another component sets.

## Measuring Real Data

`almasim noise extract` measures what a real window's calibrated
visibilities carry, from differences of channels (three apart they remove
the sky; one and two apart they show the correlation of neighbours), so no
sky model is needed:

```
almasim noise extract uid___A002_X1.ms.split.cal --spw 31 --field 2 --out noise/
almasim noise extract --windows windows.csv --out noise/ --backend slurm
```

Each window writes `windows/<name>/window.json`: its exposure (antennas,
integrations, time on source, channels and noise bandwidth, flags, the
target's elevation), the noise against WEIGHT (`scale`, 1 when WEIGHT is
right) and its channel correlation (`rho1`, `rho2`) and profile, ALMA's Tsys
across the window from SYSCAL, one Stokes I visibility's noise at unit time
and bandwidth and the efficiency it implies with the Tsys, and each
antenna's own noise shape (its rms and frequency scale; the arrays in
`antenna.npz`). Rows are sampled evenly over the observation, up to
`--max-visibilities`, and read in chunks. A windows table has the columns
`ms` and `spw`, and optionally `field`, `name`, `syscal_ms`,
`line_channels` ("a~b;c", left out as line emission) and `task_log` (an
ALMASim imaging task's log, whose "lines in channels" gives them). Measured
windows are skipped on a rerun; failures go to `failures.json`. With
`--backend slurm` each window is its own task on a Slurm worker. It needs
python-casacore (the `ms-casacore` extra).

### Fitting calibration residuals

`almasim noise fit` fits what calibration leaves on a bright window, the
values of the `slow_phase`, `fast_phase`, `antenna_gains` and
`baseline_errors` components, after `noise extract` has measured it:

```
almasim noise fit uid___A002_X1.ms.split.cal --spw 31 --field 2 \
    --task-dir cubes/uid___A002_X1.ms.split/SPW-31_FIELD-2 --out noise/
almasim noise fit --windows windows.csv --out noise/ --backend slurm
```

It predicts the window's clean model (`--task-dir`, an ALMASim imaging task:
the continuum model where the image is above 5 sigma, the cube model in the
line channels and line mask) and compares the data with it per (baseline,
scan) where the model is bright (S/N above 10). The per-scan phase scatter
beyond the noise's gives the slow phase screen, the loss of amplitude the
in-scan screen, the per-scan antenna amplitudes the gains, and what does
not close the baselines' own errors. Each is fitted against a reference
holding what was found before it, built by the simulator's own engine, so a
fitted value is what the simulator needs to reproduce the data: the screens'
strength and slope come from simulating them, not from a formula. A window
too faint for that is left unfitted, with a note in `fit.json`.

### From measurements to a candidate model

`almasim noise aggregate` reduces the windows' measurements into a candidate
model: a released model (`--from`, default `alma`) with every value the
windows measure replaced by their median, and how many windows each comes
from and their 16-84% spread recorded in `provenance.values`:

```
almasim noise extract --windows windows.csv --out noise/        # pwv column for Trx
almasim noise fit --windows windows.csv --out noise/            # task_dir column
almasim noise aggregate --out noise/                            # noise/candidate.json
almasim simulation run ... --noise-model noise/candidate.json   # try it
```

| Value | From |
| --- | --- |
| `channel_correlation` rho1, rho2 | windows with Hanning-correlated native channels (rho1 above 0.5) |
| `receiver_ripple` rms | the smooth part (Legendre modes 1-4) of the noise profile over SYSCAL's Tsys², halved |
| `antenna_ripple` rms, scale | each antenna's noise shape, windows wider than 1 GHz |
| `radiometer` receiver temperatures | per band, the Trx that gives SYSCAL's median Tsys under the simulator's Tsys model (12 m windows, their PWV and elevation) |
| `radiometer` aperture | per dish size, Ruze's law fitted to the efficiency the noise and Tsys imply |
| `slow_phase`, `fast_phase`, `antenna_gains`, `baseline_errors` | `noise fit` |

A value fewer than `--min-windows` (3) measure keeps the base's, with a
note in `provenance.notes`.

## Current Model

ALMASim uses a PWV-aware channel noise model that depends on:

- precipitable water vapor
- channel frequencies
- channel width
- integration time
- elevation
- antenna diameter
- number of antennas

The noise profile is computed per channel and then calibrated against a reference noise level used by the simulation pipeline.

## Across a Window: Tsys

By default the sky temperature is a smooth function of frequency, so the noise
barely changes across a spectral window. Calibrated ALMA data do vary: their
noise follows ALMA's own Tsys spectrum, measured on the 100 MADIV-16 windows
(the variance times WEIGHT across each window against the square of its SYSCAL
Tsys). Its shape comes from three places:

- atmospheric lines in the signal band (the 118.75 GHz oxygen line seen from
  the CO(1-0) windows at 114 GHz, water at 183, 325 and 448 GHz, oxygen at
  487 GHz);
- the same in the image band, which a sideband-separating receiver rejects by
  about 10 dB and a double-sideband one (Bands 9 and 10) not at all;
- the receiver, whose temperature ripples by about 2% of Tsys across its IF.

`--noise-model alma` (its `atmospheric_tsys` and `receiver_ripple`
components) turns this on. The sky comes from
[`atmosphere.py`](../src/almasim/services/interferometry/atmosphere.py), which
reads a table of CASA's ATM model for the ALMA site
(`scripts/generate_atmosphere_table.py` writes it; CASA is not needed to use
it). Tsys is referred above the atmosphere, (1 + g) e^(A tau) (Trx + Tsky +
g Tsky_image), with the image band one IF centre on the other side of the LO
and the sideband whose image gives the lower Tsys. The ripple is drawn from
the run's seed.

## Per Visibility: the Radiometer Equation

By default the noise is calibrated in the image: each channel's noise is set so
that the source comes out at the run's target S/N. With the `radiometer`
component (on in `--noise-model alma`) each visibility instead carries the noise ALMA's
radiometer equation gives it, from the same Tsys:
SEFD / (eta sqrt(2 n_pol dnu t)), with the SEFD 2 k Tsys / A_eff of one
antenna, eta = 0.96 x 0.88 (quantisation and correlator), the channel's
noise bandwidth dnu (see Channels below) and the simulator's integration t. The source keeps its modelled flux, so
its S/N follows from the observation. A 12 m visibility at 230 GHz over
15.6 MHz and 6 s then has ~0.1-0.2 Jy of noise, and an exported MS images, in
WSClean or CASA, to the S/N a real observation of that source would reach.

The exported MS's WEIGHT is 1 / mean(sigma^2) over a row's channels, sigma
scaled by the gain amplitude the data were multiplied by, as the ALMA
pipeline's statwt sets it.

## Channels

The archive's frequency support (`Freq.sup.`) gives each window's spectral
resolution, not its channel width. ALMA's correlator Hanning-smooths its
native channels and may average N of them into each one, and every ALMA
channel width is 15.625 MHz / 2^k. The resolution therefore tells N
([`instrument/channels.py`](../src/almasim/services/instrument/channels.py)):

| Resolution against ALMA's grid | N | Channel width | rho1 | Noise bandwidth |
| --- | --- | --- | --- | --- |
| on it | 1 | resolution / 2 | 0.667 | 2.67 x width |
| 1.156 x above it | 2 | the grid value | 0.30 | 1.60 x width |
| just below it | 4, 8, 16, ... | the grid value | 0.115, 0.052, 0.026 | 1.23, 1.10, 1.05 x width |

For N >= 4, N is the width over the window's native channel (its bandwidth
over 3840). The 10,052 windows of cycle11.csv each fall in one of the three
classes; on the 100 MADIV-16 windows the rule gives each one's RESOLUTION,
CHAN_WIDTH, EFFECTIVE_BW and the measured noise correlation of its
neighbouring channels. A resolution off the grid is taken as the channel
width.

The cube's channels are the window's (the run's metadata records
`channel_averaging`). The correlated channel noise (the `channel_correlation` component's `rho1`
and `rho2`, the native channels' correlation, 2/3 and 1/6 in
`--noise-model alma`) is diluted over the window's N; its `average` sets N
instead (0, the default, takes the window's).

## Per Antenna and Per Baseline

Two more terms, also in `--noise-model alma`, come from the same 100 windows:

- Each antenna's noise has its own smooth ripple across the window, beyond the
  array's: about 2% of its noise variance, holding over about 120 MHz (the
  median of the 73 windows wider than 1 GHz; narrow windows show finer, ~1 MHz
  structure). A baseline's noise variance is the product of its two antennas'.
  WEIGHT does not follow it, so it shows as antennas whose noise differs from
  their weights and as heavier tails in the visibility noise
  (the `antenna_ripple` component).
- On bright sources, each baseline keeps an amplitude and phase error per scan
  that does not factor into its antennas' gains, so self-calibration cannot
  remove it, and it limits the continuum's dynamic range: about 6% and 2.5 deg
  (the median of the 13 bright windows where it is measurable; the
  `baseline_errors` component). It acts on each channel's
  continuum share of the source only: what does not close in calibrated
  data is mostly continuum structure the model lacks, which leaves line
  emission alone. An external sky model, whose continuum is not known
  apart, takes it on all its emission.

## Inputs

The key inputs are:

- `pwv_mm`
- `ground_temperature_k`
- per-channel observing frequencies
- observing time
- antenna count
- array type through antenna diameter and array configuration

## SNR Handling

ALMASim supports:

- manual SNR override
- automatic SNR derivation from metadata sensitivity fields

The current auto mode uses:

- `Cont_sens_mJybeam`
- `Line_sens_10kms_mJybeam`

This keeps the default behavior tied to available ALMA metadata rather than arbitrary hardcoded values.

## What Noise Is Not

The background sky features described elsewhere are not part of the noise model.

ALMASim treats them separately:

- noise is an observing/corruption term
- background sky is an additive astrophysical sky component

## Future Directions

Likely future improvements include:

- richer correlator-dependent noise settings
- better calibration of line and continuum sensitivity relationships across bands
