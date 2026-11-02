# Noise

This page is the overview of what ALMASim's simulated visibilities carry
besides the sky: the thermal noise every run has, and the noise models that
make it look like real, calibrated ALMA data. [Noise models: a complete
guide](noise_models.md) explains each part in depth: what it is physically,
how it is measured on real data and how the simulator applies it, and it
walks through building and releasing a model.

## Location

- the thermal noise and the radiometer equation:
  [`services/interferometry/noise.py`](../src/almasim/services/interferometry/noise.py);
  the sky and Tsys:
  [`atmosphere.py`](../src/almasim/services/interferometry/atmosphere.py);
  what calibration leaves:
  [`corruptions.py`](../src/almasim/services/interferometry/corruptions.py)
- the noise models (components, released files, schema):
  [`services/noise_model/`](../src/almasim/services/noise_model/)
- ALMA's channels from the archive's resolution:
  [`services/instrument/channels.py`](../src/almasim/services/instrument/channels.py)
- measuring real data, building, validating and releasing models:
  [`services/noise/`](../src/almasim/services/noise/) (`almasim noise ...`)

## Without a Noise Model

A run without a noise model (`--noise-model none`, the default) adds white
thermal noise whose shape across the window comes from a PWV-aware channel
noise profile. The profile depends on:

- the precipitable water vapour (the archive's `PWV`)
- the channel frequencies and width
- the integration time
- the target's elevation
- the antenna diameter and the number of antennas
- the ground temperature (`ground_temperature_k`, 270 K by default)

Its level is then set by the run's target S/N: each channel's profile is
scaled so that its median is a tenth of the faintest line's peak per beam
over the S/N. The S/N is a fixed value when one is given
(`SimulationParams.snr`); otherwise it is derived from the archive row's
sensitivities:

- `Line_sens_10kms_mJybeam` over `Cont_sens_mJybeam`, clipped to 1-100,
  when both are known;
- 5 when only the continuum sensitivity is known;
- 1.3 otherwise.

This keeps the default tied to the ALMA metadata rather than to arbitrary
hard-coded values. The `radiometer` component of a noise model replaces this
with the noise a real observation would have (see
[Per Visibility](#per-visibility-the-radiometer-equation) below).

## Noise Models

What the visibilities carry besides white thermal noise is a noise model:
a named, versioned JSON file made of components, each with its own switch and
parameters. Released models ship with ALMASim in
[`services/noise_model/models/`](../src/almasim/services/noise_model/models/)
(today `alma`, version 1); a released file is never edited, a change is its
next version.

```
almasim simulation ... --noise-model alma              # the newest alma
almasim simulation ... --noise-model alma@1            # a fixed version
almasim simulation ... --noise-model candidate.json    # any model file
almasim simulation ... --noise-model alma --noise-without fast_phase,baseline_errors
almasim simulation ... --noise-model alma --noise-only radiometer,channel_correlation
almasim noise show                  # the released models
almasim noise show alma             # one model, its components on or off
almasim noise components            # every component and its parameters
almasim noise schema -o model.schema.json
```

`--corruptions` is an alias of the option, and it also takes flat JSON
objects of `CorruptionModel` fields. A model file
validates against [`noise-model.schema.json`](noise-model.schema.json):
unknown components or parameters, and values out of range, are refused. A
candidate model can therefore be used for a run before it is released.

| Component | What it adds | Parameters |
| --- | --- | --- |
| `radiometer` | each visibility's noise from the radiometer equation (below) | `receiver_temperature_k`, `aperture`, `quantisation_efficiency`, `correlator_efficiency`, `forward_efficiency` |
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
own values (`alma@1` states the instrument's, so the model reproduces on its
own), and a candidate model can try other values without a code change.

A model's `settings` hold what several components share: the scan length and
the turbulent screens' outer scale. A selection (`--noise-only`,
`--noise-without`) keeps the deselected components in the model, switched
off, so it is part of the model's identity: every product of a run with a
model records its label (`alma@1 without fast_phase`) and the SHA-256 of its
JSON, in the metadata (`noise_model`), the `sim_params_<idx>.txt` file and
the exported MS's `HISTORY` table.

To add a component, see
[Adding a component](noise_models.md#adding-a-component) in the guide.

## Across a Window: Tsys

Without `atmospheric_tsys` the sky temperature is a smooth function of
frequency, so the noise barely changes across a spectral window. Calibrated
ALMA data do vary: their noise follows ALMA's own Tsys spectrum, as measured
on the 100 calibrated cycle 11 windows of the MADIV-16 study (the variance
times WEIGHT across each window against the square of its `SYSCAL` Tsys).
Its shape comes from three places:

- atmospheric lines in the signal band (the 118.75 GHz oxygen line seen from
  the CO(1-0) windows at 114 GHz, water at 183, 325 and 448 GHz, oxygen at
  487 GHz);
- the same in the image band, which a sideband-separating receiver
  (Bands 2 to 8) rejects by about 10 dB and a double-sideband one (Bands 9
  and 10) not at all;
- the receiver, whose temperature ripples by about 2% of Tsys across its IF.

The `atmospheric_tsys` and `receiver_ripple` components of
`--noise-model alma` turn this on. The sky comes from
[`atmosphere.py`](../src/almasim/services/interferometry/atmosphere.py),
which reads a table of CASA's ATM model for the ALMA site
(`scripts/generate_atmosphere_table.py` writes it; CASA is not needed to use
it). Tsys is referred to above the atmosphere,
`(1 + g) exp(A tau) (Trx + Tsky + g Tsky_image)`, with `g` the band's
image-band gain, `A` the airmass, `tau` the zenith opacity, the image band one
IF centre on the other side of the local oscillator, and the sideband whose
image gives the lower Tsys. A model whose `radiometer` sets a
`forward_efficiency` uses the ALMA Technical Handbook's equations instead
(Eqs. 9.5 and 9.6: the forward efficiency, the ground's spillover and, for
the double-sideband bands 9 and 10, both sidebands' receiver noise; see
[`atmospheric_tsys`](noise_models.md#atmospheric_tsys-the-atmospheres-lines-in-tsys)).
The ripple is drawn from the run's seed.

## Per Visibility: the Radiometer Equation

Without `radiometer`, the noise level is set from the run's target S/N
([above](#without-a-noise-model)). With it (on in `--noise-model alma`), each
visibility instead carries the noise ALMA's radiometer equation gives it,
from the same Tsys, per real and imaginary part:

```
sigma = SEFD / (eta_c sqrt(2 n_pol dnu t))      with      SEFD = 2 k Tsys / (A eta_a)
```

with `A` the dish's area and `eta_a` its Ruze aperture efficiency,
`eta_c = 0.96 x 0.88` (quantisation and correlator), `n_pol = 2`, `dnu` the
channel's noise bandwidth (see [Channels](#channels) below) and `t` the
simulator's integration time. The source keeps its modelled flux, so its S/N
follows from the observation. A 12 m visibility at 230 GHz over 15.6 MHz and
6 s then has about 0.1-0.2 Jy of noise, and an exported MS images, in WSClean
or CASA, to the S/N a real observation of that source would reach.

The exported MS's WEIGHT is `1 / mean(sigma^2)` over a row's channels, with
sigma scaled by the gain amplitude the data were multiplied by and by the
baseline's own noise scale (below), as the ALMA pipeline's `statwt` sets it.
Its two hands, XX and YY, each carry their own noise at each visibility's
sigma and with the window's channel correlation, as two real polarisations
do; (XX + YY) / 2 is the simulated Stokes I. See
[`radiometer`](noise_models.md#radiometer-the-thermal-noise) in the guide.

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
over 3840). Every one of the 10,052 cycle 11 windows the rule was checked on
falls in one of the three classes, and on the 100 MADIV-16 windows it gives
each window's `RESOLUTION`, `CHAN_WIDTH`, `EFFECTIVE_BW` and the measured
noise correlation of its neighbouring channels. A resolution off the grid is
taken as the channel width.

The cube's channels have the window's width (their number is
`--n-channels`), and the run's metadata records `channel_averaging`, the
window's N. The `channel_correlation` component's `rho1` and `rho2` are the
native channels' correlation (2/3 and 1/6 in `--noise-model alma`); they are
diluted over the window's N, unless the component's `average` sets N (0, the
default, takes the window's).

## Per Antenna and Per Baseline

Two more terms, also in `--noise-model alma`, come from the same 100 windows:

- Each antenna's noise has its own smooth ripple across the window, beyond
  the array's: about 2% of its noise variance, holding over about 120 MHz
  (the median of the 73 windows wider than 1 GHz; narrow windows show finer,
  ~1 MHz structure). A baseline's noise variance is the product of its two
  antennas'. WEIGHT follows only its mean over the window, as `statwt`'s
  per-row weights do, so it shows as antennas whose noise differs from their
  weights from channel to channel and as heavier tails in the visibility
  noise (the `antenna_ripple` component).
- On bright sources, each baseline keeps an amplitude and phase error per
  scan that does not factor into its antennas' gains, so self-calibration
  cannot remove it, and it limits the continuum's dynamic range: about 6%
  and 2.5 deg (the median of the 13 bright windows where it is measurable;
  the `baseline_errors` component). It acts only on each channel's continuum
  share of the source: what does not close in calibrated data is mostly
  continuum structure the clean model lacks, which leaves line emission
  alone. An external sky model, whose continuum is not known apart, takes it
  on all its emission.

The antenna gains and the slow and fast atmospheric phase, which calibration
also leaves on bright sources, are described in the guide's
[components](noise_models.md#the-components-one-by-one).

## From Real Data to a Released Model

`almasim noise` measures what real calibrated windows carry, reduces the
measurements into a candidate model, checks the candidate against the same
windows and releases it. The
[worked example](noise_models.md#a-complete-worked-example-from-real-data-to-a-released-model)
in the guide runs every step with real numbers; in short:

```
almasim noise windows params.csv --images cubes/ --metadata results.csv \
    --products products.csv --out noise/                                    # windows.csv, rows.csv
almasim noise extract --windows windows.csv --out noise/ --backend slurm    # window.json
almasim noise fit --windows windows.csv --out noise/ --backend slurm        # fit.json
almasim noise aggregate --out noise/                                        # noise/candidate.json
almasim simulation ... --noise-model noise/candidate.json               # try it
almasim noise validate --model noise/candidate.json --windows windows.csv \
    --metadata rows.csv --measured noise/ --out noise/ --backend slurm      # comparison.json
almasim noise promote noise/candidate.json \
    --comparison noise/validation/<sha>/comparison.json                     # models/alma-2.json
```

Each command also takes a single MS (`almasim noise extract
uid___A002_X1.ms.split.cal --spw 31 --field 2 --out noise/`). Reading MSs
needs python-casacore (the `ms-casacore` extra).

**`noise windows`** writes the windows table and the archive rows of an
ALMASim imaging run: the finished science windows of its parameters CSV
(with `image clean-quality`'s `quality.json`),
each with its task folder and log, and its observation's row in the archive
search (linked to the MS's execution block by the resolved-products CSV).
Each window's `pwv` is its execution block's, from its QA0 report, with
`--qa0-report-root` (the extracted auxiliary deliveries), else the archive
row's; a 7 m array's windows have none (its antennas carry no water vapour
radiometer).

**The windows table** (`windows.csv`) has one row per window: the columns
`ms` and `spw`, and optionally `field`, `name`, `syscal_ms`, `line_channels`
(`a~b;c`, left out as line emission), `task_log` (an ALMASim imaging task's
log, whose "lines in channels" gives them), `pwv` (the window's PWV, for the
receiver temperatures and validate) and `task_dir` (the window's ALMASim imaging task, for
`fit` and `validate`). Windows already measured are skipped on a rerun;
failures go to `failures.json`. With `--backend slurm` each window is its own
task on a Slurm worker.

**`noise extract`** measures, from differences of channels (three apart they
remove the sky; one and two apart they show the correlation of neighbours),
so no sky model is needed: the window's exposure (with its flagged share and
the share of rows whose WEIGHT exceeds 10^4, which the imaging leaves out), its noise against WEIGHT
(`scale`, 1 when WEIGHT is right), its channel correlation (`rho1`, `rho2`)
and noise profile, ALMA's Tsys across the window from `SYSCAL`, one Stokes I
visibility's noise at unit time and bandwidth and the efficiency it implies
with that Tsys, and each antenna's own noise shape (`antenna.npz`). Rows are
sampled evenly over the observation, up to `--max-visibilities`, and read in
chunks.

**`noise fit`** fits what calibration leaves on a bright window, the values of
the `slow_phase`, `fast_phase`, `antenna_gains` and `baseline_errors`
components. It predicts the window's clean model (from `--task-dir`: the
continuum model where the image is above 5 sigma, the cube model in the line
channels and line mask) and compares the data with it per (baseline, scan)
where the model is bright (S/N above 10). Each value is fitted against a
reference built by the simulator's own engine, so it is what the simulator
needs to reproduce the data. A window too faint for that is left unfitted,
with a note in `fit.json`.

**`noise aggregate`** takes a released model (`--from`, default `alma`) and
replaces every value the windows measure by their median, recording how many
windows each comes from and their 16-84% spread in `provenance.values`:

| Value | From |
| --- | --- |
| `channel_correlation` rho1, rho2 | windows with Hanning-correlated native channels (rho1 above 0.5) |
| `receiver_ripple` rms | the smooth part (Legendre modes 1-4) of the noise profile over SYSCAL's Tsys², halved |
| `antenna_ripple` rms, scale_mhz | each antenna's noise shape, windows wider than 1 GHz |
| `radiometer` receiver temperatures | per band, the Trx that gives SYSCAL's median Tsys under the simulator's Tsys model, the base model's equation (12 m windows, their PWV and elevation) |
| `radiometer` aperture | per dish size, Ruze's law fitted to the efficiency the noise and Tsys imply |
| `slow_phase`, `fast_phase`, `antenna_gains`, `baseline_errors` | `noise fit` |

A value fewer than `--min-windows` (3) windows measure keeps the base
model's, with a note in `provenance.notes`.

**`noise validate`** simulates each real window with a model (a point
source as bright as the real image's continuum peak, the window's frequency,
channels and channel averaging from its MS's `RESOLUTION`, one execution
block of the MS's time on source by the MS's antennas on their pads, through the
window's PWV or, without one, the PWV its SYSCAL Tsys implies), exports the MS, flags as large a share of its rows as the real
image misses (the window's flagged data and its rows over the imaging's
maximum WEIGHT, `exposure.high_weight_fraction`), measures it with `noise extract` (in a process of
its own) and images it as the real one was imaged. Everything goes under `validation/<model sha256[:12]>/`, so different
models never mix. At the end, `noise compare` writes `comparison.json`: per
metric, the median of the per-window ratio (noise levels) or difference
(shapes), its 16-84% range, a paired Wilcoxon p and how many windows fall
within 10%, and the criteria's verdict. By default a model passes when the
median continuum, cube and visibility noise are within 5% of the real, the
simulated visibilities' variance is within 1% of what their WEIGHT claims,
none of those noise levels differs at p below 0.01, and no band and array
group's median continuum noise is off by more than 10%; `--criteria` takes a
JSON file of other values.

**`noise promote`** writes a candidate as the next version of a released
model, `models/<name>-<n+1>.json` in the working tree, only with a passing
comparison of that very model (matched by its SHA-256). The pull request that
adds the file moves `alma` to the new version (runs that name `alma@1` keep
the old one) and updates the tests that pin the model's values; if the
default model changes what runs produce, it is labelled `golden:supersede`
with a new baseline. The golden simulation runs one case through
`--noise-model alma@1`, so the noise components' code path is under the gate.

The whole chain, extract to validate, is also one pipeline of jobs:
`almasim.jobs.pipeline.noise_model_from_windows(folder, windows.csv,
rows.csv)` returns the four steps, each waiting for the one before it (see
[the guide](noise_models.md#all-of-it-as-one-chain-of-jobs)).

## What Noise Is Not

The background sky features described in [Background](background.md) are not
part of the noise model. ALMASim treats them separately:

- noise is an observing and calibration term;
- the background sky is an additive astrophysical sky component.
