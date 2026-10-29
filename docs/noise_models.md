# Noise Models: a Complete Guide

This page explains, from the beginning, how ALMASim makes its simulated
visibilities look like real, calibrated ALMA data, and how you can measure,
build, test and release the description it uses. It assumes no knowledge of
radio interferometry beyond what it explains, and no knowledge of ALMASim's
internals.

It has four parts:

1. [What a noise model is](#what-a-noise-model-is): the ideas, in plain words.
2. [Using a noise model in a simulation](#using-a-noise-model-in-a-simulation):
   the commands you need to simulate with one, and to pick its parts.
3. [The components, one by one](#the-components-one-by-one): for each part
   of the model, what it is physically, how it shows in real data, exactly
   how it is measured and fitted, how the measurements become the model's
   value, and exactly how the simulator applies it.
4. [A complete worked example](#a-complete-worked-example-from-real-data-to-a-released-model):
   from real archive data to a new, validated, released model, with every
   command and every file, and the numbers real windows give.

[Noise](noise.md) is the shorter overview: the noise a run has without a
model, how ALMA's channels follow from the archive's resolution, and a
summary of the commands. Every option is in the
[command-line reference](cli.md#almasim-noise).


## What a noise model is

### Visibilities, in one paragraph

An interferometer like ALMA does not take pictures. Every pair of antennas
(a *baseline*) measures, many times a second and in many frequency
*channels* at once, one complex number called a *visibility*: the amount of
sky structure at the scale and orientation set by the antennas' separation.
An image is made afterwards by Fourier-transforming many visibilities. ALMA
keeps its visibilities in a *MeasurementSet* (MS), a table with one row per
baseline and time (an *integration*, a few seconds long), each row holding
one visibility per channel and per *hand* of polarisation (XX and YY).

### What a real visibility holds besides the sky

A real, calibrated visibility is the sky's visibility plus things that are
not sky. ALMASim calls each such thing a **component** of the noise model:

| Component | In one sentence |
| --- | --- |
| `radiometer` | Random thermal noise, whose size follows from the system temperature, the dish size, the channel's bandwidth and the integration time. |
| `atmospheric_tsys` | That noise is larger at frequencies where the atmosphere absorbs (oxygen, water lines). |
| `receiver_ripple` | The receiver adds a smooth, slow ripple of the noise across the window. |
| `channel_correlation` | The noise of neighbouring channels is not independent: the correlator smooths it. |
| `antenna_ripple` | Each antenna's noise has its own smooth ripple across the window. |
| `antenna_gains` | Calibration leaves each antenna's amplitude and phase slightly wrong, scan by scan. |
| `slow_phase` | The atmosphere's water vapour delays the signal differently above each antenna; calibration removes most of it, not all. What is left changes from scan to scan. |
| `fast_phase` | The same delay also changes faster than calibration can follow, within a scan. |
| `baseline_errors` | A small error per baseline that does not belong to either antenna, so calibration cannot remove it. |

A **noise model** says which of these components a simulation applies and
how strong each one is. It is a small JSON file. ALMASim ships the released
ones (today `alma`, version 1, set from measurements of 100 real, calibrated
ALMA cycle 11 windows; its `provenance` says where each value comes from)
and you can write or build your own.

### Words used on this page

- **Window** (spectral window, spw): one contiguous band of channels in one
  observation. The measurements are made per window.
- **Scan**: a stretch of a few minutes on the target between calibrator
  visits; calibration solutions hold for a scan.
- **WEIGHT**: the MS column that says how noisy each visibility is, as
  1 / sigma^2. A visibility whose noise matches its WEIGHT has
  (noise variance) x WEIGHT = 1.
- **Tsys**: the system temperature, in kelvin. It measures how much noise
  the whole system (receiver plus atmosphere) adds. ALMA measures it during
  every observation and stores it in the MS's `SYSCAL` table.
- **Released model**: a model file shipped in ALMASim, named and versioned
  (`alma@1`). It is never edited; a change becomes a new version.
- **Candidate model**: any other model file, for example one you built from
  your own measurements. You can simulate with it straight away; it becomes
  a released version only after it passes validation.


## Using a noise model in a simulation

### The shortest path

```
almasim simulation --metadata-path rows.csv --row-idx 0 \
    --output-dir sims/ --project-name demo --noise-model alma
```

`--noise-model` names a model:

| You write | You get |
| --- | --- |
| `alma` | the newest released version of `alma` |
| `alma@1` | exactly version 1 of `alma`, even after newer ones are released |
| `path/to/model.json` | the model in that file (a candidate, or any model) |
| `'{"name": "x", "version": 1, "components": {...}}'` | a model written inline |
| `none` (or leaving the option out) | no noise model: white noise set by the run's target S/N |

`--corruptions` is the option's earlier name and still works. The flat JSON
objects it used to take (`{"gain_amp_rms": 0.05}`) are still accepted and are
turned into components.

### Choosing components

Every component can be switched off, or you can keep only some:

```
# Everything alma has, except the in-scan phase and the baseline errors:
almasim simulation ... --noise-model alma --noise-without fast_phase,baseline_errors

# Only the thermal noise and its channel correlation:
almasim simulation ... --noise-model alma --noise-only radiometer,channel_correlation
```

The components are **separable**: each one that draws random numbers draws
them from its own stream, so switching one off leaves the other components'
realisations as they were. Two runs with the same seed, one with and one
without `fast_phase`, differ only by the fast phase. (`channel_correlation`
is the exception: it shapes the thermal noise itself, so switching it off
changes the thermal noise's draws.)

A selection is part of the model's identity. The run records it: in the
products' metadata (`noise_model`: name, version, label such as
`alma@1 without fast_phase,baseline_errors`, the SHA-256 of the model's JSON,
and the components applied), in `sim_params_<idx>.txt` (`Noise Model: ...`)
and in the exported MS's `HISTORY` table.

### Looking at models

```
almasim noise show                    # the released models and their versions, and where each comes from
almasim noise show alma               # one model: each component, on or off, with its values
almasim noise show alma --json        # the model's JSON
almasim noise show my-candidate.json --noise-without slow_phase
almasim noise components              # every component and every parameter, with its meaning
almasim noise schema -o schema.json   # the JSON Schema a model file must satisfy
```

### The team's own models

Besides the models released with ALMASim (`alma`), a site can keep its own:
the same `<name>-<version>.json` files in one folder, which every user's
simulations and the console's Simulate page see. The folder is the site
profile's

```toml
[tools]
noise_models = "/SKAO_data/cycle11/almasim/noise-models"
```

(the shared install makes it, group-writable), or the folder
`ALMASIM_NOISE_MODELS` names, which wins. Jobs get the profile in their
environment, so a Slurm job resolves `--noise-model zwicky-alma@1` as the
console does. A model there is named like a released one (`zwicky-alma`, the
newest version; `zwicky-alma@1`), but may not take a released model's name:
`alma` is always ALMASim's own, and a file `alma-2.json` in the folder is
ignored with a warning, as is a file that does not load or whose name
disagrees with the model it holds. `almasim noise show` marks each model
`(package)` or `(site)`. A model is released into the folder with `almasim
noise promote … --site --name zwicky-alma` (the console's *Release…* on the
Noise models page), only with a passing validation; its provenance notes
who released it and when.

### From Python

```python
from almasim.services.noise_model import released_model, resolve_noise_model
from almasim.services.simulation import SimulationParams

alma = released_model("alma", 1)
quiet = alma.select(without=["fast_phase", "baseline_errors"])
print(quiet.label(), quiet.sha256()[:12])  # alma@1 without fast_phase,baseline_errors
params = SimulationParams.from_metadata_row(row, ..., noise_model=quiet)

candidate = resolve_noise_model("noise/candidate.json")
```

### The model file

This is `alma@1`, annotated (the real file has no comments):

```
{
  "schema_version": 1,          // the file format's version
  "name": "alma",               // lower case, digits, - and _
  "version": 1,                 // a released version never changes
  "description": "...",
  "provenance": {               // where the values come from
    "method": "hand-set from per-window fits",
    "data": "MADIV-16, 100 windows of ALMA cycle 11 calibrated data",
    "windows": 100,
    "date": "2026-10-25",
    "notes": "...",             // which value comes from which windows
    "values": null              // filled by noise aggregate: per value, n windows and spread
  },
  "settings": {                 // values several components share
    "scan_length_s": 300.0,     // gains, slow phase and baseline errors hold for one scan
    "outer_scale_km": 3.0       // the largest scale of the turbulent screens
  },
  "components": {               // a component left out is not applied
    "radiometer": {"receiver_temperature_k": [...], "aperture": [...],
                   "quantisation_efficiency": 0.96, "correlator_efficiency": 0.88},
    "atmospheric_tsys": {},
    "receiver_ripple": {"rms": 0.02},
    "channel_correlation": {"rho1": 0.6667, "rho2": 0.1667},
    "antenna_ripple": {"rms": 0.02, "scale_mhz": 120.0},
    "antenna_gains": {"amp_rms": 0.08},
    "slow_phase": {"path_um_1km": 80.0, "slope": 0.45},
    "fast_phase": {"path_um_1km": 40.0, "slope": 0.3, "coherence_s": 10.0},
    "baseline_errors": {"amp_rms": 0.06, "phase_rms_deg": 2.5}
  }
}
```

Every component also takes `"enabled": false`, which keeps it in the file but
does not apply it. Parameters have bounds (a correlation between 0 and 1, an
rms not below 0, ...); a file with an unknown component, an unknown
parameter or a value out of bounds is refused with a message naming it.
`almasim noise schema` prints the full JSON Schema, also kept in the
repository as [`docs/noise-model.schema.json`](noise-model.schema.json).


## The components, one by one

Each section answers the same five questions:

1. **What it is**, physically.
2. **How it shows in real data.**
3. **How it is measured** from a real window: by `almasim noise extract`
   (which needs only the calibrated MS) or `almasim noise fit` (which also
   needs the window's clean model, from imaging it).
4. **How it becomes the model's value**: by `almasim noise aggregate`, over
   many windows.
5. **How the simulator applies it.**

Two measurement ideas recur, so they are explained once here.

**Channel differences.** The sky changes little from one channel to the
next, the noise changes completely. So the difference of a visibility in
channel k and in channel k + l is almost pure noise, and its spread measures
the noise without knowing the sky. For complex noise with variance sigma^2
in each of its real and imaginary parts, independent between channels,

```
mean |V(k) - V(k+l)|^2 = 2 x 2 sigma^2 = 4 sigma^2
```

so `|V(k) - V(k+l)|^2 / 4` is an estimate of sigma^2, and multiplied by the
visibility's WEIGHT it should be 1 if WEIGHT is right. If neighbouring
channels are correlated by rho_l, the difference is smaller,
`4 sigma^2 (1 - rho_l)`, which is how the correlation is measured (below).
`noise extract` averages these quantities over many rows (evenly spread over
the observation, up to `--max-visibilities`, 40 million by default), leaving
out any value 20 times above its channel's median (a spectral line's edge,
interference: noise alone exceeds that about once in a million).

**Data over model.** On a bright source, divide each visibility by what the
clean model predicts for it: what is left, `V / M`, is 1 for perfect data,
and its phase and amplitude show what calibration left behind. `noise fit`
does this per *group*: all the integrations of one baseline in one scan,
`r = sum(w V conj(M)) / sum(w |M|^2)`, with w = WEIGHT / scale. It keeps the
groups where the model is bright enough to see small errors (the group's
signal-to-noise above 10, i.e. `sum(w |M|^2) > 100`), sorts them by baseline
length into 10 bins (log-spaced between the 0.5th and 99.5th percentile of
the baseline lengths) and, per bin, takes the rms of the phase of r and the
mean of ln|r|. Thermal noise alone also scatters r; how much is computed for
the same groups (a reference, below), and only the excess over it is
attributed to the component being fitted.

### `radiometer`: the thermal noise

**What it is.** Every receiver adds random noise. Its size follows the
*radiometer equation*: the noise of one visibility (per real and imaginary
part, in Jy) is

```
sigma = SEFD / (eta_c sqrt(2 n_pol dnu t))      with      SEFD = 2 k Tsys / (A eta_a)
```

where `k` is Boltzmann's constant, `Tsys` the system temperature, `A` the
dish's area, `eta_a` its aperture efficiency, `eta_c` the quantisation and
correlator efficiency, `n_pol = 2` the hands averaged into Stokes I, `dnu`
the channel's *noise bandwidth* (wider than its width when channels are
smoothed; the MS's `EFFECTIVE_BW`) and `t` the integration time. `Tsys` is
the receiver's temperature `Trx` plus the sky's (with `atmospheric_tsys`,
referred to above the atmosphere).

**How it shows in real data.** As noise that is the same, on average, in
every channel and on every baseline, and that integrates down as 1/sqrt(t)
in images.

**How it is measured** (`noise extract`):

- *The noise against WEIGHT.* From the lag-3 channel differences (three
  channels apart, independent even for smoothed channels), per channel,

  ```
  var3(k) = mean over rows and hands of |V(k) - V(k+3)|^2 / 4 x WEIGHT
  scale   = mean over channels of var3
  ```

  `scale` is 1 when the MS's WEIGHT matches its noise. Real ALMA data give
  1.0 to 1.3 (the pipeline's weights are slightly optimistic); it is
  recorded as `thermal.scale`.
- *One visibility's noise.* Stokes I, `I = (XX + YY) / 2`, from its lag-3
  differences: `sigma_I^2 = mean |I(k) - I(k+3)|^2 / 4` (values 20 times
  their channel's median left out), and the *unit noise*
  `unit = sigma_I sqrt(t w)` (`visibility.unit_jy`, in Jy s^0.5 Hz^0.5),
  which no longer depends on the integration time `t` and channel width
  `w` (the MS's `CHAN_WIDTH`), so windows can be compared.
- *Tsys.* From the MS's `SYSCAL` table: the Tsys window on the same baseband
  that covers the science window's channels, its spectrum's median over
  antennas and times, the mean of its two hands, its rolled-off edges
  (1/32 of its channels at each end) left out, interpolated onto the
  science channels (`tsys.median_k`, and its shape, below).
- *The efficiency it implies.* Solving the radiometer equation for the total
  efficiency with the measured Tsys and unit noise:

  ```
  eta = 2 k Tsys / (2 A unit) / sqrt(EFFECTIVE_BW / w)
  ```

  (`visibility.efficiency`, with the Tsys in kelvin and the unit noise
  converted to SI; it is eta_a x eta_c, 0.4 to 0.65 on real data).

**How it becomes the model's value** (`noise aggregate`):

- *Receiver temperatures.* For each 12 m window with a Tsys, a PWV (the
  archive's precipitable water vapour, given to `noise extract --pwv`) and
  an elevation, ALMASim's own Tsys model is evaluated at `Trx` = 0 K and
  100 K (it is linear in `Trx`) to find the `Trx` that gives the measured
  median Tsys. The median over the windows of each row of the table (one
  band, or bands 2-3 and 9-10 together) replaces that row's receiver
  temperature.
- *Aperture efficiency.* For each dish size (12 m, 7 m), Ruze's law
  `eta_a(nu) = eta0 exp(-(4 pi eps nu / c)^2)`, the surface's rms `eps`
  costing more at higher frequency, is fitted (robustly, in log) to the
  windows' `efficiency / eta_c`. A dish size needs at least five windows
  (or `--min-windows`, if more).

**How the simulator applies it.** For each interferometric array
configuration (total power keeps the S/N-scaled noise) ALMASim computes, per
channel, the Tsys (receiver temperature from the model's table by
frequency, plus the sky; see `atmospheric_tsys`), the SEFD with the
dish's Ruze efficiency at the window's mean frequency, and sigma from the
radiometer equation with the window's noise bandwidth and the simulator's
own integration time. Every visibility gets complex Gaussian noise of that
sigma, the source keeping its modelled flux, so the S/N is what a real
observation would reach. The exported MS's WEIGHT is
`1 / mean over channels of (sigma^2 |g|^2)` (g: the gain errors below), as
the ALMA pipeline's `statwt` sets it, and its two hands carry independent
noise (XX = I + d, YY = I - d), each at half the Stokes I weight.

Without `radiometer`, the noise is instead scaled to the run's target S/N
(from the archive's sensitivities; see
[Noise](noise.md#without-a-noise-model)), and only its shape across the
window follows Tsys.

| Parameter | Meaning | alma@1 |
| --- | --- | --- |
| `receiver_temperature_k` | Trx by frequency: rows of `upper_ghz` (the row's upper edge, GHz; null for the last), `trx_k` (K) and `bands`, edges rising | 32 (band 1), 39 (2-3), 37 (4), 50 (5), 46 (6), 63 (7), 110 (8), 160 (9-10) |
| `aperture` | Ruze parameters per dish: rows of `diameter_m`, `eta0` and `surface_rms_m` (another diameter takes the 12 m dish's) | 12 m: 0.76, 27 um; 7 m: 0.72, 24 um |
| `quantisation_efficiency`, `correlator_efficiency` | eta_c = their product | 0.96 x 0.88 |

Left out, these take ALMA's values from
`interferometry/data/alma_radiometer.json`, which every run uses for Tsys.

### `atmospheric_tsys`: the atmosphere's lines in Tsys

**What it is.** Oxygen and water absorb and emit at particular frequencies
(oxygen at 60 and 118.75 GHz, water at 183, 325 and 448 GHz, oxygen again at
487 GHz). Near them the sky is hot and opaque, so Tsys, and the noise, rise
steeply. Receivers also pick up some signal from the *image band*, mirrored
about the local oscillator, whose lines then show too.

**How it shows in real data.** As a noise level that rises towards one edge
of a window, or peaks inside it, matching the Tsys spectrum in `SYSCAL`.

**How it is measured** (`noise extract`): the noise's shape across the
window, `thermal.channel_profile` = `var3(k) / scale`, and SYSCAL's Tsys
shape, `tsys.profile` = `Tsys(k)^2 / mean` (the noise *variance* follows
Tsys squared). On the 100 MADIV-16 windows the measured profile follows
SYSCAL's Tsys^2 to 3.7% rms, against 7.3% for a flat one.

**How it becomes the model's value.** It is a switch, with no parameter:
present in a model, it is on.

**How the simulator applies it.** Tsys comes from a table of CASA's ATM
model for the ALMA site (`interferometry/data/atm_chajnantor.npz`; CASA is not
needed to use it) at the run's PWV and the target's elevation:

```
Tsys = (1 + g) exp(A tau) (Trx + Tsky + g Tsky_image)
```

with `A` the airmass, `tau` the zenith opacity, `g` the band's image-band
gain (0 for band 1, 0.1 for the sideband-separating bands 2 to 8, whose
rejection is about 10 dB, and 1 for the double-sideband bands 9 and 10) and
the image band one IF centre on the other side of the local oscillator, on
whichever side gives the lower Tsys. Off, the sky is a smooth function of
frequency instead, and Tsys is simply `Trx + Tsky`, with
`Tsky = Tground (1 - exp(-tau))` and
`tau = (0.005 + 0.0025 PWV) A (1 + nu / 400 GHz)` (PWV in mm).

### `receiver_ripple`: the receiver's ripple

**What it is.** A receiver's own temperature is not flat across its
intermediate-frequency band: it ripples smoothly by a few percent.

**How it shows in real data.** As the part of the noise's shape across a
window that SYSCAL's Tsys (measured at coarser resolution) does not follow.

**How it is measured** (`noise extract`, then `noise aggregate`): the ratio of
the measured noise profile to SYSCAL's Tsys^2 profile, 5% of the channels at
each edge left out, fitted with Legendre polynomials of order 1 to 4 (smooth
shapes, so the per-channel sampling noise stays out); the rms of that smooth
fit, halved (the profile is a variance, Tsys squared, so its ripple is twice
Tsys'), is one window's ripple. Windows with fewer than 32 channels or no
SYSCAL Tsys are skipped. The model's `rms` is the median over windows.

**How the simulator applies it.** Per array configuration, a random smooth
curve across the window, the sum of Legendre modes 1 to 4 with equal
expected power and expected rms `rms`, multiplies every channel's noise
sigma: `sigma(k) x (1 + ripple(k))`. It is drawn from the run's seed (its own
stream, `receiver`), so it is the same in every channel task.

| Parameter | Meaning | alma@1 |
| --- | --- | --- |
| `rms` | the ripple's rms, as a fraction of Tsys | 0.02 |

### `channel_correlation`: correlated neighbouring channels

**What it is.** ALMA's correlator smooths each spectrum with a Hanning window
before cutting it into channels. The noise of a channel then shares part of
its neighbours': the correlation of channels one apart is rho1 = 2/3 and two
apart rho2 = 1/6. When the correlator also averages N native channels into
each output channel, the correlation falls: 0.30, 0.115, 0.052 and 0.026 for
N = 2, 4, 8 and 16.

**How it shows in real data.** As spectra whose noise looks smoother than
white noise, and as images of neighbouring channels that are not
independent (the cube's adjacent-channel correlation).

**How it is measured** (`noise extract`): from the lag-1 and lag-2 channel
differences, which correlated noise makes smaller:

```
var_l(k) = mean |V(k) - V(k+l)|^2 / 4 x WEIGHT        (l = 1, 2)
rho_l    = 1 - mean(var_l) / mean(var3)
```

(`thermal.rho1`, `thermal.rho2`). Line channels, given with
`--line-channels` or found in the window's imaging log, are left out: there
the differences hold sky as well as noise.

**How it becomes the model's value.** The median of rho1 and rho2 over the
windows whose channels are native (rho1 above 0.5, i.e. not averaged). The
model holds the *native* correlation; the averaging of each simulated window
is applied on top.

**How the simulator applies it.** It first dilutes rho1 and rho2 over the
window's averaging N (from the archive's spectral resolution, unless
`average` sets it; see [Noise](noise.md#channels)), then builds a three-tap
kernel (a, b, a) with unit norm whose output has those correlations
(rho1 = 2ab, rho2 = a^2; when no such kernel exists, the one that matches
rho1), and makes each channel's noise as `a w(k-1) + b w(k) + a w(k+1)` from
white draws `w`, each keyed by its channel, so that channel tasks running
separately share their neighbours' draws.

| Parameter | Meaning | alma@1 |
| --- | --- | --- |
| `rho1`, `rho2` | the native channels' noise correlation one and two apart | 2/3, 1/6 (Hanning) |
| `average` | native channels per channel; 0: the window's own | 0 |

### `antenna_ripple`: each antenna's own noise shape

**What it is.** Each antenna has its own receiver and IF chain, so its noise
across the window has its own smooth shape, on top of the array's common
one. A baseline's noise variance is then the product of its two antennas'
shapes.

**How it shows in real data.** As antennas whose noise differs from what
their WEIGHT says by a few percent, differently in different channels, and
as heavier tails in the distribution of the visibility noise.

**How it is measured** (`noise extract`): for each antenna `a`, channel `k`
and hand `h`, the mean lag-3 variance `v(a, k, h)` over the baselines that
include that antenna. It is factored into an antenna term, a channel term and
what is left, `v = A(a, h) C(k, h) L(a, k, h)`, by five rounds of alternating
means (the antennas with at least half the typical amount of data, the
channels with more than 50 samples for every antenna). The leftover `L - 1`
is the antennas' own shape; its rms beyond what sampling alone gives (`sqrt(mean(1 / n))`, n the samples
per value) is `antenna_shape.excess_rms`, and the channel lag at which its
autocorrelation falls to half, times the channel width, is
`antenna_shape.half_width_mhz`. The shapes themselves, smoothed over 16
channels, are saved in `antenna.npz`.

**How it becomes the model's value.** The medians of `excess_rms` and
`half_width_mhz` over the windows wider than 1 GHz (narrow windows show only
finer structure).

**How the simulator applies it.** For each antenna, white draws across the
window's channels are smoothed by a Gaussian whose autocorrelation falls to
half at `scale_mhz` and scaled to rms `rms`: `g(a, k) = 1 + rms x smooth(k)`.
Each baseline's noise sigma is multiplied by `sqrt(g(i, k) g(j, k))`.
WEIGHT does not follow it, as in real data.

| Parameter | Meaning | alma@1 |
| --- | --- | --- |
| `rms` | fractional rms of an antenna's noise variance across the window | 0.02 |
| `scale_mhz` | frequency scale of the ripple (half-width of its autocorrelation) | 120 |

### `antenna_gains`: what calibration leaves on each antenna

**What it is.** Calibration measures each antenna's gain (amplitude and
phase) on a calibrator and transfers it to the target. The transfer is not
perfect: each antenna keeps a small amplitude and phase error per scan.

**How it shows in real data.** On bright sources only (on faint ones the
thermal noise hides it): per scan, the data over the model is off by a
factor that belongs to each antenna, the same on all its baselines.

**How it is fitted** (`noise fit`): per scan, the bright groups' ratios are
split into antenna terms by least squares,

```
ln|r(i,j)| = a(i) + a(j)            arg r(i,j) = p(i) - p(j)
```

The rms of the antenna terms `a` (each scan's mean removed) is the
antennas' amplitude scatter, `ant_lnamp_rms`. The same is computed for a
reference (the clean model times the phase screens already fitted, plus
thermal noise; the `slow_phase` section below explains how references are
built), and the gain amplitude error is the excess in quadrature:

```
gain_amp_rms = sqrt(real ant_lnamp_rms^2 - reference ant_lnamp_rms^2)
```

It is fitted twice: once against the slow screen alone (so the fast screen
can be fitted against the gains), and once more with the fast screen too
(its decorrelation also scatters the antennas' amplitudes).

**How it becomes the model's value.** The median of `gain_amp_rms` over the
fitted windows. The phase error is not fitted, so `phase_rms_deg` keeps the
base model's.

**How the simulator applies it.** For each run of an array configuration
(one per execution block) and each scan, each antenna draws an amplitude
`1 + N(0, amp_rms)` (clipped at 0) and a phase `N(0, phase_rms_deg)`. Every
visibility of baseline (i, j) in that scan
has its sky part multiplied by `g(i) conj(g(j))`, and its noise by the gain's
amplitude `|g(i) g(j)|` (as in real data, where calibration scales the noise
with the gain; the exported WEIGHT includes it). The draws come from the
gains' own random stream.

| Parameter | Meaning | alma@1 |
| --- | --- | --- |
| `amp_rms` | fractional amplitude error per antenna and scan | 0.08 |
| `phase_rms_deg` | phase error per antenna and scan (degrees) | 0 |

### `slow_phase`: the per-scan atmospheric phase

**What it is.** Water vapour slows the signal down. Over each antenna there
is a different amount of it, so each antenna sees an extra path length, and
the phase difference on a baseline grows with the antennas' separation (two
antennas close together look through nearly the same air). ALMA corrects it
with water-vapour radiometers and calibrator visits; what is left changes
from scan to scan. The extra path is a *turbulent screen*: its rms difference
between two points grows as separation^slope (5/6 for ideal Kolmogorov
turbulence) up to an outer scale of a few km.

**How it shows in real data.** On bright sources: the phase of data over
model, averaged over a scan, scatters more on long baselines than on short
ones, more than thermal noise explains.

**How it is fitted** (`noise fit`):

1. *The data.* Per bin of baseline length, the rms phase of the bright
   groups' ratios r, `phi_real(b)`.
2. *The references.* The same statistic for the clean model plus thermal
   noise only, `phi_thermal(b)`. It is built without reading the data again:
   a group's `sum w V conj(M)` for data `G M + noise` is
   `sum(G w |M|^2)` plus complex Gaussian noise with variance
   `kappa sum(w |M|^2)` in each part, where `kappa = 1 + 2 rho1 + 2 rho2`
   accounts for the channels' correlation (summing correlated channels adds
   less independent noise). With errors G drawn by the simulator's own
   engine for the window's real antennas, times and scans, the same gives
   references for any combination of components.
3. *A first guess.* The excess `sqrt(phi_real^2 - phi_thermal^2)` per bin,
   fitted with a power law in log-log, gives a trial strength P0.
4. *The fit through the engine.* For each slope on a grid (0.05, 0.2, ...,
   0.95) the engine draws its screen at strength P0 on the window's real
   antenna layout, and the simulated excess variance per bin is computed.
   Because the phase grows linearly with the screen's strength, the excess
   variance grows as its square: the real excess is fitted as
   `a x simulated excess`, by least squares over the bins, and the
   strength is `P0 sqrt(a)`. The slope that fits best is kept.

   Fitting through the engine matters: the engine's screens, on a 10 m grid
   with an outer scale, do not follow the nominal power law on very short or
   very long baselines, so a power law fitted to the data directly would
   give values the simulator would not reproduce. This way, the fitted value
   is what the simulator needs to make data like the real data.

At least three bins with 20 groups each are needed, or no slow phase is
fitted. A window with fewer than 20 bright groups in all is too faint for
any of the fits (`fit.json` says so in `notes`).

**How it becomes the model's value.** The medians of `path_um_1km` and
`slope` over the fitted windows.

**How the simulator applies it.** For each scan it draws a turbulent screen
(power spectrum `k^-(2 slope + 2)` with the outer scale, on 10 m pixels over
the array), scaled so its rms path difference 1 km apart is `path_um_1km`
microns (set at 150 m, where the screen follows its power law, and carried
to 1 km along it), and reads each antenna's extra path at its position. The
phase of baseline (i, j) is `2 pi (path(i) - path(j)) / wavelength`, so it is larger
at higher frequencies, as in real data. It multiplies the sky part of the
visibilities, constant over the scan.

| Parameter | Meaning | alma@1 |
| --- | --- | --- |
| `path_um_1km` | rms path difference of antennas 1 km apart (microns) | 80 |
| `slope` | growth of the difference with separation | 0.45 |

### `fast_phase`: the atmospheric phase within a scan

**What it is.** The same water vapour also moves on time scales of seconds,
faster than calibration follows. Within a scan, the phase wanders around its
scan average.

**How it shows in real data.** Averaging a scan's wandering phases lowers
the average's amplitude (*decorrelation*): for a phase of rms sigma about
its mean, `|A| = exp(-sigma^2 / 2)`. So on bright sources, the scan-averaged
data over model is a little *smaller* than 1, the more so on longer
baselines.

**How it is fitted** (`noise fit`): from the mean of ln|r| per bin. The loss
is the reference's mean ln|r| minus the data's, `loss(b)`. The reference
holds the slow screen and the gains already found, because gain amplitude
errors also lower ln|r|, by the same amount on every baseline. The loss is
fitted, through the engine as for the slow phase, as

```
loss(b) = c + a x (simulated loss of the engine's fast screen at strength P0)(b)
```

The constant `c` takes what lowers ln|r| on every baseline alike (leftover
gain errors, a clean model whose flux is a little off), which is not
decorrelation. A fast screen is kept only if the loss grows with baseline by
more than three times the fit's scatter. A window whose data rise above
the model (mean ln|r| above 0.05 in any bin) has a clean model that does not
match its source, and gets no fast fit.

**How it becomes the model's value.** The medians of `path_um_1km` and
`slope` over the fitted windows. The coherence time is not fitted:
`coherence_s` keeps the base model's.

**How the simulator applies it.** Per integration, a fresh turbulent screen,
blended with the previous one as `a x previous + sqrt(1 - a^2) x fresh` with
`a = exp(-dt / coherence_s)`, so the phase wanders with that coherence time.
Each antenna's mean over its scan is removed (that part is the slow phase's),
and the whole is scaled so the median antenna pair's rms path difference
follows `path_um_1km x (separation / 1 km)^slope`. Applied like the slow
phase, but changing integration by integration.

| Parameter | Meaning | alma@1 |
| --- | --- | --- |
| `path_um_1km` | rms path difference 1 km apart within a scan (microns) | 40 |
| `slope` | growth with separation | 0.3 |
| `coherence_s` | how long the in-scan phase stays coherent (s) | 10 |

### `baseline_errors`: errors that do not close

**What it is.** Some errors belong to a baseline, not to an antenna: the
sky structure the clean model lacks, correlator effects. Because they do not
factor into antenna terms, calibration (including self-calibration) cannot
remove them. They limit how bright a source's surroundings can be imaged
cleanly (the *dynamic range*).

**How it shows in real data.** On bright sources, after the per-scan antenna
solve, the ratios still scatter: their *closure* residuals.

**How it is fitted** (`noise fit`): the rms of the antenna solve's residuals,
in amplitude (`nonclosing_lnamp_rms`) and phase (`nonclosing_phase_rms_deg`),
of the data, minus in quadrature those of the reference holding everything
else found (screens and gains, plus thermal noise):

```
baseline_amp_rms = sqrt(real nonclosing_lnamp_rms^2 - reference nonclosing_lnamp_rms^2)
```

**How it becomes the model's value.** The medians over the fitted windows.

**How the simulator applies it.** For each baseline and scan, a complex
factor `b = exp(N(0, amp_rms) + i N(0, phase_rms))`, with `phase_rms` the
`phase_rms_deg` in radians, from its own random stream. It multiplies only
each channel's *continuum* share of the source (`1 + f (b - 1)`, with `f`
the continuum's fraction in that channel), since what does not close in real data is mostly continuum structure; line
emission is left alone. An external sky model, whose continuum is not known
separately, takes it on all its emission.

| Parameter | Meaning | alma@1 |
| --- | --- | --- |
| `amp_rms` | fractional amplitude error per baseline and scan | 0.06 |
| `phase_rms_deg` | phase error per baseline and scan (degrees) | 2.5 |

### `settings`

Not components, but values components share:

| Setting | Meaning | alma@1 |
| --- | --- | --- |
| `scan_length_s` | the length of a scan: gains, slow phase and baseline errors hold for one, and the exported MS numbers its scans by it; 0 (the default when left out) makes the whole observation one scan | 300 |
| `outer_scale_km` | the largest scale of the turbulent screens | 3 |


## A complete worked example: from real data to a released model

This example builds a new `alma` model from real ALMA windows, checks it,
and releases it. Every step runs on a Slurm cluster (`--backend slurm`); each
also runs on one machine without it. Each step's output is the next one's
input, all in one folder (`noise/` below). The numbers shown come from real
ALMA cycle 11 windows (those of the MADIV-16 study that `alma@1` was set
from).

### What you need

1. **Calibrated measurement sets** of the windows you want to learn from.
   `almasim products download --generate-calibrated-visibilities` produces
   them from the ALMA archive (`<uid>.ms.split.cal`; they keep `SYSCAL`).
2. **Their imaging**, made by ALMASim (`almasim image compute-parameters`
   and `almasim image image-from-ms ... --cube --uv-contsub`, then
   `almasim image clean-quality`). Each window's *imaging task* directory
   (for example `cubes/uid___A002_X114a61b_X31de.ms.split/SPW-31_FIELD-2/`)
   holds its clean models, residuals and `quality.json`. `noise fit` reads
   the clean models; `noise validate` compares the images.
3. **The archive rows** of the windows (`almasim metadata --save-csv
   rows.csv ...`), with a `name` column added that names each row's window.
   `noise validate` simulates each window from its row. `almasim noise
   windows` (below) writes them for an imaging run.
4. **python-casacore** (`pip install 'almasim[ms-casacore]'`), to read the
   MSs, and WSClean and CASA data for the imaging.

### The windows table

One row per window, as a CSV (`windows.csv`). Only `ms` and `spw` are
required; the other columns enable more of the chain:

| Column | Meaning | Used by |
| --- | --- | --- |
| `ms` | the calibrated MS | all |
| `spw` | its spectral window, in the MS's own numbering | all |
| `field` | the target field's id or name (default: every field) | all |
| `name` | the window's name (default: uid, spw and field) | all |
| `pwv` | the archive's PWV for the observation (mm) | aggregate (receiver temperatures) |
| `line_channels` | channels of line emission to leave out, `a~b;c` | extract |
| `task_log` | the imaging task's log, whose "lines in channels" gives them | extract |
| `syscal_ms` | another MS to read SYSCAL from (default: `ms`) | extract |
| `task_dir` | the window's imaging task | fit, validate |

```
ms,spw,field,name,pwv,task_log,task_dir
/data/uid___A002_X114a61b_X31de.ms.split.cal,31,2,w81,1.1,/data/cubes/uid___A002_X114a61b_X31de.ms.split/SPW-31_FIELD-2.log,/data/cubes/uid___A002_X114a61b_X31de.ms.split/SPW-31_FIELD-2
/data/uid___A002_X11cd86f_X53c7.ms.split.cal,11,2,spt0441,0.6,...,...
```

`almasim noise windows` writes this table, and the archive rows `noise
validate` needs, from an ALMASim imaging run:

```
almasim noise windows imaging_parameters.csv --images cubes/ \
    --metadata results.csv --products products.csv --out noise/
```

It takes the windows of the parameters CSV that the run under `--images`
finished and `image clean-quality` assessed (their `quality.json`, which
`noise fit` and `validate` read; `--all-tasks` takes them all) and that are science targets
(`--all-fields`: calibrators too), each with its task folder and log. Each
window's archive row is its observation's row in the archive search
(`--metadata`), the row of the window's own source when the observation has
several; its `PWV` fills `pwv`. The resolved-products CSV (`--products`;
default: `*products*.csv` beside the search) links each MS's execution block
to its observation. It writes `noise/windows.csv` and `noise/rows.csv` (the
archive rows, with a `name` column), and lists the windows it left out and
why. The console's *Noise models* page runs it for you.

### Step 1: measure every window (`noise extract`)

```
almasim noise extract --windows windows.csv --out noise/ --backend slurm \
    --slurm-queue <partition> --slurm-workers 8
```

Each window becomes its own Slurm task. It writes
`noise/windows/<name>/window.json` and `antenna.npz`. Windows already
measured are skipped when you run it again; failures are listed in
`noise/failures.json` and do not stop the others. Six real windows took 53 s
and 1.4 GB in all.

This is window 81 (band 6, 12 m), shortened:

```
{
  "name": "w81",
  "exposure": {
    "antennas": 43, "baselines": 903, "integrations": 55, "integration_s": 6.048,
    "time_on_source_s": 332.6, "channels": 240, "channel_width_hz": 7812500.0,
    "noise_bandwidth_hz": 8196777.3, "frequency_ghz": 267.73,
    "flagged_fraction": 0.0033, "elevation_deg": 60.0
  },
  "thermal": {
    "scale": 1.297,        // the noise is 1.3 x what WEIGHT says (in variance)
    "rho1": 0.025,         // channels averaged (N = 4 or more): nearly independent
    "rho2": -0.001,
    "channel_profile": [0.8727, 0.8727, 0.8727, 0.8639, ...]
  },
  "visibility": {
    "sigma_jy": 0.369,     // one Stokes I visibility (6 s, 7.8 MHz)
    "unit_jy": 2538.7,     // sigma x sqrt(t x w)
    "efficiency": 0.628    // aperture x correlator efficiency implied with Tsys
  },
  "tsys": {"median_k": 133.7, "profile": [0.8452, 0.8481, ...], "spectral_window": 23},
  "antenna_shape": {"excess_rms": 0.038, "half_width_mhz": 93.75, "antennas_used": 43}
}
```

### Step 2: fit the bright windows (`noise fit`)

```
almasim noise fit --windows windows.csv --out noise/ --backend slurm \
    --slurm-queue <partition> --slurm-memory 64GB
```

It needs Step 1's `window.json` (it reads the noise level and channel
correlation from it) and the `task_dir` column. It writes
`noise/windows/<name>/fit.json`; failures are listed in
`noise/failures-fit.json`. Faint windows get a `fit.json` saying why nothing
was fitted. By default the clean model is the continuum model where the real
continuum image is above 5 sigma (widened by the beam) plus the cube model
in the line channels, inside the line mask (`--continuum-model`,
`--line-model` to change). Five bright windows took 47 minutes and up to
24 GB (most of it Sgr A, an 83 GB MS).

This is SPT0441-46 (band 7), shortened:

```
{
  "bright_groups": 3678,             // (baseline, scan) groups with S/N above 10
  "kappa": 2.67,                     // Hanning-correlated channels: 1 + 2 x 2/3 + 2 x 1/6
  "bins": [
    {"baseline_m": 80.2, "groups": 754,
     "phase_rms_deg": 15.9, "thermal_phase_rms_deg": 8.0,   // scatter beyond the noise's
     "lnamp_mean": -0.018, "thermal_lnamp_mean": -0.001},   // a little decorrelation
    ...
  ],
  "slow_phase": {"path_um_1km": 93.8, "slope": 0.5, "bins": 9},
  "fast_phase": {"path_um_1km": 56.9, "slope": 0.05, "bins": 9},
  "gain_amp_rms": 0.090,
  "baseline_amp_rms": 0.0,
  "baseline_phase_rms_deg": 6.53
}
```

### Step 3: build a candidate (`noise aggregate`)

```
almasim noise aggregate --out noise/ --from alma
```

It reads every `window.json` and `fit.json` under `noise/windows/` and writes
`noise/candidate.json`: `alma` (its newest version) with every value the
windows measure replaced by their median. It prints, for each value, the
median, the 16th-84th percentile range and how many windows it comes from,
and records the same in the candidate's `provenance.values`. A value fewer
than 3 windows measure (`--min-windows`) keeps `alma`'s, and
`provenance.notes` says so. The candidate is named `alma-candidate`, version
`alma`'s + 1 (`--name`, `--version` to choose).

### Step 4: try the candidate

The candidate is an ordinary model file:

```
almasim noise show noise/candidate.json
almasim simulation --metadata-path rows.csv --row-idx 0 --output-dir try/ \
    --noise-model noise/candidate.json
```

You can also edit it by hand (the schema checks every change) and try that.

### Step 5: validate it against the real windows (`noise validate`)

```
almasim noise validate --model noise/candidate.json --windows windows.csv \
    --metadata rows.csv --measured noise/ --out noise/ --backend slurm \
    --wsclean-bin /path/to/wsclean --casa-data-root /path/to/casa-data
```

For each window, ALMASim simulates it from its archive row with the
candidate (a point source at the window's frequency, with its channels, up
to 960 of them, and their averaging as its MS gives them, one execution
block), measures the simulated MS with `noise extract`, images it as the
real windows were imaged (`image compute-parameters`, `image image-from-ms
--cube --uv-contsub`, `image clean-quality`), and measures the residuals of
both images: their noise level, skew, kurtosis and 5-sigma tails, the area
of their noise's autocorrelation in beams, and in the cube the spread of the
channel noise and the correlation of neighbouring channels. All of it goes
under `noise/validation/<the model's sha256, 12 characters>/`, so validations
of different models never mix. At the end, `noise compare` writes
`comparison.json` and prints a table: per metric, the median of the
per-window ratio (simulated over real, for noise levels and correlation
areas) or difference (for shapes: kurtosis, channel spread, neighbour
correlation), its range, a paired Wilcoxon signed-rank p (is the difference
real, or chance?) and how many windows fall within 10%; then each criterion,
passed or failed, and the verdict.

The default criteria:

| Check | Passes when |
| --- | --- |
| continuum, cube and visibility noise | each median ratio is within 5% of 1 |
| simulated visibilities against their WEIGHT | the median `scale` is within 1% of 1 |
| those three noise levels | none differs at p below 0.01 (p needs six or more windows) |
| each band and array group | its median continuum noise ratio is within 10% of 1 (groups of 3 or more windows) |

This is `alma@1` validated on four real windows (band 3 and band 6 12 m, a
band 6 7 m one), 16 minutes in all:

```
alma@1: 4 window(s) compared
  cont.sigma                   ratio      +0.9965 (+0.912..+1.16), 2 within 10%
  cube.sigma_median            ratio      +1.026 (+0.892..+1.14), 1 within 10%
  visibility.unit_jy           ratio      +1.074 (+1.01..+1.28), 3 within 10%
  cont.corr_area_beams         ratio      +1.033 (+0.959..+1.13), 3 within 10%
  cube.kurtosis                difference +0.03 (-0.00686..+0.0794)
  cube.adjacent_corr           difference -0.01894 (-0.0558..-0.0162)
  antenna_shape.excess_rms     difference -0.02256 (-0.0284..-0.0185)
  ...
  [pass] cont.sigma median within 5%: 0.9965
  [pass] cube.sigma_median median within 5%: 1.026
  [FAIL] visibility.unit_jy median within 5%: 1.074
  [pass] simulated variance x WEIGHT within 1%: 0.9999
NOT PASSED
```

Read it as: the simulated images are as noisy as the real ones (continuum
1.00, cube 1.03 in the median), the simulated visibilities carry exactly the
noise their WEIGHT says (0.9999), but one simulated visibility is 7% noisier
than a real one, mostly because of the 7 m window (1.47; the 12 m ones are
0.96 to 1.08), so `alma@1` does not pass. With four windows no Wilcoxon p is
computed (it needs six); a real validation uses tens. The shapes show where
to look next: the simulated antennas' own noise ripple is weaker than the
real one (`antenna_shape.excess_rms` 0.023 lower), which a candidate built
by `noise aggregate` from these windows would raise.

Write other criteria to a JSON file and pass `--criteria file.json`. Its
keys are `noise_tolerance` (0.05), `weight_tolerance` (0.01), `min_p`
(0.01), `group_tolerance` (0.10) and `checked` (the metrics whose p is
checked: the three noise levels), for example
`{"noise_tolerance": 0.1, "checked": []}`. `noise compare` alone re-judges
what `noise validate` produced.

### Step 6: release it (`noise promote`)

```
almasim noise promote noise/candidate.json \
    --comparison noise/validation/<sha>/comparison.json
```

It refuses unless that comparison is of this very model (same SHA-256) and
passed. It writes `src/almasim/services/noise_model/models/alma-2.json` in
your ALMASim checkout (`--models-dir` to write elsewhere), the version being
the newest released one plus 1. You then open a pull request that adds it:
`alma` then means version 2, `alma@1` stays available, the tests that pin
`alma`'s values are updated, and if the default model changes what runs
produce, the pull request is labelled `golden:supersede` with a new golden
baseline. The golden simulation runs one case through
`--noise-model alma@1`, so the noise components' code path is under the
gate.

With `--site` it goes to the site's models folder instead (see
[the team's own models](#the-teams-own-models)), under a name of its own
(`--name zwicky-alma`; `alma` is refused), and no pull request follows:
simulations can name it as soon as it is written. That is what the
console's *Release…* runs; the console never writes into the installed
package.

### All of it as one chain of jobs

```python
from almasim.jobs.pipeline import noise_model_from_windows, submit_pipeline
from almasim.jobs.profile import load_profile

steps = noise_model_from_windows(
    "noise/", "windows.csv", "rows.csv", validate=["--wsclean-bin", "/path/to/wsclean"]
)
submit_pipeline(steps, load_profile(), title="alma from cycle 11")
```

Extract, fit, aggregate and validate run as ALMASim jobs (Slurm jobs under
a cluster profile), each starting when the one before it succeeds; the
`extract`, `fit` and `validate` arguments add options to those steps.
Promotion stays a person's decision.

The console's *Noise models* page runs this chain for an imaging run, from
`noise windows` to `noise validate`, and shows the candidate and the
verdict ([console](console.md#noise-models)). Each step there takes
`--allow-partial`: a window that fails is listed in the step's failures
file and left out, and the step fails only when every window does.

## Adding a component

Suppose real data showed a slow drift of each antenna's amplitude through
the observation, and you want ALMASim to simulate it.

1. **Describe it** in `src/almasim/services/noise_model/components.py`: a
   class with its parameters (a bound and a description each) and
   `engine_fields`, which says what it sets on the simulator's
   `CorruptionModel`:

   ```python
   class AmplitudeDrift(NoiseComponent):
       """Each antenna's amplitude drifting through the observation."""

       rms: float = Field(ge=0, le=1, description="Fractional rms of the drift.")

       def engine_fields(self) -> dict[str, Any]:
           return {"amplitude_drift_rms": self.rms}
   ```

2. **Name it** in the same file, as a field of `Components`:
   `amplitude_drift: AmplitudeDrift | None = None`.
3. **Simulate it**: add `amplitude_drift_rms: float = 0.0` to
   `CorruptionModel` (`interferometry/corruptions.py`) and apply it where the
   antenna errors are drawn (`draw_antenna_corruptions`), from a random
   stream of its own spawned after the existing ones (`rng.spawn`), so it
   does not move the other components' draws.
4. **Measure it** (optional, but how a model gets its value): a statistic in
   `noise extract` or `noise fit`, and its reduction in `noise aggregate`.
5. Run `scripts/generate_noise_model_schema.py` and
   `scripts/generate_cli_reference.py`.

The schema, the checking of model files, `--noise-only` and
`--noise-without`, `noise show` and `noise components` follow from step 2. A
test checks that every component sets only `CorruptionModel` fields, and
none another component sets.


## When something goes wrong

| Message | Meaning, and what to do |
| --- | --- |
| `not a released model` / `no released noise model 'x'` | the name is not released and no such file exists; `almasim noise show` lists the released ones |
| `noise model 'alma' has no version n` | that version was never released; the message lists the ones that were |
| `unknown noise component(s) ...` | a misspelt component in `--noise-only`/`--noise-without`; `almasim noise components` lists them |
| `... has no component(s) ...` | the model does not hold that component (it was left out of its file) |
| `run noise extract first` (noise fit) | `fit` needs the window's `window.json`, made by `extract` for the same MS, spw and field |
| `too faint to fit` in `fit.json` | fewer than 20 bright (baseline, scan) groups: the window cannot show calibration residuals; its values come from other windows |
| `the data rise above the model` in `fit.json` | the clean model does not match the source (often extended emission it lacks): no fast phase from this window |
| `kept the base's` in a candidate's notes | fewer than `--min-windows` windows measured that value |
| `... did not pass its validation` (promote) | the comparison's failed checks are listed; fix the model, validate again |
| `the comparison is of ...` (promote) | the comparison was made for another model (or the candidate changed since); validate this one |
| `'alma' is a model ALMASim releases` (promote --site) | a team model takes a name of its own: `--name zwicky-alma` |
| `no site models folder` (promote --site) | set the profile's `[tools] noise_models` (or `ALMASIM_NOISE_MODELS`); `almasim update-shared` makes the shared install's |


## Where the code is

| What | Where |
| --- | --- |
| model files, schema, selection | `src/almasim/services/noise_model/` |
| the simulator's corruptions, thermal noise and Tsys | `src/almasim/services/interferometry/corruptions.py`, `noise.py`, `atmosphere.py` |
| ALMA's radiometer data, the atmosphere table | `src/almasim/services/interferometry/data/alma_radiometer.json`, `atm_chajnantor.npz` |
| ALMA's channels (averaging, correlation, noise bandwidth) | `src/almasim/services/instrument/channels.py` |
| extract, fit, aggregate, validate, compare, promote | `src/almasim/services/noise/` |
| the commands | `src/almasim/cli/noise.py` (`almasim noise ...`) |
