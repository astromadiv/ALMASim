# Noise

This page describes ALMASim’s current noise model.

## Location

Noise handling lives primarily in [`src/almasim/services/interferometry/noise.py`](../src/almasim/services/interferometry/noise.py).

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

`--corruptions alma` (or the `atmospheric_tsys` and `receiver_ripple_rms`
fields of a corruption model) turns this on. The sky comes from
[`atmosphere.py`](../src/almasim/services/interferometry/atmosphere.py), which
reads a table of CASA's ATM model for the ALMA site
(`scripts/generate_atmosphere_table.py` writes it; CASA is not needed to use
it). Tsys is referred above the atmosphere, (1 + g) e^(A tau) (Trx + Tsky +
g Tsky_image), with the image band one IF centre on the other side of the LO
and the sideband whose image gives the lower Tsys. The ripple is drawn from
the run's seed.

## Per Antenna and Per Baseline

Two more terms, also in `--corruptions alma`, come from the same 100 windows:

- Each antenna's noise has its own smooth ripple across the window, beyond the
  array's: about 2% of its noise variance, holding over about 120 MHz (the
  median of the 73 windows wider than 1 GHz; narrow windows show finer, ~1 MHz
  structure). A baseline's noise variance is the product of its two antennas'.
  WEIGHT does not follow it, so it shows as antennas whose noise differs from
  their weights and as heavier tails in the visibility noise
  (`antenna_ripple_rms`, `antenna_ripple_mhz`).
- On bright sources, each baseline keeps an amplitude and phase error per scan
  that does not factor into its antennas' gains, so self-calibration cannot
  remove it, and it limits the continuum's dynamic range: about 6% and 2.5 deg
  (the median of the 13 bright windows where it is measurable;
  `baseline_amp_rms`, `baseline_phase_rms_deg`).

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
