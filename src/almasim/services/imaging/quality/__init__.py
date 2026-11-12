"""Cleaning-quality statistics and moment maps for imaged tasks.

Measures, from the images a WSClean task leaves (``wsclean-*.fits`` and
``wsclean-cube-*.fits``) and its log, how much source signal the clean left
in the residual: the statistics used to tune the MADIV-16 imaging (masked
clean, wider images, continuum restart, visibility-plane continuum
subtraction, self-calibration, the cube line mask).

Continuum, around the brightest pixel of the restored image (the source):

* ``snr``: its peak over the residual noise (1.4826 x MAD, more than 2.5
  beams from the source);
* ``residual_share_1beam`` / ``_2beam``: the residual summed within one or
  two beams of the source over the restored image summed there, the share of
  the source still in the residual;
* ``residual_peak_sigma`` and ``residual_core_sigma``: the largest absolute
  residual within one beam and the mean within half a beam, in noise units;
* ``residual_rings_sigma``: the residual's mean in rings of 0.5 beam out to
  5 beams. A positive core with a negative ring at 1-2 beams is flux the
  clean did not reach (the dirty beam's first sidelobe); a positive ring
  inside a negative one is a shape mismatch, not missing flux;
* ``residual_symmetric_sigma`` / ``residual_antisymmetric_sigma``: the
  residual near the source split under a 180-degree rotation; phase errors
  leave mostly antisymmetric structure;
* ``model_flux_jy`` and ``model_components``;
* ``noise_at_mask_creation_ratio``: the noise WSClean estimated in its first
  major cycle over the one in its last (from the log). Well above 1, the
  auto-mask was built while sidelobes inflated the noise and misses the
  source's fainter parts.
* ``diverged``: how many times WSClean stopped the continuum's clean for
  divergence (its total flux growing instead of converging), over every
  pass (from the log). Above 0, the restored image and residual carry the
  diverged model and are not a measure of the noise.

Cube, with the source's spectrum (or the channel ranges a visibility-plane
continuum subtraction logged) split into line and line-free channels:

* ``channel_noise_mjy``: median per-channel residual noise;
* ``linefree_residual_sigma`` / ``linefree_residual_share``: the residual
  stacked over the line-free channels at the source, in units of the stacked
  noise and as a share of the stacked image (continuum left in the cube);
* ``line_residual_share``, ``line_residual_mean_sigma``,
  ``line_residual_peak_sigma``: the residual stacked over the line channels
  inside the line mask (``wsclean-cube-linemask.fits``, or a moment-0 mask
  built the same way), as a share of the stacked image and in noise units;
* ``model_flux_jy``: the cube model's total;
* ``diverged``: how many channel cleans WSClean stopped for divergence, over
  every pass (from the log).

Moment maps (``quality/`` next to the images): moment 0 (Jy/beam km/s),
1 (km/s, intensity weighted over pixels above 3 channel sigma) and 8 (peak,
Jy/beam) of the line channels, and moment 0 of the residual over the line
and over the line-free channels, which show uncleaned emission directly.

The statistics live in ``continuum`` and ``cube``, the per-task files and the
CSV summary in ``report`` (MADIV-28 T4.10).
"""

from .continuum import RINGS_BEAMS, continuum_quality, divergences, noise_history
from .cube import QUALITY_DIRNAME, cube_quality
from .report import (
    QUALITY_FILENAME,
    SUMMARY_COLUMNS,
    SUMMARY_FILENAME,
    assess_task,
    find_task_dirs,
    summary_rows,
    write_summary,
)

__all__ = [
    "QUALITY_DIRNAME",
    "QUALITY_FILENAME",
    "RINGS_BEAMS",
    "SUMMARY_COLUMNS",
    "SUMMARY_FILENAME",
    "assess_task",
    "continuum_quality",
    "cube_quality",
    "divergences",
    "find_task_dirs",
    "noise_history",
    "summary_rows",
    "write_summary",
]
