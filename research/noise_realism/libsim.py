"""ALMASim's own simulations of the MADIV-16 windows, with the corruptions the
counterparts measured (the "alma" preset), to compare with the real data.

    python libsim.py rows OUT.csv              each window's archive row (cycle11.csv,
                                               by its OUS), its Freq moved to the window
    python libsim.py run I ROWS.csv OUT_DIR [SCALES.csv]
                                               window I's simulation, written as an MS, its
                                               model cube times the window's scale
    python libsim.py run I ROWS.csv OUT_DIR - SKY
                                               window I observed on a sky made from the
                                               real source (SKY: point, gaussian, mixture)
    python libsim.py scales PASS1_IMG OUT.csv  each window's scale: the real continuum S/N
                                               over the first pass's (iter-f0-control)

The source is scaled to the real window's continuum S/N, so that the images
compare: a first pass at the library's own flux (continuum imaging only) gives
the S/N the model reaches, and the second pass multiplies the model cube by
the real S/N over it. The radiometer noise does not depend on the flux, so the
S/N scales with it.

The run is what `almasim simulation run` does with that row (the redshift of
its catalogue line, a point source, the row's array and integration time),
with the MS written (the CLI leaves it out) and the window's own channel count
(at most MAX_CHANNELS).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window import name, subset  # noqa: E402

META = Path("/SKAO_data/cycle11/metadata")
NOISE = Path("/SKAO_data/cycle11/MADIV-28/noise")
BASE = NOISE / "base"
LIBRARY = Path(os.environ.get("ALMASIM_SRC", "/home1/mdelliveneri/ALMASim-atm/src/almasim"))
MAX_CHANNELS = 960


def rows(out: Path) -> None:
    """One archive row per window: its OUS's row, Freq at the window's centre."""
    from casacore.tables import table

    meta = pd.read_csv(META / "cycle11.csv")
    products = pd.read_csv(META / "resolved_cycle11_products.csv", usecols=["uid", "filename"])
    out_rows = []
    for i, row in subset().iterrows():
        eb = Path(row.ms).name.split(".ms")[0].removeprefix("uid___")
        ous = products[products.filename.str.contains(eb, regex=False)].uid.unique()
        match = meta[meta.member_ous_uid.isin(ous)]
        if match.empty:
            print(f"{i}: no archive row for {eb}")
            continue
        with table(str(BASE / f"{name(row)}.ms" / "SPECTRAL_WINDOW"), ack=False) as t:
            chan = np.asarray(t.getcell("CHAN_FREQ", 0))
        record = match.iloc[0].to_dict()
        record.update(
            window=int(i),
            window_name=name(row),
            Freq=float(np.mean(chan)) / 1e9,
            n_channels=int(min(len(chan), MAX_CHANNELS)),
        )
        out_rows.append(record)
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f"{len(out_rows)} windows -> {out}")


def scales(pass1: Path, out: Path) -> None:
    """Real continuum S/N (iter-f0-control) over pass 1's, per window."""
    import json

    names = dict(line.split() for line in (NOISE / "diag" / "index-names.txt").read_text().splitlines())
    rows = []
    for i, window_name in names.items():
        quality = pass1 / f"w{i}" / "cubes" / f"w{i}_{i}" / "SPW-0_FIELD-0" / "quality.json"
        real = NOISE / "iter-f0-control" / "metrics" / f"{window_name}.json"
        if not quality.exists() or not real.exists():
            print(f"{i}: missing {'pass 1' if not quality.exists() else 'real'}")
            continue
        sim_snr = json.loads(quality.read_text())["continuum"]["snr"]
        real_snr = json.loads(real.read_text())["q.continuum.snr"]
        rows.append({"window": int(i), "name": window_name, "real_snr": real_snr,
                     "pass1_snr": sim_snr, "scale": real_snr / sim_snr})
    pd.DataFrame(rows).sort_values("window").to_csv(out, index=False)
    print(f"{len(rows)} scales -> {out}")


SKIES = ("point", "gaussian", "mixture")
MAX_COMPONENTS = 8
MIN_SHARE = 0.02  # clean components above this share of the brightest set the shape


def _real_models(window_name: str):
    """The real window's WSClean models (MADIV-16's imaging, as the counterparts
    take them): the continuum where the real image has emission
    (counterpart._emission_mask), the line cube after uvcontsub inside its line
    mask (Jy/pixel per channel), the cube's channel frequencies (Hz) and the
    pixel size (arcsec). Outside the masks the models are mostly noise peaks."""
    from astropy.io import fits

    from counterpart import _emission_mask
    from window import ROOT, WINDOWS

    windows = pd.read_csv(WINDOWS)
    row = next(r for r in windows.itertuples() if name(r) == window_name)
    task = Path(ROOT) / "cubes" / row.task
    with fits.open(task / "wsclean-model.fits") as hdul:
        cont = np.squeeze(np.nan_to_num(np.asarray(hdul[0].data, dtype=float)))
        cell = abs(float(hdul[0].header["CDELT1"])) * 3600.0
    cont = cont * _emission_mask(task)
    with fits.open(task / "wsclean-cube-model.fits") as hdul:
        line = np.nan_to_num(np.asarray(hdul[0].data, dtype=float))
        h = hdul[0].header
    while line.ndim > 3:
        line = line[0] if line.shape[0] == 1 else line[:, 0]
    if (task / "wsclean-cube-linemask.fits").exists():
        with fits.open(task / "wsclean-cube-linemask.fits") as hdul:
            mask = np.asarray(hdul[0].data)
        while mask.ndim > 3:
            mask = mask[0] if mask.shape[0] == 1 else mask[:, 0]
        if mask.shape == line.shape:
            line = line * (mask > 0)
    nchan = line.shape[0]
    freq = float(h["CRVAL3"]) + (np.arange(nchan) + 1 - float(h["CRPIX3"])) * float(h["CDELT3"])
    return cont, line, freq, cell


def _shape(image: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Flux-weighted centroid (arcsec from the image centre, x and y as in the
    FITS pixel axes) and covariance (arcsec^2) of an image's clean components."""
    positive = np.where(image > MIN_SHARE * image.max(), image, 0.0)
    y, x = np.indices(image.shape)
    w = positive / positive.sum()
    centre = np.array([image.shape[1] // 2, image.shape[0] // 2])
    mean = np.array([(w * x).sum(), (w * y).sum()])
    dx, dy = x - mean[0], y - mean[1]
    cov = np.array([[(w * dx * dx).sum(), (w * dx * dy).sum()], [(w * dx * dy).sum(), (w * dy * dy).sum()]])
    return (mean - centre) * cell, cov * cell**2


def _mixture(image: np.ndarray, cell: float, max_components: int = MAX_COMPONENTS):
    """A Gaussian mixture fitted to an image's clean components (each weighted by
    its flux, EM), its number of components chosen by BIC: means (arcsec from
    the image centre) and covariances (arcsec^2), with each component's flux share."""
    keep = image > MIN_SHARE * image.max()
    y, x = np.nonzero(keep)
    w = image[keep] / image[keep].sum()
    pts = np.stack([x, y], axis=1).astype(float)
    n = max(len(pts), 1)
    rng = np.random.default_rng(0)
    best = None
    for k in range(1, min(max_components, len(pts)) + 1):
        means = pts[rng.choice(len(pts), size=k, replace=False, p=w)]
        covs = np.array([np.eye(2) * max(np.average(((pts - pts.mean(0)) ** 2).sum(1), weights=w) / 2, 1.0)] * k)
        pis = np.full(k, 1.0 / k)
        for _ in range(200):
            dens = np.empty((len(pts), k))
            for j in range(k):
                d = pts - means[j]
                inv = np.linalg.inv(covs[j])
                dens[:, j] = pis[j] * np.exp(-0.5 * np.einsum("ni,ij,nj->n", d, inv, d)) / (
                    2 * np.pi * np.sqrt(np.linalg.det(covs[j])))
            total = dens.sum(1, keepdims=True) + 1e-300
            resp = dens / total * w[:, None]
            nk = resp.sum(0) + 1e-12
            means = resp.T @ pts / nk[:, None]
            for j in range(k):
                d = pts - means[j]
                covs[j] = (resp[:, j, None] * d).T @ d / nk[j] + np.eye(2) * 0.25
            pis = nk / nk.sum()
        loglike = float(n * (w * np.log(total[:, 0])).sum())
        bic = -2 * loglike + (6 * k - 1) * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, means.copy(), covs.copy(), pis.copy())
    _, means, covs, pis = best
    centre = np.array([image.shape[1] // 2, image.shape[0] // 2])
    return (means - centre) * cell, covs * cell**2, pis


def _half_flux(image: np.ndarray, cell: float, offset: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """The Gaussian of the components' axis ratio and orientation (their second
    moments) whose half-flux ellipse holds half their flux, as theirs does:
    second moments alone follow faint wings and make a compact source broad."""
    positive = np.where(image > MIN_SHARE * image.max(), image, 0.0)
    y, x = np.nonzero(positive)
    flux = positive[y, x]
    centre = np.array([image.shape[1] // 2, image.shape[0] // 2]) + offset / cell
    evals, evecs = np.linalg.eigh(cov / cell**2 + np.eye(2) * 1e-6)
    axes = np.sqrt(evals / evals.max())  # minor, major (major = 1)
    uv = (np.stack([x - centre[0], y - centre[1]], axis=1) @ evecs) / axes
    radius = np.hypot(uv[:, 0], uv[:, 1])
    order = np.argsort(radius)
    cumulative = np.cumsum(flux[order]) / flux.sum()
    r50 = float(radius[order][np.searchsorted(cumulative, 0.5)])
    sigma = max(r50, 0.5) / np.sqrt(2 * np.log(2))  # pixels, along the major axis
    return (evecs @ np.diag((sigma * axes) ** 2) @ evecs.T) * cell**2


def real_sky(window_name: str, kind: str, n_pix: int, cell: float, freqs_hz: np.ndarray, seed: int):
    """A sky closest to the real source: its continuum and line flux per channel
    (the real WSClean models, in Jy), spread as a point at the source's
    centroid, a Gaussian of its size and orientation (the second moments of its
    clean components), or a Gaussian mixture fitted to the clean components.
    Returns the cube (channel, y, x; Jy/pixel) and each channel's continuum share."""
    cont, line, freq, real_cell = _real_models(window_name)
    continuum = float(cont.sum())
    spectrum = line.sum(axis=(1, 2))
    nearest = np.abs(freq[None, :] - np.asarray(freqs_hz)[:, None]).argmin(axis=1)
    inside = (freqs_hz >= freq.min() - abs(freq[1] - freq[0])) & (freqs_hz <= freq.max() + abs(freq[1] - freq[0]))
    line_flux = np.clip(np.where(inside, spectrum[nearest], 0.0), 0.0, None)
    shape_from = cont if continuum > 0 and cont.max() > 0 else np.clip(line, 0, None).sum(axis=0)
    if shape_from.max() <= 0:  # no emission: the sky is empty, as the real one looks
        print(f"sky {kind}: no emission in the real models; empty sky")
        return np.zeros((len(freqs_hz), n_pix, n_pix)), np.ones(len(freqs_hz))
    offset, cov = _shape(shape_from, real_cell)
    cov = _half_flux(shape_from, real_cell, offset, cov)
    centre = np.array([n_pix // 2, n_pix // 2]) + offset / cell
    cov = cov / cell**2 + np.eye(2) * 0.25  # at least half a cell
    y, x = np.indices((n_pix, n_pix), dtype=float)
    d = np.stack([x - centre[0], y - centre[1]], axis=-1)
    inv = np.linalg.inv(cov)
    envelope = np.exp(-0.5 * np.einsum("...i,ij,...j->...", d, inv, d))
    if kind == "point":
        template = np.zeros((n_pix, n_pix))
        x0, y0 = np.floor(centre).astype(int)
        fx, fy = centre - np.floor(centre)
        for (xi, yi), share in (((x0, y0), (1 - fx) * (1 - fy)), ((x0 + 1, y0), fx * (1 - fy)),
                                ((x0, y0 + 1), (1 - fx) * fy), ((x0 + 1, y0 + 1), fx * fy)):
            if 0 <= xi < n_pix and 0 <= yi < n_pix:
                template[yi, xi] += share
    elif kind == "gaussian":
        template = envelope
    elif kind == "mixture":
        means, covs, pis = _mixture(shape_from, real_cell)
        template = np.zeros((n_pix, n_pix))
        for mean, comp, pi in zip(means, covs, pis):
            c = np.array([n_pix // 2, n_pix // 2]) + mean / cell
            dd = np.stack([x - c[0], y - c[1]], axis=-1)
            ci = comp / cell**2 + np.eye(2) * 0.25
            g = np.exp(-0.5 * np.einsum("...i,ij,...j->...", dd, np.linalg.inv(ci), dd))
            if g.sum() > 0:  # a component wholly off this grid adds nothing
                template += pi * g / g.sum()
        print(f"mixture: {len(pis)} components")
    else:
        raise ValueError(f"sky {kind!r}: one of {SKIES}")
    if not template.sum() > 0:  # the source lies off this grid: at its edge instead
        template = np.zeros((n_pix, n_pix))
        xe, ye = np.clip(np.round(centre).astype(int), 0, n_pix - 1)
        template[ye, xe] = 1.0
    template = template / template.sum()
    total = continuum + line_flux
    share = np.clip(np.where(total > 0, continuum / np.where(total > 0, total, 1.0), 1.0), 0.0, 1.0)
    print(f"sky {kind}: continuum {continuum:.4g} Jy, line peak {line_flux.max():.4g} Jy, "
          f"offset {offset[0]:.2f}\", {offset[1]:.2f}\", size (sqrt eig) "
          f"{np.sqrt(np.clip(np.linalg.eigvalsh(cov), 0, None)) * cell}\"")
    return template[None, :, :] * total[:, None, None], share


def run(i: int, rows_csv: Path, out_dir: Path, scales_csv: Path | None = None, sky: str | None = None) -> None:
    from almasim.services import astro
    from almasim.services.astro.spectral import with_redshifts
    from almasim.services.compute import create_backend
    from almasim.services.simulation import (
        SimulationParams,
        export_results,
        generate_clean_cube,
        simulate_observation,
    )

    table = pd.read_csv(rows_csv)
    picked = table[table.window == i]
    if picked.empty:
        raise SystemExit(f"window {i} has no row in {rows_csv}")
    rest_frequency, _ = astro.get_line_info(LIBRARY)
    row = with_redshifts(picked.iloc[[0]].reset_index(drop=True), rest_frequency).iloc[0]
    run_dir = out_dir / f"w{i}"
    run_dir.mkdir(parents=True, exist_ok=True)
    params = SimulationParams.from_metadata_row(
        row,
        idx=i,
        main_dir=LIBRARY,
        output_dir=run_dir,
        tng_dir=run_dir / "tng",
        galaxy_zoo_dir=run_dir / "galaxy_zoo",
        hubble_dir=run_dir / "hubble",
        project_name=f"w{i}",
        source_type="point",
        save_mode="fits",
        persist=True,
        ml_dataset_path=None,
        n_pix=256,
        n_channels=int(row.n_channels),
        seed=1000 + i + (1000 * (1 + SKIES.index(sky)) if sky else 0),
        corruptions="alma",
        ms_export=True,
        ms_save_mode="msv2",
    )
    with create_backend("sync") as backend:
        clean = generate_clean_cube(params, logger=print, compute_backend=backend)
        if sky is not None:
            freqs = np.asarray(clean.channel_frequencies_hz, dtype=float)
            cube, share = real_sky(str(row.window_name), sky, clean.model_cube.shape[-1],
                                   float(clean.cell_size_arcsec), freqs, seed=7000 + i)
            cubes = {id(clean.model_cube): clean.model_cube}
            cubes.update({id(r.skymodel): r.skymodel for r in clean.interferometer_runs})
            for target in cubes.values():
                target[...] = cube.reshape(target.shape)
            clean.continuum_fraction = share
        if scales_csv is not None:
            scale = float(pd.read_csv(scales_csv).set_index("window").loc[i, "scale"])
            print(f"model cube x {scale:.4g} (real continuum S/N)")
            cubes = {id(clean.model_cube): clean.model_cube}
            cubes.update({id(r.skymodel): r.skymodel for r in clean.interferometer_runs})
            cubes.update({id(r.skymodel): r.skymodel for r in getattr(clean, "total_power_runs", [])
                          if hasattr(r, "skymodel")})
            for cube in cubes.values():
                cube *= scale
        results = simulate_observation(clean, compute_backend=backend)
        exported = export_results(params, clean, results, logger=print)
    print("MS", exported.get("ms_path"))


if __name__ == "__main__":
    if sys.argv[1] == "rows":
        rows(Path(sys.argv[2]))
    elif sys.argv[1] == "scales":
        scales(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        scales_arg = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
        run(int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]),
            Path(scales_arg) if scales_arg else None, sys.argv[6] if len(sys.argv) > 6 else None)
