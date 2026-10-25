"""The report's numbers and pictures: noise statistics of the simulated counterparts
against the real windows, and real/simulated line images side by side.

    python report.py OUT_DIR

Writes OUT_DIR/report.json (the rows the charts draw) and OUT_DIR/pairs/*.png.
Only the latest model: iter-f2-sim for the faint windows (noise measured across
the whole track, the channel kernel and profile, line and continuum models masked
to the real emission) and iter-f2-bright for the bright ones (the same, plus each
window's fitted atmosphere and antenna amplitudes), with iter-f3-sim and
iter-f3-bright in their place for the 19 windows whose channel filter used to fall
back to white noise, and iter-f4-bright for the 13 bright windows with fitted
per-(baseline, scan) errors. The visibilities: uvfinal5/ (each antenna's and
hand's own noise shape across the window), uvfinal9/ for those 13 bright windows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import METRICS, compare, load  # noqa: E402

NOISE = Path("/SKAO_data/cycle11/MADIV-28/noise")
MADIV16 = Path("/SKAO_data/cycle11/MADIV-16/cubes")
C_KMS = 299_792.458

# The noise metrics the report is about, with the words it uses for them.
NOISE_METRICS = {
    "q.continuum.noise_jy_beam": "Continuum noise",
    "cont.kurtosis": "Continuum kurtosis",
    "cont.corr_area_beams": "Continuum noise correlation area",
    "q.continuum.noise_at_mask_creation_ratio": "Continuum noise at mask creation",
    "q.continuum.residual_peak_sigma": "Continuum residual peak",
    "q.cube.channel_noise_mjy": "Cube channel noise",
    "cube.sigma_spread": "Cube noise spread across channels",
    "cube.adjacent_corr": "Cube adjacent-channel correlation",
    "cube.corr_area_beams": "Cube noise correlation area",
    "cube.kurtosis": "Cube kurtosis",
    "q.cube.linefree_residual_sigma": "Cube line-free residual",
}


# Simulations whose cube cleaning diverged (cube noise 5-7x the real, model flux
# thousands of Jy): both windows with very large, extended real line models.
DIVERGED = {"uid___A002_X110ed3d_X312d_spw22", "uid___A002_X1104ab5_X12cf9_spw20"}


def faint_windows() -> set[str]:
    """Windows whose continuum S/N is under 20 (diag/lists.log, from window.py)."""
    lists = (NOISE / "diag" / "lists.log").read_text().splitlines()
    faint = set(lists[1].split()[1:])
    return {line.split()[2] for line in lists if line.startswith("row ") and line.split()[1] in faint}


# Per class, the iterations of the latest model; a later one replaces an earlier
# one's windows (iter-f3 reran the windows whose channel filter fell back to white).
LATEST = {"faint": ("iter-f2-sim", "iter-f3-sim"), "bright": ("iter-f2-bright", "iter-f3-bright", "iter-f4-bright")}


def latest_metrics() -> dict[str, dict[str, dict]]:
    """Per class (faint, bright), the latest counterparts' metrics by window."""
    faint = faint_windows()
    out = {}
    for label, its in LATEST.items():
        m = {w: v for it in its for w, v in load(NOISE / it).items()}
        out[label] = {w: v for w, v in m.items() if w not in DIVERGED and (w in faint) == (label == "faint")}
    return out


def metric_rows(control: dict, sims: dict[str, dict]) -> list[dict]:
    rows = []
    for label, sim in sims.items():
        if len(set(control) & set(sim)) < 6:
            continue
        result = compare(control, sim)
        for metric, words in NOISE_METRICS.items():
            r = result["metrics"].get(metric)
            if r is None:
                continue
            values = []
            for w in sorted(set(control) & set(sim)):
                a, b = control[w].get(metric), sim[w].get(metric)
                if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
                    continue
                if r["kind"] == "log":
                    if a > 0 and b > 0:
                        values.append(100 * (b / a - 1))
                else:
                    values.append(b - a)
            v = np.array(values)
            rows.append({"windows": label, "metric": words, "unit": "%" if r["kind"] == "log" else "abs",
                         "n": int(len(v)), "median": round(float(np.median(v)), 4),
                         "lo": round(float(np.percentile(v, 16)), 4), "hi": round(float(np.percentile(v, 84)), 4),
                         "p": None if r["p"] is None else round(r["p"], 4)})
    return rows


def scatter_rows(control: dict, sims: dict[str, dict]) -> list[dict]:
    rows = []
    for label, sim in sims.items():
        for w in sorted(set(control) & set(sim)):
            for metric, key, scale in (("cube", "q.cube.channel_noise_mjy", 1.0),
                                       ("continuum", "q.continuum.noise_jy_beam", 1e3)):  # mJy/beam
                a, b = control[w].get(key), sim[w].get(key)
                if a and b and np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                    rows.append({"window": w, "kind": metric, "windows": label, "real": a * scale, "sim": b * scale})
    return rows


UV_STATS = (
    ("variance", "Variance against WEIGHT"),
    ("kurtosis", "Excess kurtosis"),
    ("channel_variance_spread", "Noise spread across channels"),
    ("antenna_variance_spread", "Noise spread across antennas"),
    ("time_lag1_corr", "Correlation of consecutive integrations"),
)


def _uv_results() -> list[dict]:
    """Each window's real visibilities against its latest counterpart: uvfinal5/ (with
    each antenna's noise shape), uvfinal9/ for the bright windows with per-baseline
    errors; uvfinal/ and uvfinal3/ where those are missing."""
    files = {f.stem: f for d in ("uvfinal", "uvfinal3", "uvfinal5", "uvfinal9") for f in (NOISE / d).glob("*.json")}
    out = []
    for f in sorted(files.values(), key=lambda p: int(p.stem)):
        r = json.loads(f.read_text())
        label = next((k for k in r if isinstance(r[k], dict) and "variance" in r[k] and k != "real"), None)
        if label:
            out.append((r, label))
    return out


def uv_rows() -> list[dict]:
    rows = []
    faint = faint_windows()
    for r, label in _uv_results():
        for key, words in UV_STATS:
            a, b = r["real"].get(key), r[label].get(key)
            if a is not None and b is not None:
                rows.append({"window": r["window"], "source": r["source"], "ghz": round(r["ghz"], 1),
                             "windows": "faint" if r["window"] in faint else "bright",
                             "stat": words, "real": round(a, 5), "sim": round(b, 5)})
    return rows


def baseline_rows(count: int = 2) -> list[dict]:
    """Phase scatter and amplitude loss by baseline, real and latest, for the bright
    windows with the most bright baseline-scans."""
    faint = faint_windows()
    bright = [(sum(r["real"]["group_n"]), r, label) for r, label in _uv_results() if r["window"] not in faint]
    rows = []
    for _, r, label in sorted(bright, key=lambda x: -x[0])[:count]:
        edges = np.asarray(r["real"]["uv_edges_klambda"]) * 1e3
        lam = C_KMS * 1e3 / (r["ghz"] * 1e9)
        centres = np.sqrt(edges[:-1] * edges[1:]) * lam
        for data, key in (("Real", "real"), ("Simulated", label)):
            for b, ph, la, n in zip(centres, r[key]["group_phase_rms_deg"], r[key]["group_lnamp_mean"],
                                    r["real"]["group_n"], strict=True):
                if ph is None or n < 20:
                    continue
                rows.append({"source": r["source"], "data": data, "baseline_m": round(float(b)),
                             "phase_deg": round(ph, 2), "lnamp": round(la, 4)})
    return rows


def line_channels(task_log: Path) -> list[tuple[int, int]]:
    for text in task_log.read_text(errors="replace").splitlines():
        if "lines in channels" in text:
            spec = text.rsplit("lines in channels", 1)[1].strip()
            if spec == "none":
                return []
            return [(int(p.partition("~")[0]), int(p.partition("~")[2] or p.partition("~")[0]))
                    for p in spec.split(";")]
    return []


def cube_and_freq(path: Path):
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        h = hdul[0].header
    while data.ndim > 3:
        data = data[0]
    freq = h["CRVAL3"] + (np.arange(data.shape[0]) + 1 - h["CRPIX3"]) * h["CDELT3"]
    return np.nan_to_num(data), freq, h


def moment0(cube_dir: Path, ranges_hz: list[tuple[float, float]]):
    cube, freq, h = cube_and_freq(cube_dir / "wsclean-cube-image.fits")
    sel = np.zeros(len(freq), dtype=bool)
    half = abs(h["CDELT3"]) / 2
    for lo, hi in ranges_hz:
        sel |= (freq + half >= lo) & (freq - half <= hi)
    dv = C_KMS * abs(h["CDELT3"]) / np.mean(freq)
    return cube[sel].sum(axis=0) * dv, int(sel.sum()), h


def pairs(out: Path, control: dict, windows_csv: Path, count: int = 10) -> list[dict]:
    """Moment 0 over the real line channels, real and simulated, for the windows with the strongest lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from casacore.tables import table

    w = pd.read_csv(windows_csv)
    names = {f"{Path(t.split('/')[0]).name.removesuffix('.ms.split')}_spw{s}": t for t, s in zip(w.task, w.spw, strict=True)}
    from astropy.io import fits
    from scipy.ndimage import binary_dilation

    # Every window with a simulated cube, the latest model's where there are several.
    sim_dirs = {}
    for it in (it for its in LATEST.values() for it in its):
        for f in (NOISE / it / "cubes").glob("*/SPW-0_FIELD-0/wsclean-cube-image.fits"):
            if f.parent.parent.name not in DIVERGED:
                sim_dirs[f.parent.parent.name] = it
    scored = []
    for name in sorted(set(control) & set(sim_dirs)):
        task = MADIV16 / names[name]
        ranges = line_channels(task.parent / f"{task.name}.log")
        mask_path = task / "wsclean-cube-linemask.fits"
        real_dir = NOISE / "iter-f0-control" / "cubes" / name / "SPW-0_FIELD-0"
        if not ranges or not mask_path.exists() or not (real_dir / "wsclean-cube-image.fits").exists():
            continue
        with table(str(NOISE / "base" / f"{name}.ms" / "SPECTRAL_WINDOW"), ack=False) as t:
            chan = np.asarray(t.getcell("CHAN_FREQ", 0))
            width = np.abs(np.asarray(t.getcell("CHAN_WIDTH", 0)))
        hz = [(min(chan[a], chan[b]) - width[a] / 2, max(chan[a], chan[b]) + width[b] / 2)
              for a, b in ranges if b < len(chan)]
        if not hz:
            continue
        m_real, nsel, _ = moment0(real_dir, hz)
        line_mask = np.asarray(fits.getdata(mask_path), dtype=float)
        while line_mask.ndim > 2:
            line_mask = line_mask[0]
        if nsel == 0 or line_mask.shape != m_real.shape:
            continue
        away = ~binary_dilation(line_mask > 0, iterations=10)
        if away.sum() < 1000:
            continue
        rms = 1.4826 * np.median(np.abs(m_real[away] - np.median(m_real[away])))
        near = binary_dilation(line_mask > 0, iterations=2)
        peak = float(m_real[near].max() / rms) if rms > 0 and near.any() else 0.0
        if peak >= 8:  # a line the real data detect, inside the real line mask
            scored.append((peak, name, hz, m_real, nsel))
    scored.sort(reverse=True)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    meta = []
    for snr, name, hz, m_real, nsel in scored[:count]:
        sim_iter = sim_dirs[name]
        model = "bright" if sim_iter in LATEST["bright"] else "faint"
        m_sim, _, h = moment0(NOISE / sim_iter / "cubes" / name / "SPW-0_FIELD-0", hz)
        # Noise of each moment map away from the line: outside the real line mask,
        # widened by 10 pixels (extended emission would otherwise raise it).
        task = MADIV16 / names[name]
        line_mask = np.asarray(fits.getdata(task / "wsclean-cube-linemask.fits"), dtype=float)
        while line_mask.ndim > 2:
            line_mask = line_mask[0]
        away = ~binary_dilation(line_mask > 0, iterations=10)
        if away.shape != m_real.shape or away.sum() < 1000:
            away = np.ones(m_real.shape, dtype=bool)
        rms = 1.4826 * np.median(np.abs(m_real[away] - np.median(m_real[away])))
        # Crop round the real line emission: the mask's extent, at least 64 pixels.
        emission = m_real > 4 * rms
        ys, xs = np.nonzero(emission) if emission.any() else ([m_real.shape[0] // 2], [m_real.shape[1] // 2])
        cy, cx = int(np.median(ys)), int(np.median(xs))
        half = max(32, int(1.3 * max(np.ptp(ys), np.ptp(xs)) / 2) + 8)
        y0, y1 = max(cy - half, 0), min(cy + half, m_real.shape[0])
        x0, x1 = max(cx - half, 0), min(cx + half, m_real.shape[1])
        real_c, sim_c = m_real[y0:y1, x0:x1], m_sim[y0:y1, x0:x1]
        lo, hi = np.percentile(real_c, [1, 99.7])
        rms_sim = 1.4826 * np.median(np.abs(m_sim[away] - np.median(m_sim[away])))
        arcsec = abs(h["CDELT2"]) * 3600
        ext = [0, (x1 - x0) * arcsec, 0, (y1 - y0) * arcsec]
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), constrained_layout=True)
        for ax, img, title, noise in ((axes[0], real_c, "Real", rms), (axes[1], sim_c, "Simulated", rms_sim)):
            im = ax.imshow(img, origin="lower", cmap="inferno", vmin=lo, vmax=hi, extent=ext)
            ax.set_title(f"{title}  (peak {img.max() / noise:.0f}σ, noise off-line {noise * 1e3:.1f} mJy/beam km/s)", fontsize=9)
            ax.set_xlabel("arcsec")
        axes[0].set_ylabel("arcsec")
        fig.colorbar(im, ax=axes, shrink=0.85, label="Jy/beam km/s")
        source = w.source[[k for k, t in enumerate(w.task) if t == names[name]][0]]
        ghz = float(np.mean([np.mean(r) for r in hz]) / 1e9)
        fig.suptitle(f"{source}  {ghz:.1f} GHz  moment 0 over the real line channels ({nsel} cube channels)", fontsize=10)
        png = out / "pairs" / f"{name}.png"
        fig.savefig(png, dpi=110)
        plt.close(fig)
        meta.append({"window": name, "source": str(source), "ghz": round(ghz, 1), "model": model,
                     "iteration": sim_iter, "real_peak_sigma": round(float(m_real.max() / rms), 1),
                     "sim_peak_sigma": round(float(m_sim.max() / rms_sim), 1),
                     "real_rms": float(rms), "sim_rms": float(rms_sim), "channels": nsel, "png": str(png)})
    return meta


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    control = load(NOISE / "iter-f0-control")
    latest = latest_metrics()
    report = {
        "counts": {"real": len(control), **{k: len(set(v) & set(control)) for k, v in latest.items()},
                   "diverged": sorted(DIVERGED)},
        "metrics": metric_rows(control, latest),
        "scatter": scatter_rows(control, latest),
        "uv": uv_rows(),
        "baseline": baseline_rows(),
    }
    report["pairs"] = pairs(out, control, NOISE / "windows.csv")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["counts"]), len(report["metrics"]), len(report["scatter"]), len(report["uv"]),
          len(report["baseline"]), len(report["pairs"]))
