"""Measure each window's residual atmospheric phase from its visibilities.

    python fit_screen.py UV_DIR BASE_CONFIG.json OUT_CONFIG.json [NOISE_LABEL]
    python fit_screen.py gains UV_DIR CONFIG.json OUT_CONFIG.json LABEL
    python fit_screen.py baselines UV_DIR CONFIG.json OUT_CONFIG.json LABEL

From uvcompare's per-(baseline, scan) data over model: the phase scatter of
the real data beyond what thermal noise alone gives (the "thermal-profile"
simulation, same noise), by baseline length in metres, fitted with a power
law sigma(b) = sigma_1km (b / 1 km)^slope. The phase difference between two
antennas is 2 pi dL / lambda, so sigma_1km is a path difference in microns,
which is what the scan screen (scan_screen_path_um_1km) takes.

What varies within a scan does not show in the scan's average phase but
lowers its amplitude: |A| = exp(-sigma^2 / 2) for a phase of rms sigma about
the scan's mean. The mean ln|A| of the real data below the thermal one's
gives that sigma per bin, fitted the same way (fast_screen_path_um_1km). A
window whose data rise above the model (ln|A| > 0.05 in any bin) has a model
that does not match the source, and the loss there is not the atmosphere's:
it gets no fast screen. Windows without enough bright baseline-scans (three
bins) keep no screen.

"gains": the per-antenna, per-scan amplitude scatter of the real data beyond
that of the simulation LABEL (uvcompare's antenna_solve, compared on the
same baseline-scans) is the antenna gain amplitude error left by the
calibration (gain_amp_rms), added to the windows of CONFIG that have it.
Run it on a comparison of the real data with the screens already fitted:
the atmosphere's decorrelation also differs from antenna to antenna.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

C = 299_792_458.0


def _baselines_m(result: dict) -> np.ndarray:
    edges = np.asarray(result["real"]["uv_edges_klambda"]) * 1e3
    return np.sqrt(edges[:-1] * edges[1:]) * C / (result["ghz"] * 1e9)


def fit(result: dict, noise_label: str = "thermal-profile") -> dict | None:
    real, thermal = result["real"], result[noise_label]
    b, excess = [], []
    for bl, pr, pt, n in zip(
        _baselines_m(result), real["group_phase_rms_deg"], thermal["group_phase_rms_deg"], real["group_n"], strict=True
    ):
        if pr is None or pt is None or n < 20 or pr <= pt:
            continue
        b.append(bl)
        excess.append(np.radians(np.sqrt(pr**2 - pt**2)))
    return _power_law(b, excess, result["ghz"])


def fit_fast(result: dict, noise_label: str = "thermal-profile") -> dict | None:
    real, thermal = result["real"], result[noise_label]
    if any(a is not None and a > 0.05 for a in real["group_lnamp_mean"]):
        return None
    b, sigma = [], []
    for bl, ar, at, n in zip(
        _baselines_m(result), real["group_lnamp_mean"], thermal["group_lnamp_mean"], real["group_n"], strict=True
    ):
        if ar is None or at is None or n < 20 or ar >= at:
            continue
        b.append(bl)
        sigma.append(np.sqrt(2 * (at - ar)))
    return _power_law(b, sigma, result["ghz"])


def _power_law(b: list, sigma: list, ghz: float) -> dict | None:
    """sigma (rad) = sigma_1km (b / 1 km)^slope, by least squares in log-log."""
    if len(b) < 3:
        return None
    lam = C / (ghz * 1e9)
    x, y = np.log(np.asarray(b) / 1_000), np.log(sigma)
    slope, intercept = np.polyfit(x, y, 1)
    # The screens take slopes in (0.05, 0.95); outside them (an excess that does
    # not grow with baseline, or grows faster than turbulence can) the strength
    # is refitted at the nearest allowed slope, to match the measured baselines.
    if not 0.05 <= slope <= 0.95:
        slope = min(max(slope, 0.05), 0.95)
        intercept = float(np.mean(y - slope * x))
    sigma_1km = float(np.exp(intercept))
    return {
        "bins": len(b),
        "baseline_m": [round(float(x)) for x in b],
        "excess_deg": [round(float(np.degrees(x)), 2) for x in sigma],
        "slope": round(float(slope), 3),
        "phase_1km_deg": round(float(np.degrees(sigma_1km)), 2),
        "path_um_1km": round(sigma_1km * lam / (2 * np.pi) * 1e6, 1),
    }


def fit_gains(result: dict, label: str) -> float | None:
    real, sim = result["real"], result[label]
    if "ant_lnamp_rms" not in real or "ant_lnamp_rms" not in sim:
        return None
    excess = real["ant_lnamp_rms"] ** 2 - sim["ant_lnamp_rms"] ** 2
    return round(float(np.sqrt(excess)), 4) if excess > 0 else None


def gains_main(uv_dir: Path, config_path: Path, out: Path, label: str) -> None:
    config = json.loads(config_path.read_text())
    by_window = config.setdefault("by_window", {})
    for f in sorted(uv_dir.glob("*.json")):
        r = json.loads(f.read_text())
        rms = fit_gains(r, label)
        if rms:
            by_window.setdefault(r["window"], {})["gain_amp_rms"] = rms
        print(f"{r['source'][:24]:24s} {r['ghz']:6.1f} GHz  antenna amplitude: {rms or 'nothing to fit'}")
    out.write_text(json.dumps(config, indent=1))


def fit_baselines(result: dict, label: str) -> dict:
    """The per-(baseline, scan) amplitude and phase the real data has beyond
    what the simulation's antenna solve leaves (noise, screens): what does not
    close, in quadrature."""
    real, sim = result["real"], result[label]
    out = {}
    for key, name in (("nonclosing_lnamp_rms", "baseline_amp_rms"), ("nonclosing_phase_rms_deg", "baseline_phase_rms_deg")):
        if key in real and key in sim:
            excess = real[key] ** 2 - sim[key] ** 2
            if excess > 0:
                out[name] = round(float(np.sqrt(excess)), 4)
    return out


def baselines_main(uv_dir: Path, config_path: Path, out: Path, label: str, windows: set | None = None) -> None:
    config = json.loads(config_path.read_text())
    by_window = config.setdefault("by_window", {})
    for f in sorted(uv_dir.glob("*.json")):
        r = json.loads(f.read_text())
        if windows is not None and r["window"] not in windows:
            continue
        fit = fit_baselines(r, label)
        if fit:
            by_window.setdefault(r["window"], {}).update(fit)
        print(f"{r['source'][:24]:24s} {r['ghz']:6.1f} GHz  baseline-based: {fit or 'nothing to fit'}")
    out.write_text(json.dumps(config, indent=1))


if __name__ == "__main__" and sys.argv[1] == "baselines":
    baselines_main(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5])
elif __name__ == "__main__" and sys.argv[1] == "gains":
    gains_main(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5])
elif __name__ == "__main__":
    uv_dir, base, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    noise_label = sys.argv[4] if len(sys.argv) > 4 else "thermal-profile"
    config = json.loads(base.read_text())
    config["by_window"] = {}
    fits = {}
    for f in sorted(uv_dir.glob("*.json")):
        r = json.loads(f.read_text())
        slow, fast = fit(r, noise_label), fit_fast(r, noise_label)
        fits[r["window"]] = {"source": r["source"], "ghz": r["ghz"], "fit": slow, "fast": fast}
        entry = {}
        for label, result, key in (("scan", slow, "scan_screen"), ("fast", fast, "fast_screen")):
            if result:
                entry[f"{key}_path_um_1km"] = result["path_um_1km"]
                entry[f"{key}_slope"] = result["slope"]
                print(f"{r['source'][:24]:24s} {r['ghz']:6.1f} GHz  {label}: {result['path_um_1km']:6.1f} um at 1 km "
                      f"({result['phase_1km_deg']:5.1f} deg), slope {result['slope']:+.2f}, {result['bins']} bins "
                      f"{result['baseline_m'][0]}-{result['baseline_m'][-1]} m")
            else:
                print(f"{r['source'][:24]:24s} {r['ghz']:6.1f} GHz  {label}: nothing to fit")
        if entry:
            config["by_window"][r["window"]] = entry
    out.write_text(json.dumps(config, indent=1))
    (out.parent / (out.stem + ".fits.json")).write_text(json.dumps(fits, indent=1))
