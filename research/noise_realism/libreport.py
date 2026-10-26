"""The 100 windows in six columns: the real data, the counterpart simulated from
its clean model, the library's own simulation of the archive row with its
source scaled to the real continuum S/N (libsim.py, two passes), and the
library on three skies made from the real source (libsim.py real_sky: point,
Gaussian, Gaussian mixture).

    python libreport.py OUT_DIR

Writes rows-NN.png (ten windows each: the continuum images on one field and
one colour scale per row, each panel labelled with its noise, S/N and
kurtosis), stats.csv (per window and column) and summary.json (per column,
against the real data: medians, Wilcoxon p, windows within 10%).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window import ROOT, WINDOWS, name  # noqa: E402

NOISE = Path("/SKAO_data/cycle11/MADIV-28/noise")
LIB = NOISE / "libsim"
# The counterpart's latest model per window, first found (as report.py's LATEST).
COUNTERPART = ("iter-f10-bright", "iter-f3-bright", "iter-f2-bright", "iter-f3-sim", "iter-f2-sim")
SKIES = ("point", "gaussian", "mixture")
COLUMNS = ("real", "counterpart", "snmatched", *SKIES)
TITLES = {
    "real": "Real",
    "counterpart": "Counterpart (clean model)",
    "snmatched": "Library: S/N-matched",
    "point": "Library: point",
    "gaussian": "Library: Gaussian",
    "mixture": "Library: mixture",
}
STATS = {  # metric -> "log" (ratio to real) or "diff"
    "cont.sigma": "log",
    "q.continuum.snr": "log",
    "cont.kurtosis": "diff",
    "cont.corr_area_beams": "log",
    "q.continuum.residual_peak_sigma": "diff",
    "cube.sigma_median": "log",
    "cube.sigma_spread": "diff",
    "cube.adjacent_corr": "diff",
    "cube.kurtosis": "diff",
    "cube.corr_area_beams": "log",
}
UV_STATS = ("variance", "kurtosis", "antenna_variance_spread", "time_lag1_corr", "channel_variance_spread")


def windows() -> list[tuple[int, str, str]]:
    """(index, name, MADIV-16 task) for the 100 windows."""
    tasks = {name(r): r.task for r in pd.read_csv(WINDOWS).itertuples()}
    out = []
    for line in (NOISE / "diag" / "index-names.txt").read_text().splitlines():
        i, n = line.split()
        out.append((int(i), n, tasks.get(n, "")))
    return out


def paths(i: int, window_name: str, task: str) -> dict[str, tuple[Path, Path, Path | None]]:
    """Per column: continuum image, metrics JSON, visibility stats JSON (or None)."""
    found = {"real": (Path(ROOT) / "cubes" / task / "wsclean-image.fits",
                      NOISE / "iter-f0-control" / "metrics" / f"{window_name}.json",
                      NOISE / "uvfinal5" / f"{i}.json")}
    for it in COUNTERPART:
        metrics = NOISE / it / "metrics" / f"{window_name}.json"
        if metrics.exists():
            found["counterpart"] = (NOISE / it / "cubes" / window_name / "SPW-0_FIELD-0" / "wsclean-image.fits",
                                    metrics, None)
            break
    found["snmatched"] = (LIB / "img-p2" / f"w{i}" / "cubes" / f"w{i}_{i}" / "SPW-0_FIELD-0" / "wsclean-image.fits",
                          NOISE / "iter-lib" / "metrics" / f"{window_name}.json",
                          LIB / "uv-p2" / f"{window_name}.json")
    for sky in SKIES:
        found[sky] = (LIB / "img-sky" / sky / f"w{i}" / "cubes" / f"w{i}_{i}" / "SPW-0_FIELD-0" / "wsclean-image.fits",
                      NOISE / f"iter-sky-{sky}" / "metrics" / f"{window_name}.json",
                      LIB / f"uv-sky-{sky}" / f"{window_name}.json")
    return found


def _image(path: Path):
    from astropy.io import fits

    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        cell = abs(float(hdul[0].header["CDELT2"])) * 3600.0
    while data.ndim > 2:
        data = data[0]
    return np.nan_to_num(data), cell


def _metrics(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _uv(path: Path | None, key: str) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text()).get(key, {})


def figures(rows: list, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for start in range(0, len(rows), 10):
        chunk = rows[start : start + 10]
        fig, axes = plt.subplots(len(chunk), len(COLUMNS), figsize=(3.2 * len(COLUMNS), 3.3 * len(chunk)), squeeze=False)
        for r, (i, window_name, task, found, metrics) in enumerate(chunk):
            real_sigma = metrics["real"].get("cont.sigma") or 1.0
            images = {}
            for col in COLUMNS:
                if col in found and found[col][0].exists():
                    images[col] = _image(found[col][0])
            # One field for the row: the real image's inner 60%, at most what each image covers.
            half = 0.3 * min(img.shape[0] * cell for img, cell in images.values()) if images else 1.0
            if "real" in images:
                half = 0.3 * images["real"][0].shape[0] * images["real"][1]
                half = min([half] + [0.5 * img.shape[0] * cell for img, cell in images.values()])
            vmax = max(metrics["real"].get("cont.peak") or 0.0, 5 * real_sigma)
            for c, col in enumerate(COLUMNS):
                ax = axes[r, c]
                ax.set_xticks([])
                ax.set_yticks([])
                if col not in images:
                    ax.text(0.5, 0.5, "not run yet" if col not in ("real", "counterpart") else "n/a", ha="center", va="center",
                            transform=ax.transAxes, color="0.5")
                else:
                    img, cell = images[col]
                    n = img.shape[0]
                    k = int(min(half / cell, n / 2))
                    cut = img[n // 2 - k : n // 2 + k, n // 2 - k : n // 2 + k]
                    ax.imshow(np.arcsinh(cut / real_sigma), origin="lower", cmap="inferno",
                              vmin=np.arcsinh(-3), vmax=np.arcsinh(vmax / real_sigma),
                              extent=(k * cell, -k * cell, -k * cell, k * cell))
                m = metrics.get(col, {})
                if m:
                    sigma = m.get("cont.sigma")
                    snr = m.get("q.continuum.snr")
                    kurt = m.get("cont.kurtosis")
                    label = (f"σ {sigma * 1e3:.3g} mJy/b" if sigma else "σ –") + \
                            (f"  S/N {snr:.3g}" if snr else "") + (f"  κ {kurt:+.2f}" if kurt is not None else "")
                    ax.set_xlabel(label, fontsize=8)
                if r == 0:
                    ax.set_title(TITLES[col], fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"{i}: {window_name[-26:]}", fontsize=8)
        fig.suptitle(f"Continuum images, windows {chunk[0][0]}–{chunk[-1][0]} "
                     "(asinh, per row in units of the real noise: −3σ to the real peak)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(out / f"rows-{start // 10:02d}.png", dpi=72)
        plt.close(fig)


def statistics(rows: list, out: Path) -> None:
    table = []
    for i, window_name, _, found, metrics in rows:
        record = {"window": i, "name": window_name}
        for col in COLUMNS:
            for key in STATS:
                record[f"{col}:{key}"] = metrics.get(col, {}).get(key)
            uv = _uv(found[col][2] if col in found else None, "real" if col == "real" else "sim")
            for key in UV_STATS:
                record[f"{col}:uv.{key}"] = uv.get(key)
        table.append(record)
    frame = pd.DataFrame(table)
    frame.to_csv(out / "stats.csv", index=False)

    from scipy.stats import wilcoxon

    summary: dict = {}
    for col in COLUMNS[1:]:
        summary[col] = {}
        for key, how in [*STATS.items(), *((f"uv.{k}", "diff") for k in UV_STATS)]:
            real = pd.to_numeric(frame[f"real:{key}"], errors="coerce")
            sim = pd.to_numeric(frame[f"{col}:{key}"], errors="coerce")
            ok = real.notna() & sim.notna()
            if how == "log":
                ok &= (real > 0) & (sim > 0)
                diff = np.log10(sim[ok] / real[ok])
            else:
                diff = sim[ok] - real[ok]
            if not ok.sum():
                continue
            entry = {"n": int(ok.sum()), "median_real": float(real[ok].median()),
                     "median_sim": float(sim[ok].median()),
                     "median": float(10 ** diff.median() if how == "log" else diff.median()),
                     "p": float(wilcoxon(diff).pvalue) if ok.sum() > 5 and np.any(diff != 0) else None}
            if how == "log":
                entry["within_10pct"] = int((np.abs(10**diff - 1) <= 0.1).sum())
            summary[col][key] = entry
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for col, entries in summary.items():
        print(f"== {col}")
        for key, e in entries.items():
            extra = f" within10% {e['within_10pct']}/{e['n']}" if "within_10pct" in e else ""
            p = f"{e['p']:.2f}" if e["p"] is not None else "–"
            print(f"  {key:34s} n {e['n']:3d} real {e['median_real']:.4g} sim {e['median_sim']:.4g} "
                  f"{'ratio' if 'within_10pct' in e else 'diff'} {e['median']:.4g} p {p}{extra}")


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, window_name, task in windows():
        found = paths(i, window_name, task)
        metrics = {col: _metrics(found[col][1]) for col in found}
        rows.append((i, window_name, task, found, metrics))
    statistics(rows, out)
    figures(rows, out)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
