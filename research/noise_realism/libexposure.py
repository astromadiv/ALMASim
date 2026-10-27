"""What each window observes, real against library: where the library's image noise
(0.6 of the real) comes from.

Per MS: antennas, integrations, time on source, channels and their width,
correlations, flagged share, the median noise per visibility (1/sqrt(WEIGHT)) and
the natural-weighting continuum noise 1/sqrt(sum of unflagged weights). The image
noise over the latter is the imaging factor.

    python libexposure.py OUT.csv

LIB_RUNS and LIB_ITER (default runs-p2, iter-lib) pick the library set, WINDOWS
(space-separated indices) a subset.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

NOISE = Path("/SKAO_data/cycle11/MADIV-28/noise")
LIB = NOISE / "libsim"
CHUNK = 20_000
LIB_RUNS = os.environ.get("LIB_RUNS", "runs-p2")
LIB_ITER = os.environ.get("LIB_ITER", "iter-lib")


def measure(ms: Path) -> dict:
    from casacore.tables import table

    with table(str(ms / "SPECTRAL_WINDOW"), ack=False) as sw:
        width = np.abs(np.asarray(sw.getcell("CHAN_WIDTH", 0), dtype=float))
    with table(str(ms), ack=False) as t:
        spectrum = "WEIGHT_SPECTRUM" in t.colnames() and t.iscelldefined("WEIGHT_SPECTRUM", 0)
        a1, a2 = t.getcol("ANTENNA1"), t.getcol("ANTENNA2")
        cross = a1 != a2
        times, interval = t.getcol("TIME"), t.getcol("INTERVAL")
        total_w = n_vis = n_flag = 0.0
        sigmas = []
        for start in range(0, t.nrows(), CHUNK):
            n = min(CHUNK, t.nrows() - start)
            flag = t.getcol("FLAG", start, n)  # (row, chan, corr)
            if spectrum:
                w = t.getcol("WEIGHT_SPECTRUM", start, n)
            else:
                w = np.broadcast_to(t.getcol("WEIGHT", start, n)[:, None, :], flag.shape)
            ok = ~flag & cross[start:start + n, None, None] & (w > 0)
            total_w += float(w[ok].sum())
            n_vis += float(ok.sum())
            n_flag += float((flag & cross[start:start + n, None, None]).sum())
            sigmas.append(1 / np.sqrt(w[ok][::97]))
    ants = np.union1d(a1[cross], a2[cross])
    uniq, idx = np.unique(times[cross], return_index=True)
    sig = np.concatenate(sigmas) if sigmas else np.array([np.nan])
    return {
        "antennas": len(ants),
        "integrations": len(uniq),
        "t_int": float(np.median(interval[cross])),
        "time_on_source": float(interval[cross][idx].sum()),
        "channels": len(width),
        "chan_width": float(np.median(width)),
        "bandwidth": float(width.sum()),
        "vis": n_vis,
        "flagged": n_flag / max(n_flag + n_vis, 1),
        "sigma_vis": float(np.median(sig)),
        "sigma_natural": float(1 / np.sqrt(total_w)) if total_w > 0 else np.nan,
    }


def window(args: tuple[int, str]) -> dict:
    i, name = args
    row = {"window": i, "name": name}
    for key, ms in (("real", NOISE / "base" / f"{name}.ms"),
                    ("lib", LIB / LIB_RUNS / f"w{i}" / f"w{i}_{i}.ms")):
        try:
            row.update({f"{key}:{k}": v for k, v in measure(ms).items()})
        except Exception as exc:  # noqa: BLE001 - one bad MS must not stop the table
            print(f"{i} {key}: {exc}", flush=True)
    for key, path in (("real", NOISE / "iter-f0-control" / "metrics" / f"{name}.json"),
                      ("lib", NOISE / LIB_ITER / "metrics" / f"{name}.json")):
        if path.exists():
            m = json.loads(path.read_text())
            row[f"{key}:cont_sigma"] = m.get("cont.sigma")
            row[f"{key}:cube_sigma"] = m.get("cube.sigma_median")
    print(f"done {i}", flush=True)
    return row


def main(out: Path) -> None:
    names = [line.split() for line in open(NOISE / "diag" / "index-names.txt")]
    keep = set(os.environ.get("WINDOWS", "").split())
    jobs = [(int(i), n) for i, n in names if not keep or i in keep]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(window, jobs))
    meta = pd.read_csv(LIB / "rows.csv", usecols=["window", "Int.Time", "antenna_arrays"])
    meta["row_antenna_entries"] = meta.antenna_arrays.str.split().str.len()
    df = pd.DataFrame(rows).merge(meta.drop(columns="antenna_arrays"), on="window", how="left")
    df.to_csv(out, index=False)
    print(f"{len(df)} windows -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
