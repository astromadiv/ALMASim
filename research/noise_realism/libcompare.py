"""The library's own simulations of the windows (libsim.py, pass 2) against the
real windows, in the visibilities: each uvcompare statistic, real (uvfinal5)
against library (libsim/uv-p2), per window and across them.

    python libcompare.py [OUT.json]

Images are compared with compare.py (iter-f0-control against iter-lib).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

NOISE = Path("/SKAO_data/cycle11/MADIV-28/noise")
STATS = (
    "variance",
    "kurtosis",
    "channel_variance_spread",
    "antenna_variance_spread",
    "time_lag1_corr",
    "broadband_variance",
)


def main(out: Path | None) -> None:
    names = dict(line.split() for line in (NOISE / "diag" / "index-names.txt").read_text().splitlines())
    rows = []
    for i, name in names.items():
        lib = NOISE / "libsim" / "uv-p2" / f"{name}.json"
        real = NOISE / "uvfinal5" / f"{i}.json"
        if lib.exists() and real.exists():
            rows.append((int(i), json.loads(real.read_text())["real"], json.loads(lib.read_text())["sim"]))
    summary = {"windows": len(rows)}
    print(f"{len(rows)} windows")
    print(f"{'statistic':26s} {'real':>9s} {'library':>9s} {'diff':>9s} {'p':>6s}")
    for stat in STATS:
        pairs = np.array([(r[stat], s[stat]) for _, r, s in rows if r.get(stat) is not None and s.get(stat) is not None])
        if not len(pairs):
            continue
        diff = pairs[:, 1] - pairs[:, 0]
        p = float(wilcoxon(diff).pvalue) if len(diff) > 5 and np.any(diff) else float("nan")
        real, lib = np.median(pairs, axis=0)
        summary[stat] = {"real": real, "library": lib, "median_diff": float(np.median(diff)), "p": p}
        print(f"{stat:26s} {real:9.4f} {lib:9.4f} {np.median(diff):9.4f} {p:6.2f}")
    if out:
        out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
