"""The visibility statistics of an ALMASim-written MS, as uvcompare.py measures the
real windows: z = (DATA - MODEL_DATA) sqrt(WEIGHT), the noise relative to what
its weights say.

    python libuv.py MS OUT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uvcompare import NBINS, Stats  # noqa: E402

CHUNK = 50_000
PARALLEL = {5, 8, 9, 12}  # RR, LL, XX, YY


def main(ms: Path, out: Path) -> None:
    from casacore.tables import table

    with table(str(ms / "POLARIZATION"), ack=False) as pol:
        corr = np.asarray(pol.getcell("CORR_TYPE", 0))
    hands = np.flatnonzero(np.isin(corr, list(PARALLEL)))
    with table(str(ms / "SPECTRAL_WINDOW"), ack=False) as sw:
        freq = np.asarray(sw.getcell("CHAN_FREQ", 0), dtype=float)
    with table(str(ms), ack=False) as t:
        nrow = t.nrows()
        uvw = t.getcol("UVW")
        uvdist = np.hypot(uvw[:, 0], uvw[:, 1]) * freq.mean() / 299_792_458.0
        lo, hi = np.percentile(uvdist[uvdist > 0], [0.5, 99.5])
        edges = np.geomspace(lo, hi, NBINS + 1)
        nant = int(max(t.getcol("ANTENNA1").max(), t.getcol("ANTENNA2").max()) + 1)
        stats = Stats(edges, len(freq), nant)
        for start in range(0, nrow, CHUNK):
            n = min(CHUNK, nrow - start)
            data = t.getcol("DATA", startrow=start, nrow=n)[:, :, hands]
            model = t.getcol("MODEL_DATA", startrow=start, nrow=n)[:, :, hands]
            flag = t.getcol("FLAG", startrow=start, nrow=n)[:, :, hands]
            weight = t.getcol("WEIGHT", startrow=start, nrow=n)[:, hands]
            cols = {
                "a1": t.getcol("ANTENNA1", startrow=start, nrow=n),
                "a2": t.getcol("ANTENNA2", startrow=start, nrow=n),
                "scan": t.getcol("SCAN_NUMBER", startrow=start, nrow=n),
                "time": t.getcol("TIME", startrow=start, nrow=n),
            }
            good = ~flag & (weight[:, None, :] > 0)
            norm2 = np.where(weight > 0, weight, 0.0)
            sigma = np.sqrt(1.0 / np.where(weight > 0, weight, np.inf))
            uvbin = np.clip(np.searchsorted(edges, uvdist[start : start + n]) - 1, 0, NBINS - 1)
            z = (data - model) * np.sqrt(norm2)[:, None, :]
            stats.add(z, good, uvbin, cols["a1"], cols["a2"], cols, data, model, sigma, norm2)
    result = {"ms": str(ms), "nrow": int(nrow), "nchan": len(freq), "nant": nant, "sim": stats.result()}
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
