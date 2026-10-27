"""The imaging factor (image noise over the natural-weighting noise), split into
what the pipeline's Briggs weighting alone gives and what is left (calibration
residuals, sidelobes, cleaning).

The theoretical Briggs noise of a continuum (MFS) image, as WSClean weights it:
each visibility's density W is the sum of natural weights in its uv cell (the
image grid, both (u, v) and (-u, -v)), f^2 = (5 10^-R)^2 / (sum W_k^2 / sum w),
its imaging weight w / (1 + W f^2), and the image noise
sqrt(sum w'^2 / w) / sum w'. Every STRIDE-th channel stands for its neighbours.

    python libweighting.py OUT.csv I [I ...]      real and library (runs-eba) MSs
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NOISE = Path("/SKAO_data/cycle11/MADIV-28/noise")
LIB = NOISE / "libsim"
C = 299_792_458.0
CHUNK = 20_000
STRIDE = 8


def imaging_grid(log: Path) -> tuple[float, int, float]:
    """Cell (rad), image size and robust of the pipeline's continuum WSClean run."""
    text = log.read_text(errors="replace")
    command = re.search(r"wsclean -name \S+/wsclean (.*)", text).group(1)
    cell = float(re.search(r"-scale ([0-9.eE+-]+)asec", command).group(1)) / 206_264.806
    size = int(re.search(r"-size (\d+)", command).group(1))
    robust = float(re.search(r"-weight briggs ([0-9.+-]+)", command).group(1))
    return cell, size, robust


def _chunks(ms: Path):
    from casacore.tables import table

    with table(str(ms / "SPECTRAL_WINDOW"), ack=False) as sw:
        freq = np.asarray(sw.getcell("CHAN_FREQ", 0), dtype=float)
    channels = np.arange(0, len(freq), STRIDE)
    with table(str(ms), ack=False) as t:
        spectrum = "WEIGHT_SPECTRUM" in t.colnames() and t.iscelldefined("WEIGHT_SPECTRUM", 0)
        for start in range(0, t.nrows(), CHUNK):
            n = min(CHUNK, t.nrows() - start)
            uvw = t.getcol("UVW", start, n)
            cross = t.getcol("ANTENNA1", start, n) != t.getcol("ANTENNA2", start, n)
            flag = t.getcol("FLAG", start, n)[:, channels, :]
            if spectrum:
                w = t.getcol("WEIGHT_SPECTRUM", start, n)[:, channels, :]
            else:
                w = np.broadcast_to(t.getcol("WEIGHT", start, n)[:, None, :], flag.shape)
            # Stokes I from the two hands: each hand's weight adds.
            w = np.where(flag, 0.0, w).sum(axis=2) * cross[:, None]
            u = uvw[:, 0:1] * freq[channels][None, :] / C
            v = uvw[:, 1:2] * freq[channels][None, :] / C
            yield u.ravel(), v.ravel(), w.ravel()


def briggs(ms: Path, cell: float, size: int, robust: float) -> dict:
    du = 1.0 / (size * cell)
    half = size // 2

    def cells(u, v):
        iu = np.rint(u / du).astype(np.int64) + half
        iv = np.rint(v / du).astype(np.int64) + half
        ok = (iu >= 0) & (iu < size) & (iv >= 0) & (iv < size)
        return iu * size + iv, ok

    density = np.zeros(size * size)
    for u, v, w in _chunks(ms):
        for sign in (1, -1):
            k, ok = cells(sign * u, sign * v)
            density += np.bincount(k[ok], weights=w[ok], minlength=size * size)
    sum_w = density.sum() / 2
    f2 = (5 * 10 ** (-robust)) ** 2 / (np.sum(density**2) / (2 * sum_w))
    s_w = s_w2_over_w = s_nat = 0.0
    for u, v, w in _chunks(ms):
        k, ok = cells(u, v)
        good = ok & (w > 0)
        wi = w[good] / (1 + density[k[good]] * f2)
        s_w += wi.sum()
        s_w2_over_w += np.sum(wi**2 / w[good])
        s_nat += w[good].sum()
    natural = 1 / np.sqrt(s_nat)
    return {"natural": natural, "briggs": np.sqrt(s_w2_over_w) / s_w}


def main(out: Path, windows: list[int]) -> None:
    names = dict(line.split() for line in open(NOISE / "diag" / "index-names.txt"))
    rows = []
    for i in windows:
        name = names[str(i)]
        log = next((LIB / "img-eba" / f"w{i}" / "cubes").glob("*/SPW-0_FIELD-0.log"))
        cell, size, robust = imaging_grid(log)
        row = {"window": i, "cell_arcsec": cell * 206_264.806, "size": size, "robust": robust}
        for key, ms, metrics in (
            ("real", NOISE / "base" / f"{name}.ms", NOISE / "iter-f0-control" / "metrics" / f"{name}.json"),
            ("lib", LIB / "runs-eba" / f"w{i}" / f"w{i}_{i}.ms", NOISE / "iter-lib-eba" / "metrics" / f"{name}.json"),
        ):
            b = briggs(ms, cell, size, robust)
            image = json.loads(metrics.read_text())["cont.sigma"]
            row.update({f"{key}:natural": b["natural"], f"{key}:briggs": b["briggs"], f"{key}:image": image,
                        f"{key}:weighting": b["briggs"] / b["natural"], f"{key}:residual": image / b["briggs"]})
        print(f"done {i}", flush=True)
        rows.append(row)
    d = pd.DataFrame(rows)
    for k in ("weighting", "residual"):
        d[f"ratio:{k}"] = d[f"lib:{k}"] / d[f"real:{k}"]
    d["ratio:imaging"] = (d["lib:image"] / d["lib:natural"]) / (d["real:image"] / d["real:natural"])
    d["ratio:natural"] = d["lib:natural"] / d["real:natural"]
    d["ratio:image"] = d["lib:image"] / d["real:image"]
    d.to_csv(out, index=False)
    cols = ["window", "real:weighting", "lib:weighting", "real:residual", "lib:residual",
            "ratio:natural", "ratio:weighting", "ratio:residual", "ratio:image"]
    print(d[cols].round(3).to_string(index=False))
    print(d[cols[1:]].median().round(3).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), [int(i) for i in sys.argv[2:]])
