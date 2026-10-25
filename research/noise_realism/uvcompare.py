"""Compare a window's real calibrated visibilities with simulated ones, before imaging.

    python uvcompare.py window I CONFIG.json [CONFIG.json ...] OUT.json
    python uvcompare.py summary OUT_DIR

Both real and simulated data have the same model visibilities subtracted
(the clean model the simulations are made from), and the residuals are
scaled by each visibility's expected noise, z = (V - M) sqrt(WEIGHT / scale),
so pure thermal noise gives unit variance per component. Then, for the real
data and for each configuration's simulation (regenerated with the seed its
imaged counterpart used):

- variance and excess kurtosis of z, overall and by baseline length:
  unmodelled sky shows on short baselines, atmospheric errors on long ones;
- variance by channel (the noise across the window) and by antenna;
- the correlation of consecutive integrations on the same baseline (the
  channel-averaged residual): thermal noise is white in time, gain and
  atmospheric errors are not;
- where the source is bright (|M| > 5 sigma): the scatter of the amplitude
  ratio and of the phase of V / M, by baseline length (decorrelation and
  phase errors);
- the variance of the channel-averaged residual (what a continuum image
  averages: noise or sky correlated over many channels raises it, beyond the
  neighbour correlation the simulation shares);
- the spread of the variance per (antenna, scan) beyond its sampling noise:
  antennas noisier, scan by scan, than their WEIGHT says.
"""

from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from counterpart import CHUNK, Simulator, _tables  # noqa: E402
from errors import C, ErrorModel  # noqa: E402
from window import name, subset  # noqa: E402

ROOT = "/SKAO_data/cycle11/MADIV-16/cubes"
BASE = Path("/SKAO_data/cycle11/MADIV-28/noise/base")
NBINS = 10


class Stats:
    """Running sums of the residual statistics for one data set."""

    def __init__(self, edges: np.ndarray, nchan: int, nant: int) -> None:
        self.edges = edges
        nb = len(edges) - 1
        self.bin_n = np.zeros(nb)
        self.bin_z2 = np.zeros(nb)
        self.bin_z4 = np.zeros(nb)
        self.chan_n = np.zeros(nchan)
        self.chan_z2 = np.zeros(nchan)
        self.ant_n = np.zeros(nant)
        self.ant_z2 = np.zeros(nant)
        self.lag_xy = 0.0
        self.lag_xx = 0.0
        self.ratio = {k: np.zeros(nb) for k in ("n", "la", "la2", "ph2")}
        # Per (baseline, scan), and per (baseline, 30 s within a scan): sums of
        # w V M* and w |M|^2, and the group's uv bin. Comparing the two shows how
        # much of the in-scan loss a 30 s self-calibration solution could follow.
        self.groups: dict[int, list] = {}
        self.groups30: dict[int, list] = {}
        # Sums of WEIGHT z^2 and WEIGHT: the variance an image sees.
        self.wz2 = 0.0
        self.w = 0.0
        # Channel-averaged residual: sums of |zbar|^2 / 2 and their count.
        self.bb_z2 = 0.0
        self.bb_n = 0
        # Per (antenna, scan): sums of z^2 and their count.
        self.as_z2: dict[int, float] = {}
        self.as_n: dict[int, float] = {}

    def add(self, z, good, uvbin, a1, a2, cols, data, clean, sigma, norm2) -> None:
        """z, good: (nrow, nchan, npol); uvbin (nrow,); sigma, norm2 (nrow, npol).

        ``norm2`` is WEIGHT / scale: the inverse noise variance per component.
        """
        w = np.where(good, norm2[:, None, :], 0.0)
        num_row = (w * data * np.conj(clean)).sum(axis=(1, 2))
        den_row = (w * np.abs(clean) ** 2).sum(axis=(1, 2))
        gkey = (a1.astype(np.int64) * 10_000 + a2) * 100_000 + cols["scan"].astype(np.int64)
        _accumulate(self.groups, gkey, num_row, den_row, uvbin)
        tbin = np.floor(cols["time"] / 30).astype(np.int64) % 1_000_000
        _accumulate(self.groups30, gkey * 1_000_000 + tbin, num_row, den_row, uvbin)
        z2 = np.where(good, z.real**2 + z.imag**2, 0.0) / 2  # per component
        self.wz2 += float((z2 * w).sum())
        self.w += float((w * good).sum())
        z4 = np.where(good, z.real**4 + z.imag**4, 0.0) / 2
        cnt = good.astype(float)
        per_row_n = cnt.sum(axis=(1, 2))
        per_row_z2 = z2.sum(axis=(1, 2))
        per_row_z4 = z4.sum(axis=(1, 2))
        np.add.at(self.bin_n, uvbin, per_row_n)
        np.add.at(self.bin_z2, uvbin, per_row_z2)
        np.add.at(self.bin_z4, uvbin, per_row_z4)
        self.chan_n += cnt.sum(axis=(0, 2))
        self.chan_z2 += z2.sum(axis=(0, 2))
        for a in (a1, a2):
            np.add.at(self.ant_n, a, per_row_n)
            np.add.at(self.ant_z2, a, per_row_z2)
        # Consecutive integrations of one baseline: the channel-averaged residual.
        nch = np.maximum(cnt.sum(axis=1), 1)  # (nrow, npol)
        zbar = np.where(good, z, 0).sum(axis=1) / np.sqrt(nch)
        full = cnt.sum(axis=1) > 0.5 * z.shape[1]
        self.bb_z2 += float((np.abs(zbar[full]) ** 2).sum() / 2)
        self.bb_n += int(full.sum())
        for a in (a1, a2):
            akey = a.astype(np.int64) * 100_000 + cols["scan"].astype(np.int64)
            keys, inverse = np.unique(akey, return_inverse=True)
            z2k = np.bincount(inverse, per_row_z2, len(keys))
            nk = np.bincount(inverse, per_row_n, len(keys))
            for k, zz, nn in zip(keys.tolist(), z2k, nk, strict=True):
                self.as_z2[k] = self.as_z2.get(k, 0.0) + zz
                self.as_n[k] = self.as_n.get(k, 0.0) + nn
        key = a1.astype(np.int64) * 10_000 + a2
        order = np.lexsort((cols["time"], key))
        k, tm, sc, zb = key[order], cols["time"][order], cols["scan"][order], zbar[order]
        same = (k[1:] == k[:-1]) & (sc[1:] == sc[:-1])
        if same.any():
            dt = np.diff(tm)[same]
            step = np.median(dt) if dt.size else 0
            near = same.copy()
            near[same] = dt <= 1.5 * step
            self.lag_xy += float(np.real(zb[1:][near] * np.conj(zb[:-1][near])).sum())
            self.lag_xx += float((np.abs(zb[1:][near]) ** 2 + np.abs(zb[:-1][near]) ** 2).sum() / 2)
        # Where the model is bright: data over model.
        m = clean
        bright = good & (np.abs(m) > 5 * np.sqrt(2) * sigma[:, None, :])
        if bright.any():
            r = data[bright] / m[bright]
            rows = np.broadcast_to(uvbin[:, None, None], bright.shape)[bright]
            np.add.at(self.ratio["n"], rows, 1)
            la = np.log(np.abs(r))
            np.add.at(self.ratio["la"], rows, la)
            np.add.at(self.ratio["la2"], rows, la**2)
            np.add.at(self.ratio["ph2"], rows, np.angle(r) ** 2)

    def group_result(self, min_den: float = 100.0, groups: dict | None = None, prefix: str = "group") -> dict:
        """Data over model per (baseline, scan), where it has S/N > 10: by uv bin,
        the phase rms (deg), and the mean and rms of ln|A|; with the noise alone
        would give (the median 1/sqrt(sum w|M|^2), per component)."""
        nb = len(self.edges) - 1
        a = [[] for _ in range(nb)]
        sig = [[] for _ in range(nb)]
        for num, den, b in (self.groups if groups is None else groups).values():
            if den > min_den:
                a[b].append(num / den)
                sig[b].append(1 / np.sqrt(den))
        out = {f"{prefix}_n": [len(x) for x in a]}
        out[f"{prefix}_phase_rms_deg"] = [
            float(np.degrees(np.sqrt(np.mean(np.angle(x) ** 2)))) if len(x) > 5 else None for x in a
        ]
        out[f"{prefix}_lnamp_mean"] = [float(np.mean(np.log(np.abs(x)))) if len(x) > 5 else None for x in a]
        out[f"{prefix}_lnamp_rms"] = [float(np.std(np.log(np.abs(x)))) if len(x) > 5 else None for x in a]
        out[f"{prefix}_noise_deg"] = [float(np.degrees(np.median(s))) if len(s) > 5 else None for s in sig]
        return out

    def antenna_solve(self, min_den: float = 100.0) -> dict:
        """Per scan, ln A of each bright (baseline, scan) split into antenna terms,
        ln A_ij = ln g_i + conj(ln g_j), by least squares: the rms of the antennas'
        amplitude (ln|g|) and phase, and of what does not close (baseline-based:
        the model not matching the source, or noise). Noise alone gives some of
        each; compare with the thermal simulation."""
        by_scan: dict[int, list] = {}
        for key, (num, den, _) in self.groups.items():
            if den > min_den:
                scan = key % 100_000
                pair = key // 100_000
                by_scan.setdefault(scan, []).append((pair // 10_000, pair % 10_000, num / den))
        amp, ph, res_a, res_p = [], [], [], []
        for rows in by_scan.values():
            ants = sorted({a for r in rows for a in r[:2]})
            if len(rows) < 2 * len(ants):
                continue
            index = {a: k for k, a in enumerate(ants)}
            design = np.zeros((len(rows), len(ants)))
            dphase = np.zeros((len(rows), len(ants)))
            la = np.array([np.log(abs(r[2])) for r in rows])
            lp = np.array([np.angle(r[2]) for r in rows])
            for k, (i, j, _) in enumerate(rows):
                design[k, index[i]] = design[k, index[j]] = 1
                dphase[k, index[i]], dphase[k, index[j]] = 1, -1
            ga = np.linalg.lstsq(design, la, rcond=None)[0]
            gp = np.linalg.lstsq(dphase, lp, rcond=None)[0]  # up to a common phase
            amp.extend(ga - ga.mean())
            ph.extend(gp - gp.mean())
            res_a.extend(la - design @ ga)
            res_p.extend(np.angle(np.exp(1j * (lp - dphase @ gp))))
        if not amp:
            return {}
        return {
            "ant_lnamp_rms": float(np.std(amp)),
            "ant_phase_rms_deg": float(np.degrees(np.std(ph))),
            "nonclosing_lnamp_rms": float(np.std(res_a)),
            "nonclosing_phase_rms_deg": float(np.degrees(np.std(res_p))),
        }

    def antenna_scan_spread(self) -> dict:
        """The rms of each (antenna, scan)'s variance over the mean, less what its
        number of samples alone gives (per component z^2 has variance 2)."""
        n = np.array(list(self.as_n.values()))
        v = np.array([self.as_z2[k] for k in self.as_n]) / np.maximum(n, 1)
        keep = n > 100
        if keep.sum() < 10:
            return {}
        ratio = v[keep] / np.average(v[keep], weights=n[keep])
        observed = float(np.average((ratio - 1) ** 2, weights=n[keep]))
        sampling = float(np.average(2.0 / n[keep], weights=n[keep]))
        return {
            "antenna_scan_spread": float(np.sqrt(max(observed - sampling, 0.0))),
            "antenna_scan_spread_raw": float(np.sqrt(observed)),
        }

    def result(self) -> dict:
        n = np.maximum(self.bin_n, 1)
        var = self.bin_z2 / n
        tot_n, tot_z2, tot_z4 = self.bin_n.sum(), self.bin_z2.sum(), self.bin_z4.sum()
        tvar = tot_z2 / max(tot_n, 1)
        rn = np.maximum(self.ratio["n"], 1)
        chan = self.chan_z2 / np.maximum(self.chan_n, 1)
        ant = self.ant_z2 / np.maximum(self.ant_n, 1)
        used_ant = self.ant_n > 0
        return {
            "variance": float(tvar),
            "kurtosis": float(tot_z4 / max(tot_n, 1) / tvar**2 - 3) if tvar else None,
            "uv_edges_klambda": (self.edges / 1e3).round(2).tolist(),
            "uv_variance": var.round(4).tolist(),
            "uv_kurtosis": np.where(var > 0, self.bin_z4 / n / np.maximum(var, 1e-12) ** 2 - 3, 0).round(3).tolist(),
            "uv_count": self.bin_n.astype(int).tolist(),
            "channel_variance_spread": float(np.std(chan[self.chan_n > 0]) / np.mean(chan[self.chan_n > 0])),
            "channel_variance": chan.round(4).tolist(),
            "antenna_variance_spread": float(np.std(ant[used_ant]) / np.mean(ant[used_ant])),
            "time_lag1_corr": float(self.lag_xy / self.lag_xx) if self.lag_xx else None,
            "bright_n": self.ratio["n"].astype(int).tolist(),
            "bright_amp_rms": np.sqrt(self.ratio["la2"] / rn - (self.ratio["la"] / rn) ** 2).round(4).tolist(),
            "bright_amp_mean": (self.ratio["la"] / rn).round(4).tolist(),
            "bright_phase_rms_deg": np.degrees(np.sqrt(self.ratio["ph2"] / rn)).round(3).tolist(),
            "variance_weighted": float(self.wz2 / self.w) if self.w else None,
            "broadband_variance": float(self.bb_z2 / self.bb_n) if self.bb_n else None,
            **self.antenna_scan_spread(),
            **self.group_result(),
            **self.group_result(min_den=25.0, groups=self.groups30, prefix="group30"),
            **self.antenna_solve(),
        }


def _accumulate(groups: dict, gkey: np.ndarray, num_row, den_row, uvbin) -> None:
    keys, inverse = np.unique(gkey, return_inverse=True)
    num = np.zeros(len(keys), dtype=np.complex128)
    den = np.zeros(len(keys))
    np.add.at(num, inverse, num_row)
    np.add.at(den, inverse, den_row)
    first_bin = np.zeros(len(keys), dtype=int)
    first_bin[inverse[::-1]] = uvbin[::-1]
    for k, nu, de, b in zip(keys.tolist(), num, den, first_bin, strict=True):
        g = groups.get(k)
        if g is None:
            groups[k] = [nu, de, int(b)]
        else:
            g[0] += nu
            g[1] += de


def window(i: int, configs: list[Path], out: Path) -> None:
    row = subset().iloc[i]
    win = name(row)
    ms = BASE / f"{win}.ms"
    task = Path(ROOT) / row.task
    table = _tables()
    with table(str(ms), ack=False) as t:
        sims = []
        for path in configs:
            config = json.loads(path.read_text())
            seed = config.get("seed", 0) * 1000003 + zlib.crc32(win.encode()) % 1000003
            errors = {**config["errors"], **config.get("by_window", {}).get(win, {})}
            sims.append((path.stem, Simulator(ms, task, ErrorModel.from_dict(errors), seed, t)))
        first = sims[0][1]
        freq_mid = float(np.mean(first.chan_freq))
        uvdist = np.hypot(*t.getcol("UVW")[:, :2].T) * freq_mid / C
        lo, hi = np.percentile(uvdist[uvdist > 0], [0.5, 99.5])
        edges = np.geomspace(lo, hi, NBINS + 1)
        nant = int(max(t.getcol("ANTENNA1").max(), t.getcol("ANTENNA2").max()) + 1)
        stats = {k: Stats(edges, len(first.chan_freq), nant) for k in ["real", *(n for n, _ in sims)]}
        scale = first.noise["scale"]
        hands = first.hands
        # UV_ROWS: read only every k-th chunk, about this many rows in all (the same
        # rows for the real data and every simulation).
        stride = max(1, t.nrows() // int(os.environ.get("UV_ROWS") or t.nrows()))
        for start in range(0, t.nrows(), CHUNK * stride):
            n = min(CHUNK, t.nrows() - start)
            real = t.getcol("DATA", startrow=start, nrow=n)[:, :, hands]
            flag = t.getcol("FLAG", startrow=start, nrow=n)[:, :, hands]
            cols, clean, data = first.chunk(t, start, n)
            vis = clean[:, :, hands[0]]
            weight = cols["weight"][:, hands]
            good = ~flag & (weight[:, None, :] > 0)
            norm2 = np.where(weight > 0, weight, 0) / scale
            norm = np.sqrt(norm2)[:, None, :]
            sigma = np.sqrt(scale / np.where(weight > 0, weight, np.inf))
            uvbin = np.clip(np.searchsorted(edges, np.hypot(*cols["uvw"][:, :2].T) * freq_mid / C) - 1, 0, NBINS - 1)
            m = clean[:, :, hands]
            stats["real"].add((real - m) * norm, good, uvbin, cols["a1"], cols["a2"], cols, real, m, sigma, norm2)
            for k, (label, sim) in enumerate(sims):
                if k:
                    _, _, data = sim.chunk(t, start, n, vis=vis)
                d = data[:, :, hands]
                stats[label].add((d - m) * norm, good, uvbin, cols["a1"], cols["a2"], cols, d, m, sigma, norm2)
    result = {"window": win, "source": row.source, "ghz": float(row.ghz), "noise": first.noise, "stride": stride}
    result.update({k: s.result() for k, s in stats.items()})
    out.write_text(json.dumps(result, indent=1))


def summary(out_dir: Path) -> None:
    files = sorted(out_dir.glob("*.json"))
    rows = [json.loads(f.read_text()) for f in files]
    labels = [k for k in rows[0] if isinstance(rows[0][k], dict) and "variance" in rows[0][k]]
    print(f"{len(rows)} windows; data sets: {', '.join(labels)}\n")
    for metric in ("variance", "kurtosis", "time_lag1_corr", "channel_variance_spread", "antenna_variance_spread"):
        print(f"{metric}: median over windows")
        for label in labels:
            vals = [r[label][metric] for r in rows if r[label][metric] is not None]
            print(f"  {label:20s} {np.median(vals):+.4f}  (16-84%: {np.percentile(vals, 16):+.4f} .. {np.percentile(vals, 84):+.4f})")
    print("\nshort- vs long-baseline variance (first and last uv bin), median over windows:")
    for label in labels:
        s = np.median([r[label]["uv_variance"][0] for r in rows])
        l = np.median([r[label]["uv_variance"][-1] for r in rows])
        print(f"  {label:20s} short {s:.3f}  long {l:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "window":
        window(int(sys.argv[2]), [Path(p) for p in sys.argv[3:-1]], Path(sys.argv[-1]))
    else:
        summary(Path(sys.argv[2]))
