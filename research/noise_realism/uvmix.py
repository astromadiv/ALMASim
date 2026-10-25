"""Where does the real visibilities' excess kurtosis come from? A noise level
that changes without WEIGHT following it (by antenna, scan or integration),
or a few outliers.

    python uvmix.py I CONFIG.json OUT.json [STRIDE_ROWS]
    python uvmix.py profile OUT.npz BASE.ms

For the real data and for the window's counterpart (CONFIG, its seed), with
the clean model subtracted and scaled by WEIGHT as in uvcompare.py, on the
channels whose real variance over the track is within 5% of the simulated
one (so no unmodelled line sky):

- each row's (baseline, integration) mean z^2 over its channels and hands;
- per integration, ln v_ij split by least squares into antenna terms,
  ln v_ij = u_i + u_j + e_ij; the antenna terms then split into their
  antenna mean over the track, the scan mean about it, and what is left
  within the scan; e_ij is baseline-based. Each part's rms, real and
  simulated (the simulation gives what sampling noise alone gives);
- the fraction of components beyond 5 and 8 sigma;
- per (antenna, hand, channel), the variance over the track, divided by its
  antenna-and-hand mean and by its channel-and-hand mean: the rms of what is
  left (each antenna's own spectral shape of the noise, which one array-wide
  channel profile misses), and the rms over channels of the two hands'
  profiles' ratio. Again the simulation gives the sampling noise.

Only every STRIDE-th chunk of rows is read (integrations stay whole within a
chunk), at least 1.5 million rows in all.
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from counterpart import Simulator, _tables  # noqa: E402
from errors import ErrorModel, antenna_noise_shape  # noqa: E402
from window import name, subset  # noqa: E402

ROOT = "/SKAO_data/cycle11/MADIV-16/cubes"
BASE = Path("/SKAO_data/cycle11/MADIV-28/noise/base")
UV = Path("/SKAO_data/cycle11/MADIV-28/noise")
CHUNK = 20_000


class Rows:
    def __init__(self) -> None:
        self.parts: dict[str, list] = {k: [] for k in ("a1", "a2", "time", "scan", "v", "n")}
        self.big5 = 0
        self.big8 = 0
        self.count = 0
        self.ant_z2: np.ndarray | None = None  # (nant, nchan, npol): sums of |z|^2
        self.ant_n: np.ndarray | None = None
        self.z2 = 0.0
        self.z4 = 0.0

    def add(self, z: np.ndarray, good: np.ndarray, cols: dict) -> None:
        comp = np.concatenate([z.real[good], z.imag[good]])
        self.big5 += int((np.abs(comp) > 5).sum())
        self.big8 += int((np.abs(comp) > 8).sum())
        self.count += comp.size
        self.z2 += float((comp**2).sum())
        self.z4 += float((comp**4).sum())
        z2 = np.where(good, z.real**2 + z.imag**2, 0.0)
        if self.ant_z2 is None:
            self.ant_z2 = np.zeros((256,) + z.shape[1:])
            self.ant_n = np.zeros((256,) + z.shape[1:])
        for a in np.union1d(cols["a1"], cols["a2"]):
            rows = (cols["a1"] == a) | (cols["a2"] == a)
            self.ant_z2[a] += z2[rows].sum(axis=0)
            self.ant_n[a] += good[rows].sum(axis=0)
        n = 2 * good.sum(axis=(1, 2))
        keep = n > 0
        for k, v in (("a1", cols["a1"]), ("a2", cols["a2"]), ("time", cols["time"]), ("scan", cols["scan"])):
            self.parts[k].append(v[keep])
        self.parts["v"].append(z2.sum(axis=(1, 2))[keep] / n[keep])
        self.parts["n"].append(n[keep])

    def antenna_channels(self) -> dict:
        total = self.ant_n.sum(axis=1)  # antennas with most of the data in every hand
        seen = total.sum(axis=1) > 0
        used = seen & (total > 0.5 * np.median(total[seen], axis=0)).all(axis=1)
        n, z2 = self.ant_n[used], self.ant_z2[used]
        ok = (n > 50).all(axis=(0, 2))  # channels every such antenna samples
        n, z2 = n[:, ok], z2[:, ok]
        v = z2 / (2 * n)
        # v ~ ant(a, p) * chan(c, p): divide out both, by alternating means.
        ant = np.ones((v.shape[0], 1, v.shape[2]))
        for _ in range(5):
            chan = (v / ant).mean(axis=0, keepdims=True)
            ant = (v / chan).mean(axis=1, keepdims=True)
        left = v / (ant * chan) - 1
        out = {
            "antchan_rms": float(np.std(left)),
            "antchan_sampling": float(np.sqrt(np.mean(1 / n))),
            "hands_ratio_rms": float(np.std(np.log(chan[0, :, 0] / chan[0, :, -1]))) if v.shape[2] > 1 else None,
        }
        # How much of it is smooth across the window: the rms of each antenna's
        # residual shape averaged in blocks of 16 channels.
        nb = left.shape[1] // 16
        if nb >= 2:
            blocks = left[:, : nb * 16].reshape(left.shape[0], nb, 16, -1).mean(axis=2)
            out["antchan_rms_16"] = float(np.std(blocks))
            out["antchan_sampling_16"] = float(np.sqrt(np.mean(1 / n)) / 4)
        return out

    def result(self) -> dict:
        p = {k: np.concatenate(v) for k, v in self.parts.items()}
        out = {
            "rows": int(len(p["v"])),
            "frac_5sigma": self.big5 / self.count,
            "frac_8sigma": self.big8 / self.count,
            "kurtosis": (self.z4 / self.count) / (self.z2 / self.count) ** 2 - 3,
            "row_lnv_rms": float(np.std(np.log(np.maximum(p["v"], 1e-6)))),
            **self.antenna_channels(),
        }
        lnv = np.log(np.maximum(p["v"], 1e-6))
        nant = int(max(p["a1"].max(), p["a2"].max()) + 1)
        tkey = np.round(p["time"], 1)
        u_rows, e_all = [], []
        for t in np.unique(tkey):
            sel = np.flatnonzero(tkey == t)
            ants = np.union1d(p["a1"][sel], p["a2"][sel])
            if len(sel) < 3 * len(ants) or len(ants) < 5:
                continue
            index = np.full(nant, -1)
            index[ants] = np.arange(len(ants))
            design = np.zeros((len(sel), len(ants)))
            design[np.arange(len(sel)), index[p["a1"][sel]]] = 1
            design[np.arange(len(sel)), index[p["a2"][sel]]] = 1
            u, *_ = np.linalg.lstsq(design, lnv[sel], rcond=None)
            e_all.append(lnv[sel] - design @ u)
            u = u - np.median(u)  # the common level of the integration is kept apart
            scan = int(p["scan"][sel[0]])
            u_rows.extend((a, scan, t, x) for a, x in zip(ants.tolist(), u.tolist(), strict=True))
        if not u_rows:
            return out
        a, s, t, u = (np.array(c) for c in zip(*u_rows, strict=True))
        ant_mean = np.zeros(nant)
        for k in np.unique(a):
            ant_mean[k] = u[a == k].mean()
        r1 = u - ant_mean[a]
        key = a.astype(np.int64) * 100_000 + s
        scan_mean = {k: r1[key == k].mean() for k in np.unique(key)}
        r2 = r1 - np.array([scan_mean[k] for k in key])
        # The integration's common level: the median row lnv per integration, by scan.
        common = np.array([np.median(lnv[tkey == x]) for x in np.unique(t)])
        out.update(
            {
                "ant_track_rms": float(np.std(ant_mean[np.unique(a)])),
                "ant_scan_rms": float(np.std(list(scan_mean.values()))),
                "ant_inscan_rms": float(np.std(r2)),
                "baseline_rms": float(np.std(np.concatenate(e_all))),
                "integration_common_rms": float(np.std(common)),
                "integrations": int(len(np.unique(t))),
            }
        )
        return out


def main(i: int, config_path: Path, out: Path, stride: int | None = None) -> None:
    row = subset().iloc[i]
    win = name(row)
    ms = BASE / f"{win}.ms"
    task = Path(ROOT) / row.task
    config = json.loads(config_path.read_text())
    seed = config.get("seed", 0) * 1000003 + zlib.crc32(win.encode()) % 1000003
    errors = {**config["errors"], **config.get("by_window", {}).get(win, {})}
    final = next((UV / d / f"{i}.json" for d in ("uvfinal3", "uvfinal") if (UV / d / f"{i}.json").exists()), None)
    prev = json.loads(final.read_text()) if final else None
    table = _tables()
    with table(str(ms), ack=False) as t:
        sim = Simulator(ms, task, ErrorModel.from_dict(errors), seed, t)
        nchan = len(sim.chan_freq)
        chans = np.ones(nchan, bool)
        if prev:
            rv, sv = np.array(prev["real"]["channel_variance"]), np.array(prev["config"]["channel_variance"])
            chans = (rv > 0) & (sv > 0) & (rv < 1.05 * sv)
        nrow = t.nrows()
        nchunks = (nrow + CHUNK - 1) // CHUNK
        stride = stride or max(1, nchunks * CHUNK // 1_500_000)
        stats = {"real": Rows(), "sim": Rows()}
        scale = sim.noise["scale"]
        hands = sim.hands
        # Chunks start at integration boundaries, so no integration is split.
        times = t.getcol("TIME")
        bounds = np.r_[0, np.flatnonzero(np.diff(times) != 0) + 1, nrow]
        for c in range(0, nchunks, stride):
            start = int(bounds[np.searchsorted(bounds, c * CHUNK)])
            stop = int(bounds[min(np.searchsorted(bounds, (c + 1) * CHUNK), len(bounds) - 1)])
            n = stop - start
            if n <= 0:
                continue
            real = t.getcol("DATA", startrow=start, nrow=n)[:, :, hands]
            flag = t.getcol("FLAG", startrow=start, nrow=n)[:, :, hands]
            cols, clean, data = sim.chunk(t, start, n)
            weight = cols["weight"][:, hands]
            good = ~flag & (weight[:, None, :] > 0) & chans[None, :, None]
            norm = np.sqrt(np.where(weight > 0, weight, 0) / scale)[:, None, :]
            m = clean[:, :, hands]
            stats["real"].add((real - m) * norm, good, cols)
            stats["sim"].add((data[:, :, hands] - m) * norm, good, cols)
    result = {
        "window": win,
        "source": row.source,
        "channels_used": int(chans.sum()),
        "nchan": nchan,
        "stride": stride,
        **{k: s.result() for k, s in stats.items()},
    }
    out.write_text(json.dumps(result, indent=1))
    # The real residuals per (antenna, channel, hand), for profile().
    real = stats["real"]
    np.savez(out.with_suffix(".npz"), z2=real.ant_z2, n=real.ant_n, chans=chans)


def profile(npz: Path, base_ms: Path) -> None:
    """Each antenna's and hand's noise shape (errors.antenna_noise_shape) from a
    window's uvmix sums, into its base as noise.antenna.npz."""
    with np.load(npz) as f:
        z2, n, chans = f["z2"], f["n"], f["chans"]
    last = int(np.flatnonzero(n.sum(axis=(1, 2)) > 0).max()) + 1
    g, h = antenna_noise_shape(z2[:last], n[:last], chans)
    np.savez(base_ms / "noise.antenna.npz", g=g, h=h)


if __name__ == "__main__" and sys.argv[1] == "profile":
    profile(Path(sys.argv[2]), Path(sys.argv[3]))
elif __name__ == "__main__":
    main(int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else None)
