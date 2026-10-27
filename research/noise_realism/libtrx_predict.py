"""The 100 windows' library/real noise ratio predicted with a recalibrated Trx
table and a Ruze aperture efficiency, from the measured ratio and the model's
own Tsys and efficiency (at transit, as the library observes)."""
import sys, types
from pathlib import Path
import numpy as np, pandas as pd
for _pkg in ("almasim", "almasim.services", "almasim.services.interferometry"):
    _m = types.ModuleType(_pkg); _m.__path__ = ["/home1/mdelliveneri/ALMASim-atm/src/" + _pkg.replace(".", "/")]; sys.modules[_pkg] = _m
from almasim.services.interferometry.atmosphere import system_temperature
from almasim.services.interferometry.noise import estimate_receiver_temperature_k
from casacore.tables import table
NOISE = Path("/SKAO_data/cycle11/MADIV-28/noise")
TRX = {1: 32.0, 3: 39.0, 4: 37.0, 5: 50.0, 6: 46.0, 7: 63.0}
RUZE = {"12m": (0.76, 27e-6), "7m": (0.72, 24e-6)}
d = pd.read_csv(sys.argv[1])
names = dict(l.split() for l in open(NOISE / "diag" / "index-names.txt"))
new = []
for _, r in d.iterrows():
    with table(str(NOISE / "base" / f"{names[str(int(r.window))]}.ms" / "SPECTRAL_WINDOW"), ack=False) as sw:
        f = np.asarray(sw.getcell("CHAN_FREQ", 0), dtype=float)
    am = 1 / np.sin(np.radians(r.el_transit))
    old = np.median(system_temperature(f, estimate_receiver_temperature_k(f), pwv_mm=r.pwv, airmass=am))
    trx = np.full(f.size, TRX.get(int(r.band), float(np.median(estimate_receiver_temperature_k(f)))))
    tnew = np.median(system_temperature(f, trx, pwv_mm=r.pwv, airmass=am))
    e0, eps = RUZE[r["array"]]
    eta_new = 0.96 * 0.88 * e0 * np.exp(-(4 * np.pi * eps * f.mean() / 3e8) ** 2)
    new.append(r.noise_ratio * (tnew / old) * (r.eta_lib / eta_new))
d["noise_ratio_new"] = new
for k in ("noise_ratio", "noise_ratio_new"):
    x = d[k]
    print(k, "median", round(x.median(), 3), "q16-84", x.quantile([.16, .84]).round(3).tolist(), "within 10%", int((abs(x - 1) <= .1).sum()), "within 20%", int((abs(x - 1) <= .2).sum()))
print(d.groupby(["band", "array"])[["noise_ratio", "noise_ratio_new"]].median().round(3).to_string())
