"""The receiver temperature each window's measured Tsys implies under the
library's Tsys model (linear in Trx), at the real elevation and PWV.

    python libtrx.py TSYS.csv
"""
import sys, types
from pathlib import Path
import numpy as np, pandas as pd
for _pkg in ("almasim", "almasim.services", "almasim.services.interferometry"):
    _m = types.ModuleType(_pkg); _m.__path__ = ["/home1/mdelliveneri/ALMASim-atm/src/" + _pkg.replace(".", "/")]; sys.modules[_pkg] = _m
from almasim.services.interferometry.atmosphere import system_temperature
from casacore.tables import table
NOISE = Path("/SKAO_data/cycle11/MADIV-28/noise")
d = pd.read_csv(sys.argv[1])
names = dict(l.split() for l in open(NOISE / "diag" / "index-names.txt"))
rows = pd.read_csv(NOISE / "libsim" / "rows.csv").set_index("window")
need = []
for _, r in d.iterrows():
    with table(str(NOISE / "base" / f"{names[str(int(r.window))]}.ms" / "SPECTRAL_WINDOW"), ack=False) as sw:
        f = np.asarray(sw.getcell("CHAN_FREQ", 0), dtype=float)
    am = 1 / np.sin(np.radians(r.el_real))
    t0 = np.median(system_temperature(f, np.full(f.size, 0.0), pwv_mm=r.pwv, airmass=am))
    t1 = np.median(system_temperature(f, np.full(f.size, 100.0), pwv_mm=r.pwv, airmass=am))
    need.append((r.tsys_real - t0) / ((t1 - t0) / 100))
d["trx_needed"] = need
print(d.groupby(["band", "array"]).agg(n=("window", "size"), trx_lib=("trx_lib", "median"), trx_needed=("trx_needed", "median"), q16=("trx_needed", lambda x: x.quantile(.16)), q84=("trx_needed", lambda x: x.quantile(.84))).round(1).to_string())
d.to_csv(sys.argv[1].replace(".csv", "-trx.csv"), index=False)
