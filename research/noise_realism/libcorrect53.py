"""Window 53's pass-2 correction, once its 400 GB pass 2 is measured: the same rule
as libsim.rescale, for that one row of scales.csv (the other rows already corrected)."""
import json, sys
from pathlib import Path
import pandas as pd
from libsim import TOLERANCE
D = Path("/SKAO_data/cycle11/MADIV-28/noise/libsim"); I = int(sys.argv[1])
q = D / f"img-p2/w{I}/cubes/w{I}_{I}/SPW-0_FIELD-0/quality.json"
f = pd.read_csv(D / "scales.csv"); k = f.index[f.window == I][0]
achieved = json.loads(q.read_text())["continuum"]["snr"]
f.loc[k, "pass2_snr"] = achieved
redo = bool(achieved and abs(achieved / f.loc[k, "real_snr"] - 1) > TOLERANCE)
if redo:
    f.loc[k, "scale"] *= f.loc[k, "real_snr"] / achieved
f.to_csv(D / "scales.tmp", index=False); (D / "scales.tmp").rename(D / "scales.csv")
print(f"window {I}: S/N {achieved:.3g} against {f.loc[k, 'real_snr']:.3g}; redo {redo}")
