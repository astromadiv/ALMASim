#!/bin/bash
#SBATCH -J noise-window -c 16 --mem=64G -t 08:00:00 -p zwickyA-CPU
# One window of one iteration: its counterpart (or, in a control iteration,
# the real split data), imaged exactly as MADIV-16 imaged the real one, then
# assessed (clean-quality) and measured.
# CUBE_ARGS (default "--cube --uv-contsub"; empty: continuum only) and IMAGE_EXTRA
# (more image-from-ms options) change the imaging, e.g. for a deeper model of the real sky.
#   sbatch --array=0-19 -o ITER/logs/%a.log window.sh ITER
set -euo pipefail
ITER=$1; I=${SLURM_ARRAY_TASK_ID:-$2}
R=/home1/mdelliveneri/ALMASim; PY=$R/.venv/bin/python; N=$R/research/noise_realism
ALMASIM=$R/.venv/bin/almasim
export OPENBLAS_NUM_THREADS=1  # WSClean refuses a multi-threaded OpenBLAS
eval "$($PY $N/window.py info "$I")"
BASE=/SKAO_data/cycle11/MADIV-28/noise/base/$NAME.ms
mkdir -p "$ITER"/{params,cubes,metrics,ms}
# One base per window, made once even when several iterations start together.
(
    flock 9
    [ -f "$BASE/noise.json" ] || $PY $N/counterpart.py base "$MS" "$SPW" "$FIELD" "$BASE"
) 9> "/SKAO_data/cycle11/MADIV-28/noise/base/$NAME.lock"
MODE=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['mode'])" "$ITER/config.json")
if [ "$MODE" = control ]; then
    DATA=$BASE
else
    DATA=$ITER/ms/$NAME.ms
    SEED=$($PY -c "import json,sys,zlib; c=json.load(open(sys.argv[1])); print(c.get('seed',0)*1000003 + zlib.crc32(sys.argv[2].encode()) % 1000003)" "$ITER/config.json" "$NAME")
    # The iteration's errors, with this window's own values (by_window) over them.
    $PY -c "import json,sys; c=json.load(open(sys.argv[1])); e=dict(c['errors']); e.update(c.get('by_window',{}).get(sys.argv[3],{})); json.dump(e, open(sys.argv[2],'w'))" "$ITER/config.json" "$ITER/ms/$NAME.errors.json" "$NAME"
    $PY $N/counterpart.py make "$BASE" "$TASK" "$DATA" --errors "$ITER/ms/$NAME.errors.json" --seed "$SEED"
fi
$PY $N/window.py params "$I" "$DATA" > "$ITER/params/$NAME.csv"
# Self-calibration as MADIV-16's, unless the config says otherwise ("" for none).
SELFCAL=$($PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('selfcal', 'inf,30s,ap:inf'))" "$ITER/config.json")
$ALMASIM image image-from-ms "$ITER/params/$NAME.csv" "$ITER/cubes" \
    --wsclean-bin /SKAO_data/spack/opt/spack/linux-zen3/wsclean-3.7.20260424-a26g7w3hqrs4ec5ldmd4fjs4gdkg65rp/bin/wsclean \
    --num-cores "${SLURM_CPUS_PER_TASK:-8}" --task-memory-gb 40 --task-timeout "${TASK_TIMEOUT:-25200}" --backend sync \
    ${CUBE_ARGS---cube --uv-contsub} ${IMAGE_EXTRA:-} ${SELFCAL:+--selfcal "$SELFCAL"} --casa-data-root /SKAO_data/cycle11/.casa-data
OUT=$(ls -d "$ITER"/cubes/"$(basename "$DATA" .ms)"*/SPW-0_FIELD-0 | head -1)
$ALMASIM image clean-quality "$OUT"
$PY $N/measure.py "$OUT" "$ITER/metrics/$NAME.json" > /dev/null
[ "$MODE" = control ] || rm -rf "$DATA"
echo "DONE $NAME"
