#!/bin/bash
#SBATCH -J noise-lib -c 8 --mem=64G -t 08:00:00 -p zwickyA-CPU -o /SKAO_data/cycle11/MADIV-28/noise/libsim/logs/%x-p%a-%A.out
# Library end to end, one window (array index = window), pass PASS:
#   1: the library's own flux, continuum imaging only (its S/N gives the scale)
#   2: the model cube scaled to the real continuum S/N (scales.csv), imaged as
#      MADIV-16 imaged the real windows, then measured as they were.
set -euo pipefail
I=${I:-$SLURM_ARRAY_TASK_ID}; PASS=${PASS:?PASS=1 or 2}
N=/SKAO_data/cycle11/MADIV-28/noise; D=$N/libsim
R=$D/runs-p$PASS; G=$D/img-p$PASS
export CASASITECONFIG=$D/casa-site/config.py MPLCONFIGDIR=$D/.mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  ALMASIM_SRC=/home1/mdelliveneri/ALMASim-atm/src/almasim
NAME=$(awk -v i=$I '$1 == i {print $2}' $N/diag/index-names.txt)
cd /home1/mdelliveneri/ALMASim/research/noise_realism
TASK=$G/w$I/cubes/w${I}_$I/SPW-0_FIELD-0
if [ "$PASS" = 1 ] && [ -f $TASK/quality.json ]; then echo "DONE $I pass 1 (already)"; exit 0; fi
SCALES=; [ "$PASS" = 2 ] && SCALES=$D/scales.csv
# A run is redone unless it finished (its log line in the run's done file).
if [ ! -f $R/w$I.done ]; then
  rm -rf $R/w$I
  /home1/mdelliveneri/ALMASim-atm/.venv-ci/bin/python libsim.py run "$I" $D/rows.csv $R $SCALES
  touch $R/w$I.done
fi
A=/home1/mdelliveneri/ALMASim/.venv/bin/almasim
P=/home1/mdelliveneri/ALMASim/.venv/bin/python
rm -rf $G/w$I/cubes; mkdir -p $G/w$I
IMAGING=(--no-cube); [ "$PASS" = 2 ] && IMAGING=(--cube --uv-contsub)
$A image compute-parameters $R/w$I $G/w$I/params.csv --pattern '*.ms'
$A image image-from-ms $G/w$I/params.csv $G/w$I/cubes --backend sync \
  --wsclean-bin /SKAO_data/spack/opt/spack/linux-zen3/wsclean-3.7.20260424-a26g7w3hqrs4ec5ldmd4fjs4gdkg65rp/bin/wsclean \
  --num-cores 8 --task-memory-gb 40 --task-timeout 10800 "${IMAGING[@]}" --casa-data-root /SKAO_data/cycle11/.casa-data
$A image clean-quality $G/w$I/cubes
if [ "$PASS" = 2 ]; then
  mkdir -p $N/iter-lib/metrics $D/uv-p2
  $P measure.py $TASK $N/iter-lib/metrics/$NAME.json > /dev/null
  $P libuv.py $R/w$I/w${I}_$I.ms $D/uv-p2/$NAME.json
else
  rm -rf $R/w$I/w${I}_$I.ms
fi
echo "DONE $I pass $PASS"
