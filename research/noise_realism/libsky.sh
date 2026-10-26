#!/bin/bash
#SBATCH -J noise-sky -c 8 --mem=64G -t 08:00:00 -p zwickyA-CPU -o /SKAO_data/cycle11/MADIV-28/noise/libsim/logs/%x-%a-%A.out
# Window I observed by the library on a sky made from the real source (libsim.py
# real_sky), imaged as MADIV-16 imaged the real windows and measured as they were.
# Array index t: window t / 3, sky (point, gaussian, mixture)[t % 3]; or I and SKY given.
# MODE=cont: simulate, image the continuum, measure it and the visibilities, keep the MS.
# MODE=cube: image the kept MS with the cube (uvcontsub) and measure everything, then drop the MS.
set -euo pipefail
SKIES=(point gaussian mixture)
T=$(( ${SLURM_ARRAY_TASK_ID:-0} + ${OFFSET:-0} )); I=${I:-$((T / 3))}; SKY=${SKY:-${SKIES[$((T % 3))]}}
N=/SKAO_data/cycle11/MADIV-28/noise; D=$N/libsim
R=$D/runs-sky/$SKY; G=$D/img-sky/$SKY; M=$N/iter-sky-$SKY/metrics; U=$D/uv-sky-$SKY
export CASASITECONFIG=$D/casa-site/config.py MPLCONFIGDIR=$D/.mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
  ALMASIM_SRC=/home1/mdelliveneri/ALMASim-atm/src/almasim
NAME=$(awk -v i=$I '$1 == i {print $2}' $N/diag/index-names.txt)
MODE=${MODE:-cube}
STAMP=$R/w$I.$MODE
if [ -f $STAMP ]; then echo "DONE $I $SKY $MODE (already)"; exit 0; fi
cd /home1/mdelliveneri/ALMASim/research/noise_realism
mkdir -p $R $M $U
if [ ! -f $R/w$I.done ]; then
  rm -rf $R/w$I
  /home1/mdelliveneri/ALMASim-atm/.venv-ci/bin/python libsim.py run "$I" $D/rows.csv $R - $SKY
  touch $R/w$I.done
fi
A=/home1/mdelliveneri/ALMASim/.venv/bin/almasim
P=/home1/mdelliveneri/ALMASim/.venv/bin/python
rm -rf $G/w$I/cubes; mkdir -p $G/w$I
$A image compute-parameters $R/w$I $G/w$I/params.csv --pattern '*.ms'
$A image image-from-ms $G/w$I/params.csv $G/w$I/cubes --backend sync \
  --wsclean-bin /SKAO_data/spack/opt/spack/linux-zen3/wsclean-3.7.20260424-a26g7w3hqrs4ec5ldmd4fjs4gdkg65rp/bin/wsclean \
  --num-cores 8 --task-memory-gb 40 --task-timeout 10800 $([ $MODE = cube ] && echo --cube --uv-contsub || echo --no-cube) \
  --casa-data-root /SKAO_data/cycle11/.casa-data
$A image clean-quality $G/w$I/cubes
$P measure.py $G/w$I/cubes/w${I}_$I/SPW-0_FIELD-0 $M/$NAME.json > /dev/null
if [ $MODE = cont ]; then
  $P libuv.py $R/w$I/w${I}_$I.ms $U/$NAME.json
else
  rm -rf $R/w$I/w${I}_$I.ms  # the images stay; the MS (GBs) goes once measured
fi
touch $STAMP
echo "DONE $I $SKY $MODE"
