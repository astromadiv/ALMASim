#!/bin/bash
#SBATCH -J noise-libreport -c 4 --mem=32G -t 02:00:00 -p zwickyA-CPU -o /SKAO_data/cycle11/MADIV-28/noise/libsim/logs/%x-%A.out
# The six-column report (libreport.py) into $OUT (default libsim/report-final).
set -euo pipefail
OUT=${OUT:-/SKAO_data/cycle11/MADIV-28/noise/libsim/report-final}
cd /home1/mdelliveneri/ALMASim/research/noise_realism
mkdir -p $OUT
/home1/mdelliveneri/ALMASim/.venv/bin/python libreport.py $OUT
ls -la $OUT
