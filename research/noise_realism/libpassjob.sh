#!/bin/bash
#SBATCH -J noise-libpass -c 32 --mem=256G -t 12:00:00 -p zwickyA-CPU -o /SKAO_data/cycle11/MADIV-28/noise/libsim/logs/%x-%a-%A.out
# Four windows of the S/N-matched library check side by side (lib.sh, 8 threads each):
# array index g runs windows 4g .. 4g+3 (or those listed in WINDOWS), pass PASS.
D=/SKAO_data/cycle11/MADIV-28/noise/libsim
G=$SLURM_ARRAY_TASK_ID
LIST=${WINDOWS:-$(seq 0 99)}
for I in $(echo $LIST | tr ' ' '\n' | sed -n "$((4 * G + 1)),$((4 * G + 4))p"); do
  I=$I PASS=$PASS bash $D/lib.sh > $D/logs/pass$PASS-$I.log 2>&1 &
done
wait
echo "group $G pass $PASS done"
