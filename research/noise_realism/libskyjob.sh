#!/bin/bash
#SBATCH -J noise-sky -c 32 --mem=256G -t 12:00:00 -p zwickyA-CPU -o /SKAO_data/cycle11/MADIV-28/noise/libsim/logs/%x-%a-%A.out
# Four sky runs side by side (sky.sh, 8 threads each): array index g runs t = 4g .. 4g+3
# (window t / 3, sky (point, gaussian, mixture)[t % 3]). MODE=cont or cube (sky.sh).
D=/SKAO_data/cycle11/MADIV-28/noise/libsim
G=$SLURM_ARRAY_TASK_ID
for t in $((4 * G)) $((4 * G + 1)) $((4 * G + 2)) $((4 * G + 3)); do
  [ $t -lt 300 ] || continue
  SKIES=(point gaussian mixture)
  I=$((t / 3)) SKY=${SKIES[$((t % 3))]} MODE=${MODE:-cont} bash $D/sky.sh > $D/logs/sky-$MODE-$t.log 2>&1 &
done
wait
echo "group $G done"
