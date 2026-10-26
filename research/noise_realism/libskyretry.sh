#!/bin/bash
#SBATCH -J noise-skyretry -c 8 --mem=400G -t 12:00:00 -p zwickyA-CPU -o /SKAO_data/cycle11/MADIV-28/noise/libsim/logs/%x-%A.out
# The sky runs killed for memory, one at a time with the memory to themselves.
SKIES=(point gaussian mixture)
for t in $TASKS; do
  I=$((t / 3)) SKY=${SKIES[$((t % 3))]} MODE=${MODE:-cont} bash /SKAO_data/cycle11/MADIV-28/noise/libsim/sky.sh > /SKAO_data/cycle11/MADIV-28/noise/libsim/logs/sky-${MODE:-cont}-$t.log 2>&1
done
