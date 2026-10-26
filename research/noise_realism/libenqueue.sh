#!/bin/bash
#SBATCH -J noise-enqueue -c 1 --mem=1G -t 00:05:00 -p zwickyA-CPU -o /SKAO_data/cycle11/MADIV-28/noise/libsim/logs/%x-%A.out
# Once the scales exist: pass 2 of every window first in the queue.
D=/SKAO_data/cycle11/MADIV-28/noise/libsim
[ -s $D/scales.csv ] || { echo "no scales"; exit 1; }
flock $D/queue.lock bash -c "{ tail -n +2 $D/scales.csv | cut -d, -f1 | sed 's/^/pass2 /'; cat $D/queue.txt; } > $D/queue.new && mv $D/queue.new $D/queue.txt"
grep -c pass2 $D/queue.txt
