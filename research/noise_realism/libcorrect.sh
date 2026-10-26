#!/bin/bash
#SBATCH -J noise-libcorrect -c 1 --mem=4G -t 00:20:00 -p zwickyA-CPU -o /SKAO_data/cycle11/MADIV-28/noise/libsim/logs/%x-%A.out
# Pass 2's correction: windows whose S/N is more than libsim.TOLERANCE off the real
# one get their scale corrected (scales.csv; pass 1's kept as scales-pass1.csv)
# and go back to the front of the worker queue.
set -euo pipefail
D=/SKAO_data/cycle11/MADIV-28/noise/libsim
cd /home1/mdelliveneri/ALMASim/research/noise_realism
[ -f $D/scales-pass1.csv ] || cp $D/scales.csv $D/scales-pass1.csv
/home1/mdelliveneri/ALMASim/.venv/bin/python libsim.py rescale $D/scales-pass1.csv $D/img-p2 $D/scales.csv
for I in $(cat $D/scales.redo); do rm -f $D/runs-p2/w$I.done; done
flock $D/queue.lock bash -c "{ sed 's/^/pass2 /' $D/scales.redo; cat $D/queue.txt; } > $D/queue.new && mv $D/queue.new $D/queue.txt"
echo "queued $(wc -l < $D/scales.redo) windows; queue now $(wc -l < $D/queue.txt)"
