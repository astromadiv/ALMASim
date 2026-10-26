#!/bin/bash
#SBATCH -J noise-worker -c 32 --mem=256G -t 3-00:00:00 -p zwickyA-CPU -o /SKAO_data/cycle11/MADIV-28/noise/libsim/logs/%x-%A.out
# Four workers, each taking the next task from queue.txt (under a lock) until it is empty:
#   "pass2 I"  the S/N-matched window I, pass 2 (lib.sh)
#   "sky t"    sky run t, cube stage (sky.sh MODE=cube)
D=/SKAO_data/cycle11/MADIV-28/noise/libsim
SKIES=(point gaussian mixture)
work() {
  while true; do
    task=$(flock $D/queue.lock bash -c "head -1 $D/queue.txt; sed -i 1d $D/queue.txt")
    [ -n "$task" ] || break
    set -- $task
    case $1 in
      pass2) I=$2 PASS=2 bash $D/lib.sh > $D/logs/pass2-$2.log 2>&1 ;;
      sky) I=$(($2 / 3)) SKY=${SKIES[$(($2 % 3))]} MODE=cube bash $D/sky.sh > $D/logs/sky-cube-$2.log 2>&1 ;;
    esac
    echo "$(date +%H:%M) $task exit $?"
  done
}
for w in 1 2 3 4; do work & sleep 5; done
wait
echo "queue empty"
