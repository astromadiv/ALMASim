#!/bin/bash
# Release the held worker jobs one by one while fewer than CAP of this user's jobs run.
# Job management only; no compute here.
CAP=${CAP:-8}; D=/SKAO_data/cycle11/MADIV-28/noise/libsim
while squeue -u $USER -h -t PD -n noise-worker -o "%r" | grep -q JobHeld; do
  running=$(squeue -u $USER -h -t R | wc -l)
  if [ $running -lt $CAP ]; then
    j=$(squeue -u $USER -h -t PD -n noise-worker -o "%i %r" | awk '$2 ~ /JobHeld/ {print $1; exit}')
    scontrol release $j && echo "$(date +%H:%M) released $j ($running running)"
  fi
  sleep 60
done
echo "$(date +%H:%M) all workers released"
