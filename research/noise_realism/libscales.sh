#!/bin/bash
#SBATCH -J noise-libscales -c 1 --mem=4G -t 00:20:00 -p zwickyA-CPU -o /SKAO_data/cycle11/MADIV-28/noise/libsim/logs/%x-%A.out
cd /home1/mdelliveneri/ALMASim/research/noise_realism
/home1/mdelliveneri/ALMASim/.venv/bin/python libsim.py scales /SKAO_data/cycle11/MADIV-28/noise/libsim/img-p1 /SKAO_data/cycle11/MADIV-28/noise/libsim/scales.csv
