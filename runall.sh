#!/usr/bin/bash
mkdir -p ~/jobs
python3 experiments.py listcommand $1 | while read config
do
    echo $config
    sbatch $config
done