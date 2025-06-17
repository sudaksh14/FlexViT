mkdir -p ~/jobs
for config in $(python3 experiments.py list $1);
do
    echo $config
    sbatch experiment_job.sh $config
done