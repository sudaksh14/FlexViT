for config in $(python3 exp_configs.py $1);
do
    echo $config
    sbatch --output=~/jobs/job_$(date +%F_%H-%M-%S)_%x.txt exp_configs.py $config
done