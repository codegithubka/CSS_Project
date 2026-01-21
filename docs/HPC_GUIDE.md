### Snellius Usage Breakdown

```
ssh kanagnostopoul@snellius.surf.nl

# On a separate terminal run the following

# Upload the entire project directory (including your models/ folder)
scp -r ~/CSS_Project kanagnostopoul@snellius.surf.nl:~/

# On the Snellius terminal

module load 2023 Python/3.11.3-GCCcore-12.3.0
python3 -m venv ~/css_env
source ~/css_env/bin/activate
pip install numpy scipy matplotlib joblib

# To do a dry run for testing the entire environment

python3 pp_analysis.py --mode full --dry-run

# For async run

python3 pp_analysis.py --mode full --output results_${SLURM_JOB_ID} --cores $SLURM_CPUS_PER_TASK --async

# To submit a job

sbatch run_analysis.sh

# Check Queue Status

squeue -u $USER

# Cancel a job

scancel <JOBID>

# Monitoring live progress

tail -f logs_<JOBID>.err

# Fetching the results once the job is done

scp -r kanagnostopoul@snellius.surf.nl:~/results_18532145 ~/Downloads/
```

The jobscript template can be found in ```run_analysis.sh``` (default rome paritition).


Snellius Partitions Page: https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting