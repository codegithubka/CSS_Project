### Snellius Usage Breakdown

```
ssh <your_username>@snellius.surf.nl

# On a separate terminal run the following

# Upload the entire project directory (including your models/ folder)
scp -r ~/Documents/CSS_Project <your_username>@snellius.surf.nl:~/

# On the Snellius terminal

module load 2023 Python/3.11.3-GCCcore-12.3.0
python3 -m venv ~/css_env
source ~/css_env/bin/activate
pip install numpy scipy matplotlib joblib

# To submit a job

sbatch run_analysis.sh

# Check Queue Status

squeue -u $USER

# Cancel a job

scancel <JOBID>

# Monitoring live progress

tail -f logs_<JOBID>.err

# Fetching the results once the job is done

scp -r <your_username>@snellius.surf.nl:~/results_18514601 ~/Downloads/
```

The jobscript template can be found in ```run_analysis.sh``` (default rome paritition).


Snellius Partitions Page: https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting