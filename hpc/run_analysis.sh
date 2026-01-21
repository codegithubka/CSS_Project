#!/bin/bash
#SBATCH --job-name=pp_evo
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=pp_analysis_%j.out
#SBATCH --error=pp_analysis_%j.err

# =============================================================================
# PP Evolutionary Analysis - Snellius Job Script
# =============================================================================
#
# ESTIMATED RESOURCES:
#   - Runtime: ~1.5-2 hours on 32 cores (15×15 grid, 25 reps)
#   - Memory: ~8 GB peak
#   - CPU hours: ~50-60 core-hours
#
# SUBMIT:     sbatch run_analysis.sh
# MONITOR:    squeue -u $USER
# CANCEL:     scancel <job_id>
# OUTPUT:     results/ directory
#
# =============================================================================

echo "========================================"
echo "PP Evolutionary Analysis"
echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Memory:     $SLURM_MEM_PER_NODE"
echo "Start:      $(date)"
echo "Working dir: $(pwd)"
echo "========================================"

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

# Load modules (adjust to your Snellius setup)
source ~/snellius_venv/bin/activate

# If you have a virtual environment with your models package:
# source ~/venvs/ca_analysis/bin/activate

# Prevent numpy/scipy from spawning extra threads (we use joblib instead)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------------------------------------------------------------
# Run Analysis
# -----------------------------------------------------------------------------

OUTPUT_DIR="results_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the full analysis
python3 pp_analysis.py \
    --mode full \
    --output $OUTPUT_DIR \
    --cores $SLURM_CPUS_PER_TASK

# For asynchronous execution (uncomment if needed)
#python3 pp_analysis.py --mode full --output $OUTPUT_DIR --cores $SLURM_CPUS_PER_TASK --async

# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------

echo ""
echo "========================================"
echo "Analysis Complete"
echo "========================================"
echo "End time:   $(date)"
echo "Results in: $OUTPUT_DIR/"
echo ""
echo "Output files:"
ls -lh $OUTPUT_DIR/
echo "========================================"