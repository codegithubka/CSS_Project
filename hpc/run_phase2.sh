#!/bin/bash
#SBATCH --job-name=pp_phase2
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/kanagnostopoul/CSS_Project/pp_phase2_%j.out
#SBATCH --error=/home/kanagnostopoul/CSS_Project/pp_phase2_%j.err

# =============================================================================
# PP Hydra Effect - Phase 2: Self-Organization (SOC Test)
# =============================================================================
#
# PHASE 2: Test if prey_death evolves toward critical point
#   - 6 initial prey_death values × 30 reps = 180 simulations
#   - Longer runs (5000 steps) for evolution to equilibrate
#   - Tracks evolved_prey_death_timeseries
#
# SUBMIT:     sbatch run_phase2.sh
# MONITOR:    squeue -u $USER
# CANCEL:     scancel <job_id>
#
# =============================================================================

echo "========================================"
echo "PP Hydra Effect - Phase 2"
echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Start:      $(date)"
echo "Working dir: $(pwd)"
echo "========================================"

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

source ~/snellius_venv/bin/activate

# Prevent numpy/scipy from spawning extra threads (joblib handles parallelism)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------------------------------------------------------------
# Run Phase 2
# -----------------------------------------------------------------------------

OUTPUT_DIR="results/phase2_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Dry run first to verify setup
echo "Dry run check:"
python3 -u scripts/experiments.py \
    --phase 2 \
    --output $OUTPUT_DIR \
    --cores $SLURM_CPUS_PER_TASK \
    --dry-run

echo ""
echo "Starting Phase 2..."
echo ""

# Run phase 2
python3 -u scripts/experiments.py \
    --phase 2 \
    --output $OUTPUT_DIR \
    --cores $SLURM_CPUS_PER_TASK

# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------

echo ""
echo "========================================"
echo "Phase 2 Complete"
echo "========================================"
echo "End time:   $(date)"
echo "Results in: $OUTPUT_DIR/"
echo ""
echo "Output files:"
ls -lh $OUTPUT_DIR/
echo ""
echo "Next steps:"
echo "  1. Download phase2_results.jsonl"
echo "  2. Plot evolved_prey_death_final vs initial prey_death"
echo "  3. Check if all runs converge to ~0.095-0.105 (critical point)"
echo "  4. If SOC confirmed, proceed to Phase 3 (finite-size scaling)"
echo "========================================"