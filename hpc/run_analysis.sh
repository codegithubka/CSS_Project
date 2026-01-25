#!/bin/bash
#SBATCH --job-name=pp_phase1
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=pp_phase1_%j.out
#SBATCH --error=pp_phase1_%j.err

# =============================================================================
# PP Hydra Effect - Phase 1: Parameter Sweep
# =============================================================================
#
# PHASE 1: Find critical point via 2D sweep of prey_birth × prey_death
#   - 15 × 15 × 15 reps × 2 (evo/non-evo) = 6,750 simulations
#   - Estimated runtime: ~2 min on 32 cores
#   - Memory: ~8 GB peak
#
# SUBMIT:     sbatch run_phase1.sh
# MONITOR:    squeue -u $USER
# CANCEL:     scancel <job_id>
#
# =============================================================================

echo "========================================"
echo "PP Hydra Effect - Phase 1"
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
# Run Phase 1
# -----------------------------------------------------------------------------

OUTPUT_DIR="results/phase1_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Dry run first to verify setup
echo "Dry run check:"
python3 -u scripts/experiments.py \
    --phase 1 \
    --output $OUTPUT_DIR \
    --cores $SLURM_CPUS_PER_TASK \
    --dry-run

echo ""
echo "Starting Phase 1..."
echo ""

# Run phase 1
python3 -u scripts/experiments.py \
    --phase 1 \
    --output $OUTPUT_DIR \
    --cores $SLURM_CPUS_PER_TASK

# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------

echo ""
echo "========================================"
echo "Phase 1 Complete"
echo "========================================"
echo "End time:   $(date)"
echo "Results in: $OUTPUT_DIR/"
echo ""
echo "Output files:"
ls -lh $OUTPUT_DIR/
echo ""
echo "Next steps:"
echo "  1. Download phase1_results.jsonl"
echo "  2. Run analysis.py to find critical point"
echo "  3. Update config.py with critical_prey_birth/death"
echo "  4. Run phase 2-5"
echo "========================================"