#!/bin/bash
#SBATCH --job-name=pp_phase4
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --output=/home/kanagnostopoul/CSS_Project/pp_phase4_%j.out
#SBATCH --error=/home/kanagnostopoul/CSS_Project/pp_phase4_%j.err

# =============================================================================
# PP Hydra Effect - Phase 4: Global Sensitivity Analysis
# =============================================================================
#
# PHASE 4: Full 4D Parameter Sweep (Global Sensitivity)
#   - Parameters: prey_birth, prey_death, pred_birth, pred_death
#   - Sweep: 0.0 to 1.0 (11 values each) = 14,641 combinations
#   - Replicates: 10 per combination
#   - Total Simulations: ~146,410
#   - Grid Size: 250x250
#
# SUBMIT:     sbatch run_phase4.sh
# MONITOR:    squeue -u $USER
# CANCEL:     scancel <job_id>
#
# =============================================================================

cd /home/kanagnostopoul/CSS_Project || exit 1

echo "========================================"
echo "PP Hydra Effect - Phase 4"
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
# Run Phase 4
# -----------------------------------------------------------------------------
OUTPUT_DIR="results/phase4_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Dry run first to verify setup and runtime estimate
echo "Dry run check:"
python3 -u scripts/experiments.py \
    --phase 4 \
    --output $OUTPUT_DIR \
    --cores $SLURM_CPUS_PER_TASK \
    --dry-run

echo ""
echo "Starting Phase 4 (4D Sweep)..."
echo ""

# Run phase 4
python3 -u scripts/experiments.py \
    --phase 4 \
    --output $OUTPUT_DIR \
    --cores $SLURM_CPUS_PER_TASK

# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Phase 4 Complete"
echo "========================================"
echo "End time:   $(date)"
echo "Results in: $OUTPUT_DIR/"
echo ""
echo "Output files:"
ls -lh $OUTPUT_DIR/
echo ""
echo "Next steps:"
echo "  1. Download phase4_results.jsonl"
echo "  2. Perform Global Sensitivity Analysis (Sobol Indices)"
echo "  3. Identify parameter dominance for extinction events"
echo "  4. Plot parameter heatmaps for predator/prey survival"
echo "========================================"