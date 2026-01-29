#!/bin/bash
#SBATCH --job-name=pp_phase6
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --mem=0
#SBATCH --output=pp_phase6_%j.out
#SBATCH --error=pp_phase6_%j.err

# =============================================================================
# PP Hydra Effect - Phase 6: Directed Hunting 4D Sweep
# =============================================================================
#
# PHASE 6: Full 4D parameter sweep with directed hunting enabled
#   - Same structure as Phase 4 but with directed_hunting=True
#   - 11^4 × 10 reps = 146,410 simulations
#   - Grid size: 250
#   - Collects time series for comparison with Phase 4
#   - Estimated runtime: ~4-6 hours on 128 cores
#   - Memory: mem=0 (use all available node memory)
#
# PURPOSE: Test if Hydra effect and SOC persist under directed hunting
#
# SUBMIT:     sbatch run_phase6.sh
# MONITOR:    squeue -u $USER
# CANCEL:     scancel <job_id>
#
# =============================================================================

echo "========================================"
echo "PP Hydra Effect - Phase 6"
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
# Run Phase 6
# -----------------------------------------------------------------------------

OUTPUT_DIR="results/phase6_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Dry run first to verify setup
echo "Dry run check:"
python3 -u scripts/experiments.py \
    --phase 6 \
    --output $OUTPUT_DIR \
    --cores $SLURM_CPUS_PER_TASK \
    --dry-run

echo ""
echo "Starting Phase 6..."
echo ""

# Run phase 6
python3 -u scripts/experiments.py \
    --phase 6 \
    --output $OUTPUT_DIR \
    --cores $SLURM_CPUS_PER_TASK

# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------

echo ""
echo "========================================"
echo "Phase 6 Complete"
echo "========================================"
echo "End time:   $(date)"
echo "Results in: $OUTPUT_DIR/"
echo ""
echo "Output files:"
ls -lh $OUTPUT_DIR/
echo ""
echo "Next steps:"
echo "  1. Download phase6_results.jsonl"
echo "  2. Compare with Phase 4 results (random hunting baseline)"
echo "  3. Analyze if Hydra effect persists under directed hunting"
echo "  4. Compare critical point locations between Phase 4 and Phase 6"
echo "  5. Check for differences in SOC signatures"
echo "========================================"