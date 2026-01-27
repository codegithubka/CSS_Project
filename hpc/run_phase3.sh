#!/bin/bash
#SBATCH --job-name=pp_phase3
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/kanagnostopoul/CSS_Project/pp_phase3_%j.out
#SBATCH --error=/home/kanagnostopoul/CSS_Project/pp_phase3_%j.err

# =============================================================================
# PP Hydra Effect - Phase 3: Finite-Size Scaling
# =============================================================================
#
# PHASE 3: Test finite-size scaling at critical point
#   - Grid sizes: 50, 100, 250, 500, 1000
#   - 20 replicates per size = 100 simulations
#   - Cluster size distributions for power-law analysis
#
# SUBMIT:     sbatch run_phase3.sh
# MONITOR:    squeue -u $USER
# CANCEL:     scancel <job_id>
#
# =============================================================================

cd /home/kanagnostopoul/CSS_Project || exit 1

echo "========================================"
echo "PP Hydra Effect - Phase 3"
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
# Run Phase 3
# -----------------------------------------------------------------------------
OUTPUT_DIR="results/phase3_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Dry run first to verify setup
echo "Dry run check:"
python3 -u scripts/experiments.py \
    --phase 3 \
    --output $OUTPUT_DIR \
    --cores $SLURM_CPUS_PER_TASK \
    --dry-run

echo ""
echo "Starting Phase 3..."
echo ""

# Run phase 3
python3 -u scripts/experiments.py \
    --phase 3 \
    --output $OUTPUT_DIR \
    --cores $SLURM_CPUS_PER_TASK

# -----------------------------------------------------------------------------
# Completion
# -----------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Phase 3 Complete"
echo "========================================"
echo "End time:   $(date)"
echo "Results in: $OUTPUT_DIR/"
echo ""
echo "Output files:"
ls -lh $OUTPUT_DIR/
echo ""
echo "Next steps:"
echo "  1. Download phase3_results.jsonl"
echo "  2. Analyze cluster size distributions P(s) for each grid size"
echo "  3. Fit power-law exponent tau from P(s) ~ s^(-tau)"
echo "  4. Check finite-size cutoff s_max ~ L^D (fractal dimension)"
echo "========================================"