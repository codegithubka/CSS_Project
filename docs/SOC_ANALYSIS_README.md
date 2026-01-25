# Self-Organized Criticality (SOC) Analysis - Summary

## File Created
**Location:** `scripts/soc_analysis.py`

## Overview
This comprehensive Python analysis script tests whether your prey-predator cellular automaton exhibits **self-organized criticality** (SOC), with focus on perturbations from initial configurations and diverse parameter sampling.

## Key Features

### 1. **Four SOC Properties Analyzed**
   - **Slow Drive:** Gradual parameter drift without immediate release
   - **Stress Build-up:** Interface-based metric tracking potential energy accumulation
   - **Intermittent Release:** Detection of avalanche cascades in population dynamics
   - **Self-Organization:** Robustness across diverse parameter combinations

### 2. **Parameter Variations** (Beyond just death/birth rates)
   - Grid sizes: 16×16 to 64×64
   - Initial densities: prey (0.1–0.4), predator (0.02–0.15)
   - Neighborhood types: Neumann & Moore
   - Update modes: Synchronous & Asynchronous
   - Rate parameters: randomly varied across valid ranges

### 3. **Metrics Computed**
   - **Stress Metric:** Normalized count of (predator/prey)↔empty adjacent pairs
     - Represents friction and interface gradient (potential energy)
   - **Avalanche Detection:** Population change magnitude thresholds
   - **Population Variance:** Rolling window variance of prey/predator counts
   - **Robustness Metrics:** 
     - Avalanche count mean/std across configurations
     - Magnitude consistency
     - Coefficient of variation (measures criticality robustness)

### 4. **Perturbation Experiment Design**
Each experiment runs 230 total steps:
- **Equilibration phase (0–80 steps):** System reaches quasi-steady state
  - Stress accumulates during slow drive
  - No parameter perturbation
- **Observation phase (80–230 steps):** Gradual parameter drift
  - Predator death rate increases by +0.05 (slow drive)
  - System responds with cascading events if critical
  - Stress release and avalanche events detected

## Visualization Output

The script generates `soc_analysis_results.png` with a **2×2 grid displaying the 4 core SOC properties**:

1. **Panel 1 (Top-Left) - Slow Drive:** Gradual parameter drift over time with equilibration and perturbation phases marked
2. **Panel 2 (Top-Right) - Build-up of Stress:** Stress accumulation with avalanche event thresholds marked as orange stars
3. **Panel 3 (Bottom-Left) - Intermittent Release:** Prey and predator population dynamics showing cascade events during perturbation
4. **Panel 4 (Bottom-Right) - Self-Organization:** Stress-density relation across diverse configurations, colored by avalanche activity

## Usage

```bash
python scripts/soc_analysis.py
```

Output:
- Console report with findings
- PNG visualization saved to workspace root: `soc_analysis_results.png`

## Key Observations from Test Run

- **8 diverse configurations** sampled with varied grid sizes, densities, neighborhoods
- **Avalanche detection:** 1/8 experiments showed clear cascade events
- **Stress persistence:** Mean stress ~0.1529 across all configurations
- **Robustness metric:** Coefficient of Variation = 2.646 (indicates some parameter-dependence; lower values →  more robust SOC)
- **Population variance:** Consistent across runs (signature of intermittent release mechanism)

## Code Structure

### Main Functions
- `compute_grid_stress()` – Interface-based stress metric
- `compute_population_variance()` – Rolling window variance calculation
- `detect_avalanche_events()` – Identify cascading population changes
- `sample_parameter_configurations()` – Generate diverse parameter sets
- `run_soc_perturbation_experiment()` – Single experiment with slow drive
- `analyze_soc_robustness()` – Cross-configuration robustness metrics
- `visualize_soc_properties()` – Comprehensive 8-panel figure
- `main()` – Orchestrates full analysis pipeline

### Configuration Space
- **Grid size:** Affects stability and relaxation dynamics
- **Densities:** Controls predator-prey interaction frequency
- **Neighborhood:** Changes spatial coupling strength
- **Rates:** Direct influence on birth/death thresholds

## Scientific Interpretation

The analysis tests the hypothesis:
> *"Does the prey-predator CA exhibit self-organized criticality independent of specific parameter choices?"*

If coefficient of variation is **low** (< 1.0) → SOC is **robust** (self-organized)
If coefficient of variation is **high** (> 1.0) → Behavior is **parameter-dependent** (requires tuning)

---

**Created:** January 2026
**Framework:** NumPy, Matplotlib, custom CA simulation
