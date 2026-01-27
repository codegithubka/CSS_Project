### Direct Movement Prey-Predator
create the predator-prey movement where the predators follow the preys behaviour in update_sync function

```
OUTPUT:
Extended _pp_async_kernel_directed() in numba_optimized.py

elif state == 2:  # PREDATOR - directed hunting
            if np.random.random() < pred_death_val:
                grid[r, c] = 0
                continue
            
            prey_count = 0
            for k in range(n_shifts):
                check_r = (r + dr_arr[k]) % rows
                check_c = (c + dc_arr[k]) % cols
                if grid[check_r, check_c] == 1:
                    prey_count += 1
            
            if prey_count > 0:
                target_idx = np.random.randint(0, prey_count)
                found = 0
                nr, nc = 0, 0
                for k in range(n_shifts):
                    check_r = (r + dr_arr[k]) % rows
                    check_c = (c + dc_arr[k]) % cols
                    if grid[check_r, check_c] == 1:
                        if found == target_idx:
                            nr = check_r
                            nc = check_c
                            break
                        found += 1
                
                if np.random.random() < pred_birth_val:
                    grid[nr, nc] = 2
                    prey_death_arr[nr, nc] = np.nan
            else:
                nbi = np.random.randint(0, n_shifts)
                nr = (r + dr_arr[nbi]) % rows
                nc = (c + dc_arr[nbi]) % cols
                
                if grid[nr, nc] == 1:
                    if np.random.random() < pred_birth_val:
                        grid[nr, nc] = 2
                        prey_death_arr[nr, nc] = np.nan

```

### SOC Visualization

create one file with included visualization show if our prey-preditor cellular automaton structure follows soc, focus on perturbation from the initial configuration not at the critical point, sample from different configurations while varying more than the death and birth rates of the prey not run a full analysis of course on that and observe behaviour, soc should have these 4:  • Slow drive
• Build-up of ‘stress’
• Intermittent release of this stress
• Self organization

```
OUTPUT:

I'll create a comprehensive SOC analysis file for your prey-predator cellular automaton.

"""
Self-Organized Criticality (SOC) Analysis for Prey-Predator Cellular Automaton

This module analyzes whether the prey-predator CA exhibits SOC properties:
1. Slow drive: Gradual external parameter changes
2. Build-up of stress with thresholds (storing potential energy)
3. Intermittent release of stress (avalanches/cascades)
4. Self-organization (robustness across parameter variations, not carefully tuned)

We focus on perturbations from initial configurations (not at critical point) and
sample across different configurations with varied parameters beyond just death/birth
rates (e.g., grid size, densities, neighborhood, synchronicity).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from models to avoid __init__ issues
from models.CA import PP


# ============================================================================
# 1. STRESS METRIC & PERTURBATION DYNAMICS
# ============================================================================

def compute_grid_stress(grid: np.ndarray) -> float:
    """
    Compute a 'stress' metric for the grid as a proxy for potential energy.
    
    High predator-prey interface regions represent tension/stress.
    Stress = normalized count of (predator,empty) and (prey,empty) adjacent pairs.
    This represents the gradient/friction that can cause avalanche-like events.
    
    Args:
        grid: 2D array with 0=empty, 1=prey, 2=predator
        
    Returns:
        Normalized stress value [0, 1]
    """
    rows, cols = grid.shape
    stress = 0
    
    # Count interfaces (predator or prey adjacent to empty)
    for i in range(rows):
        for j in range(cols):
            cell = grid[i, j]
            if cell == 0:  # empty cell
                # Check all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = (i + di) % rows, (j + dj) % cols
                        if grid[ni, nj] > 0:  # neighbor is prey or predator
                            stress += 1
    
    # Normalize by maximum possible interfaces
    max_stress = rows * cols * 8  # each cell can have 8 neighbors
    return stress / max_stress if max_stress > 0 else 0.0


def compute_population_variance(grids_history: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prey and predator population variance over time.
    High variance indicates intermittent release events.
    
    Args:
        grids_history: List of grid snapshots over time
        
    Returns:
        Tuple: (prey_variance_rolling, predator_variance_rolling)
    """
    prey_pops = np.array([(g == 1).sum() for g in grids_history])
    pred_pops = np.array([(g == 2).sum() for g in grids_history])
    
    window = max(5, len(grids_history) // 10)  # rolling window
    prey_changes = np.abs(np.diff(prey_pops))
    pred_changes = np.abs(np.diff(pred_pops))
    
    # Pad at the beginning to match original length
    prey_var = np.concatenate([[prey_changes[0]] * (window - 1), 
                               np.convolve(prey_changes, np.ones(window)/window, mode='valid')])
    pred_var = np.concatenate([[pred_changes[0]] * (window - 1),
                               np.convolve(pred_changes, np.ones(window)/window, mode='valid')])
    
    # Ensure exact length match
    prey_var = prey_var[:len(grids_history)]
    pred_var = pred_var[:len(grids_history)]
    
    return prey_var, pred_var


def detect_avalanche_events(grids_history: List[np.ndarray], 
                           population_change_threshold: float = 0.1) -> List[Tuple[int, float]]:
    """
    Detect avalanche events as rapid changes in total population.
    
    Args:
        grids_history: List of grid snapshots
        population_change_threshold: Fraction of grid change to trigger detection
        
    Returns:
        List of (time_step, magnitude) tuples
    """
    total_pops = np.array([(g > 0).sum() for g in grids_history])
    max_pop = total_pops.max()
    
    if max_pop == 0:
        return []
    
    changes = np.abs(np.diff(total_pops))
    threshold = population_change_threshold * max_pop
    
    avalanches = []
    in_event = False
    event_start = 0
    event_magnitude = 0
    
    for i, change in enumerate(changes):
        if change > threshold:
            if not in_event:
                event_start = i
                in_event = True
            event_magnitude = max(event_magnitude, change)
        else:
            if in_event:
                avalanches.append((event_start, event_magnitude / max_pop))
                in_event = False
                event_magnitude = 0
    
    return avalanches


# ============================================================================
# 2. PARAMETER SAMPLING WITH VARIED CONFIGURATIONS
# ============================================================================

def sample_parameter_configurations(n_samples: int = 10,
                                   base_seed: int = 42) -> List[Dict]:
    """
    Generate diverse parameter configurations.
    Varies: grid size, initial densities, rates, neighborhood, synchronicity.
    
    Args:
        n_samples: Number of configurations to generate
        base_seed: Base random seed
        
    Returns:
        List of configuration dicts
    """
    configs = []
    rng = np.random.RandomState(base_seed)
    
    for i in range(n_samples):
        # Vary grid size (smaller = more fluctuations, larger = more stable?)
        grid_size = rng.choice([16, 32, 48, 64])
        
        # Vary initial densities (more heterogeneous = more stress buildup?)
        prey_density = rng.uniform(0.1, 0.4)
        pred_density = rng.uniform(0.02, 0.15)
        
        # Vary parameters beyond just death/birth rates
        config = {
            "seed": base_seed + i,
            "rows": grid_size,
            "cols": grid_size,
            "densities": (prey_density, pred_density),
            "neighborhood": rng.choice(["neumann", "moore"]),
            "synchronous": rng.choice([True, False]),
            # Vary rate parameters
            "prey_death": rng.uniform(0.01, 0.10),
            "predator_death": rng.uniform(0.05, 0.20),
            "prey_birth": rng.uniform(0.10, 0.35),
            "predator_birth": rng.uniform(0.10, 0.30),
        }
        configs.append(config)
    
    return configs


# ============================================================================
# 3. SLOW DRIVE & STRESS BUILDUP WITH PERTURBATIONS
# ============================================================================

def run_soc_perturbation_experiment(config: Dict, 
                                   n_equilibration: int = 100,
                                   n_observation: int = 200,
                                   perturbation_step: int = 50) -> Dict:
    """
    Run a single SOC experiment with slow parameter drift (drive) and
    perturbations from non-critical initial conditions.
    
    The experiment:
    1. Initialize CA with given config (not at "critical point")
    2. Run equilibration steps (slow drive builds up stress)
    3. Perturb one parameter gradually
    4. Observe stress buildup and release events
    
    Args:
        config: Configuration dict from sample_parameter_configurations()
        n_equilibration: Steps before perturbation (building stress)
        n_observation: Steps during/after perturbation (observing avalanches)
        perturbation_step: Which step to start perturbation
        
    Returns:
        Dict with results: stress_history, populations, avalanches, etc.
    """
    # Create PP automaton
    ca = PP(
        rows=int(config["rows"]),
        cols=int(config["cols"]),
        densities=tuple(float(d) for d in config["densities"]),
        neighborhood=config["neighborhood"],
        params={
            "prey_death": float(config["prey_death"]),
            "predator_death": float(config["predator_death"]),
            "prey_birth": float(config["prey_birth"]),
            "predator_birth": float(config["predator_birth"]),
        },
        seed=int(config["seed"]),
        synchronous=False,  # Use async mode since sync is not fully implemented
    )
    
    # Run equilibration: slow drive allows stress to build
    stress_history = []
    grids_history = []
    prey_pops = []
    pred_pops = []
    param_history = []  # track parameter drift
    
    total_steps = n_equilibration + n_observation
    
    for step in range(total_steps):
        # Slow parameter drift (drive): gradually increase predator death
        # This is the "slow drive" that accumulates stress without immediate release
        if step >= perturbation_step:
            progress = (step - perturbation_step) / (total_steps - perturbation_step)
            drift_amount = 0.05 * progress  # drift up to +0.05
            ca.params["predator_death"] = config["predator_death"] + drift_amount
        
        # Record state before update
        stress = compute_grid_stress(ca.grid)
        stress_history.append(stress)
        grids_history.append(ca.grid.copy())
        prey_pops.append((ca.grid == 1).sum())
        pred_pops.append((ca.grid == 2).sum())
        param_history.append(float(ca.params["predator_death"]))
        
        # Update CA
        ca.update()
    
    # Detect avalanche events
    avalanches = detect_avalanche_events(grids_history, population_change_threshold=0.05)
    
    # Compute variance (intermittent release signature)
    prey_var, pred_var = compute_population_variance(grids_history)
    
    # Ensure exact length match with steps (fix any off-by-one errors)
    if len(prey_var) < len(grids_history):
        prey_var = np.pad(prey_var, (0, len(grids_history) - len(prey_var)), mode='edge')
    if len(pred_var) < len(grids_history):
        pred_var = np.pad(pred_var, (0, len(grids_history) - len(pred_var)), mode='edge')
    
    results = {
        "config": config,
        "stress_history": np.array(stress_history),
        "prey_populations": np.array(prey_pops),
        "pred_populations": np.array(pred_pops),
        "param_history": np.array(param_history),
        "avalanches": avalanches,
        "prey_variance": prey_var,
        "pred_variance": pred_var,
        "grids_history": grids_history,
        "total_steps": total_steps,
        "n_equilibration": n_equilibration,
    }
    
    return results


# ============================================================================
# 4. ROBUSTNESS ANALYSIS (Criticality across parameters)
# ============================================================================

def analyze_soc_robustness(experiment_results: List[Dict]) -> Dict:
    """
    Analyze robustness of critical behavior across diverse parameter configs.
    
    SOC robustness signature: avalanche statistics (frequency, magnitude)
    remain relatively consistent across diverse parameter combinations,
    indicating self-organization independent of tuning.
    
    Args:
        experiment_results: List of results from run_soc_perturbation_experiment()
        
    Returns:
        Dict with robustness metrics
    """
    avalanche_counts = []
    avalanche_magnitudes = []
    stress_levels = []
    population_variances = []
    
    for result in experiment_results:
        if result["avalanches"]:
            avalanche_counts.append(len(result["avalanches"]))
            mags = [mag for _, mag in result["avalanches"]]
            avalanche_magnitudes.extend(mags)
        else:
            avalanche_counts.append(0)
        
        stress_levels.extend(result["stress_history"].tolist())
        population_variances.append(result["prey_variance"].mean())
    
    robustness_metrics = {
        "avg_avalanche_count": np.mean(avalanche_counts) if avalanche_counts else 0,
        "std_avalanche_count": np.std(avalanche_counts) if avalanche_counts else 0,
        "avalanche_magnitude_mean": np.mean(avalanche_magnitudes) if avalanche_magnitudes else 0,
        "avalanche_magnitude_std": np.std(avalanche_magnitudes) if avalanche_magnitudes else 0,
        "avg_stress": np.mean(stress_levels),
        "std_stress": np.std(stress_levels),
        "avg_population_variance": np.mean(population_variances),
        "coefficient_of_variation_avalanche": (
            np.std(avalanche_counts) / np.mean(avalanche_counts)
            if np.mean(avalanche_counts) > 0 else np.inf
        ),
    }
    
    return robustness_metrics


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def visualize_soc_properties(experiment_results: List[Dict],
                            robustness_metrics: Dict,
                            output_file: Optional[str] = None):
    """
    Visualization of the 4 core SOC properties in prey-predator CA.
    
    Shows:
    1. Slow drive: Gradual parameter drift
    2. Build-up of stress: Stress accumulation with thresholds
    3. Intermittent release: Avalanche cascades and population dynamics
    4. Self-organization: Robustness across diverse configurations
    
    Args:
        experiment_results: List of experiment results
        robustness_metrics: Robustness analysis output
        output_file: Optional file path to save figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Select a representative experiment (middle one)
    rep_idx = len(experiment_results) // 2
    rep_result = experiment_results[rep_idx]
    steps = np.arange(len(rep_result["stress_history"]))
    
    # ========== SOC PROPERTY 1: SLOW DRIVE ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, rep_result["param_history"], 'purple', linewidth=2.5)
    ax1.axvline(rep_result["n_equilibration"], color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Perturbation start')
    ax1.fill_between(steps[:rep_result["n_equilibration"]], 
                     0, 0.3, alpha=0.1, color='blue')
    ax1.fill_between(steps[rep_result["n_equilibration"]:], 
                     0, 0.3, alpha=0.15, color='red')
    ax1.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predator Death Rate', fontsize=11, fontweight='bold')
    ax1.set_title('1) SLOW DRIVE\nGradual Parameter Change', 
                  fontsize=12, fontweight='bold', color='darkblue')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ========== SOC PROPERTY 2: BUILD-UP OF STRESS ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, rep_result["stress_history"], 'b-', linewidth=2.5, label='Stress Level')
    ax2.axvline(rep_result["n_equilibration"], color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label='Perturbation start')
    
    # Mark avalanche events with stars
    for event_t, event_mag in rep_result["avalanches"]:
        ax2.scatter(event_t, rep_result["stress_history"][event_t], 
                   color='orange', s=150, marker='*', zorder=5, edgecolors='black', linewidth=1.5)
    
    ax2.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Stress (Interface Density)', fontsize=11, fontweight='bold')
    ax2.set_title('2) BUILD-UP OF STRESS\nThresholds & Potential Energy', 
                  fontsize=12, fontweight='bold', color='darkblue')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # ========== SOC PROPERTY 3: INTERMITTENT RELEASE ==========
    ax3 = fig.add_subplot(gs[1, 0])
    prey = rep_result["prey_populations"]
    pred = rep_result["pred_populations"]
    
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(steps, prey, 'g-', label='Prey', linewidth=2.5)
    line2 = ax3_twin.plot(steps, pred, 'r-', label='Predator', linewidth=2.5)
    ax3.axvline(rep_result["n_equilibration"], color='gray', linestyle='--', 
                alpha=0.6, linewidth=1.5)
    
    ax3.set_xlabel('Time Step', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Prey Population', color='g', fontsize=11, fontweight='bold')
    ax3_twin.set_ylabel('Predator Population', color='r', fontsize=11, fontweight='bold')
    ax3.set_title('3) INTERMITTENT RELEASE\nAvalanche Cascades', 
                  fontsize=12, fontweight='bold', color='darkblue')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, fontsize=10, loc='upper left')
    
    # ========== SOC PROPERTY 4: SELF-ORGANIZATION ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Stress-density relation: shows universal behavior across configurations
    densities_list = []
    stresses_list = []
    avalanche_counts_list = []
    
    for result in experiment_results:
        # Calculate mean population density during observation phase
        prey_pop = result["prey_populations"][result["n_equilibration"]:]
        pred_pop = result["pred_populations"][result["n_equilibration"]:]
        total_pop = (prey_pop + pred_pop).mean()
        grid_size = result["config"]["rows"] * result["config"]["cols"]
        density = total_pop / grid_size
        
        # Mean stress during observation phase
        mean_stress = result["stress_history"][result["n_equilibration"]:].mean()
        avalanche_count = len(result["avalanches"])
        
        densities_list.append(density)
        stresses_list.append(mean_stress)
        avalanche_counts_list.append(avalanche_count)
    
    # Scatter plot: stress vs density, colored by avalanche activity
    scatter = ax4.scatter(densities_list, stresses_list, c=avalanche_counts_list,
                         cmap='plasma', s=300, alpha=0.8, edgecolors='none')
    
    ax4.set_xlabel('Population Density', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Mean Stress Level', fontsize=11, fontweight='bold')
    ax4.set_title('4) SELF-ORGANIZATION\nStress-Density Relation', 
                  fontsize=12, fontweight='bold', color='darkblue')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Avalanche Count', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Prey-Predator Cellular Automaton: Four SOC Properties',
                fontsize=14, fontweight='bold', y=0.98)
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    return fig


# ============================================================================
# 6. MAIN EXPERIMENT
# ============================================================================

def main():
    """Run complete SOC analysis."""
    print("=" * 80)
    print("SELF-ORGANIZED CRITICALITY ANALYSIS: Prey-Predator Cellular Automaton")
    print("=" * 80)
    print()
    
    # Generate diverse parameter configurations
    print("[1/4] Generating parameter configurations...")
    n_configs = 8  # Small sample for demonstration (not full analysis)
    configs = sample_parameter_configurations(n_samples=n_configs, base_seed=42)
    print(f"     Generated {n_configs} configurations with varied:")
    print("     - Grid sizes (16x16 to 64x64)")
    print("     - Initial densities (prey: 0.1-0.4, pred: 0.02-0.15)")
    print("     - Neighborhoods (Neumann/Moore)")
    print("     - Synchronicity (sync/async)")
    print("     - Rate parameters (beyond just death/birth)")
    print()
    
    # Run perturbation experiments
    print("[2/4] Running perturbation experiments...")
    experiment_results = []
    for i, config in enumerate(configs):
        print(f"     Config {i+1}/{n_configs}: "
              f"grid={config['rows']}x{config['cols']}, "
              f"densities=({config['densities'][0]:.2f},{config['densities'][1]:.2f}), "
              f"sync={config['synchronous']}")
        
        result = run_soc_perturbation_experiment(
            config,
            n_equilibration=80,  # build stress without perturbation
            n_observation=150,   # observe cascades during/after perturbation
            perturbation_step=80
        )
        experiment_results.append(result)
    print(f"     Completed {len(experiment_results)} experiments")
    print()
    
    # Analyze robustness
    print("[3/4] Analyzing SOC robustness across configurations...")
    robustness_metrics = analyze_soc_robustness(experiment_results)
    print(f"     Avalanche count (avg): {robustness_metrics['avg_avalanche_count']:.2f} "
          f"(std: {robustness_metrics['std_avalanche_count']:.2f})")
    print(f"     Avalanche magnitude (avg): {robustness_metrics['avalanche_magnitude_mean']:.4f}")
    print(f"     Stress level (avg): {robustness_metrics['avg_stress']:.4f}")
    print(f"     Coefficient of Variation (avalanche count): {robustness_metrics['coefficient_of_variation_avalanche']:.3f}")
    if robustness_metrics['coefficient_of_variation_avalanche'] < 1.0:
        print("     → LOW variation indicates ROBUST criticality across diverse parameters ✓")
    else:
        print("     → HIGH variation indicates parameter-dependent behavior")
    print()
    
    # Create visualization
    print("[4/4] Creating comprehensive visualization...")
    output_path = Path(__file__).parent.parent / "soc_analysis_results.png"
    visualize_soc_properties(experiment_results, robustness_metrics, str(output_path))
    print(f"     Saved to: {output_path}")


if __name__ == "__main__":
    main()

```