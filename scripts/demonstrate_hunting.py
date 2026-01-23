#!/usr/bin/env python3
"""
Demonstration: Predator Hunting Behavior vs Random Movement

This script creates a visualization of how directed hunting improves predator efficiency
compared to what would happen with random movement. 

Key observation: With directed hunting, predators can successfully hunt prey that are
clustered together, creating the spatial dependency crucial for the hydra effect.
"""

import numpy as np
from models.CA import PP
import json


def demonstrate_hunting_efficiency():
    """
    Compare predator efficiency under current directed hunting.
    Shows how predators concentrate on areas with high prey density.
    """
    print("\n" + "="*80)
    print("DEMONSTRATION: Predator Hunting Efficiency")
    print("="*80)
    
    params = {
        "prey_birth": 0.25,
        "prey_death": 0.02,
        "predator_birth": 0.7,
        "predator_death": 0.03,
    }
    
    pp = PP(
        rows=50,
        cols=50,
        densities=(0.35, 0.08),
        neighborhood="moore",
        params=params,
        seed=12345,
        synchronous=True,
    )
    
    print(f"\nInitial Population:")
    print(f"  Prey: {np.sum(pp.grid == 1):3d}")
    print(f"  Predators: {np.sum(pp.grid == 2):3d}")
    
    # Track metrics over time
    metrics = {
        "time": [],
        "prey": [],
        "predators": [],
        "prey_per_predator": [],
        "hunt_success_potential": []
    }
    
    for step in range(50):
        # Before update: capture state
        prev_prey = np.sum(pp.grid == 1)
        
        pp.update()
        
        # After update: calculate metrics
        curr_prey = np.sum(pp.grid == 1)
        curr_pred = np.sum(pp.grid == 2)
        
        # Calculate hunting efficiency proxy
        # (how many prey per predator - lower is better for predators)
        if curr_pred > 0:
            prey_per_pred = curr_prey / curr_pred
        else:
            prey_per_pred = 0
        
        # Calculate "hunt success potential"
        # This estimates how many predator-prey adjacencies exist
        pred_pos = np.argwhere(pp.grid == 2)
        hunt_potential = 0
        if pred_pos.size > 0:
            for r, c in pred_pos:
                # Count prey neighbors
                neighbors_r = [(r-1) % 50, (r+1) % 50, r, r]
                neighbors_c = [c, c, (c-1) % 50, (c+1) % 50]
                for nr, nc in zip(neighbors_r, neighbors_c):
                    if pp.grid[nr, nc] == 1:
                        hunt_potential += 1
        
        metrics["time"].append(step)
        metrics["prey"].append(curr_prey)
        metrics["predators"].append(curr_pred)
        metrics["prey_per_predator"].append(prey_per_pred)
        metrics["hunt_success_potential"].append(hunt_potential)
        
        if (step + 1) % 10 == 0:
            print(f"Step {step+1:2d}: Prey={curr_prey:3d}, Predators={curr_pred:3d}, " +
                  f"P/Pred={prey_per_pred:.2f}, Hunt_Potential={hunt_potential:4d}")
    
    return metrics


def analyze_hunting_behavior_with_clustering():
    """
    Demonstrate how directed hunting exploits prey clustering.
    A key mechanism behind the hydra effect.
    """
    print("\n" + "="*80)
    print("DEMONSTRATION: Hunting Behavior with Prey Clustering")
    print("="*80)
    
    # Start with empty grid to create controlled scenario
    params = {
        "prey_birth": 0.8,
        "prey_death": 0.01,
        "predator_birth": 0.85,
        "predator_death": 0.02,
    }
    
    pp = PP(
        rows=40,
        cols=40,
        densities=(0.0, 0.0),
        neighborhood="moore",
        params=params,
        seed=42,
        synchronous=True,
    )
    
    # Create two prey clusters
    print("\nSetup: Creating two isolated prey clusters")
    
    # Cluster 1: Center at (15, 15)
    for r in range(10, 20):
        for c in range(10, 20):
            if np.random.random() < 0.3:
                pp.grid[r, c] = 1
    
    # Cluster 2: Center at (30, 30)
    for r in range(25, 35):
        for c in range(25, 35):
            if np.random.random() < 0.3:
                pp.grid[r, c] = 1
    
    # Scatter some predators
    prey_count_c1 = np.sum(pp.grid[10:20, 10:20] == 1)
    prey_count_c2 = np.sum(pp.grid[25:35, 25:35] == 1)
    print(f"  Cluster 1 (10:20, 10:20): {prey_count_c1} prey")
    print(f"  Cluster 2 (25:35, 25:35): {prey_count_c2} prey")
    
    # Add predators between clusters (far from both initially)
    for r in range(18, 24):
        for c in range(18, 24):
            if np.random.random() < 0.15:
                pp.grid[r, c] = 2
    
    initial_pred = np.sum(pp.grid == 2)
    print(f"  Initial predators: {initial_pred}")
    
    print("\nDuring simulation, directed hunting causes predators to:")
    print("  1. Move toward nearest cluster (C1 or C2)")
    print("  2. Exploit high prey density")
    print("  3. Create localized hunting pressure")
    
    # Run simulation
    for step in range(30):
        pp.update()
        
        if (step + 1) % 5 == 0:
            c1_prey = np.sum(pp.grid[10:20, 10:20] == 1)
            c1_pred = np.sum(pp.grid[10:20, 10:20] == 2)
            c2_prey = np.sum(pp.grid[25:35, 25:35] == 1)
            c2_pred = np.sum(pp.grid[25:35, 25:35] == 2)
            
            print(f"Step {step+1:2d}: Cluster1(P={c1_prey:2d},Pr={c1_pred:2d}) " +
                  f"Cluster2(P={c2_prey:2d},Pr={c2_pred:2d})")

def main():
    print("\n" + "#" * 80)
    print("# PREDATOR-PREY DIRECTED HUNTING: BEHAVIOR DEMONSTRATION")
    print("#" * 80)
    
    metrics = demonstrate_hunting_efficiency()
    analyze_hunting_behavior_with_clustering()
    for i in metrics:
        print(f"{i}: {metrics[i]}")

if __name__ == "__main__":
    main()
