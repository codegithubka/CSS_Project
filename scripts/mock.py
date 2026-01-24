import numpy as np
from pathlib import Path
import json
from dataclasses import asdict
import sys
import os

sys.path.append(os.getcwd())
from scripts.pp_analysis import Config, save_sweep_binary

def generate_mock_data(output_dir: str):
    cfg = Config(default_grid=100, n_prey_birth=10, n_prey_death=10, n_replicates=2)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    prey_births = np.linspace(0.1, 0.4, 10)
    prey_deaths = np.linspace(0.01, 0.1, 10)
    results = []

    for pb in prey_births:
        for pd in prey_deaths:
            for evo in [False, True]:
                for rep in range(cfg.n_replicates):
                    hydra_factor = 5000 * np.exp(-(pd - 0.04)**2 / 0.001) if not evo else 6000
                    base_pop = (pb * 10000) - (pd * 20000) + hydra_factor
                    
                    # Ensure pop doesn't go negative
                    prey_mean = max(50, base_pop + np.random.normal(0, 100))
                    
                    # Mock PCF data
                    dist = np.linspace(0.5, 20, 20)
                    # C_cr < 1 indicates segregation
                    seg_idx = 0.8 + (pd * 2) 
                    
                    res = {
                        "prey_birth": float(pb),
                        "prey_death": float(pd),
                        "with_evolution": evo,
                        "prey_mean": float(prey_mean),
                        "pred_mean": float(prey_mean * 0.4),
                        "prey_survived": bool(prey_mean > 100),
                        "prey_tau": 2.05 + np.random.normal(0, 0.05), 
                        "evolved_prey_death_mean": float(pd * 1.2) if evo else np.nan,
                        "segregation_index": float(seg_idx),
                        "prey_clustering_index": 1.5,
                        "pcf_distances": dist.tolist(),
                        "pcf_prey_prey_mean": (1.0 + np.exp(-dist/2)).tolist()
                    }
                    results.append(res)

    # Save as .npz to mimic the real output
    save_sweep_binary(results, out_path / "sweep_results.npz")
    
    # Save a mock config
    with open(out_path / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
        
    print(f"Mock data generated in {output_dir}")

if __name__ == "__main__":
    generate_mock_data("mock_results")