import json
import csv
import os
import matplotlib.pyplot as plt

import json
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

def process_results():
    results_path = os.path.expanduser('~/Documents/formicabot_ws/simulation_results/swarm_results_latest.json')
    output_prefix = os.path.expanduser('~/Documents/formicabot_ws/simulation_results/swarm_analysis')
    
    if not os.path.exists(results_path):
        print(f"Error: Could not find {results_path}")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    history = data.get('cluster_history', [])
    if not history:
        print("No cluster history found.")
        return

    # 1. Save Detailed CSV
    keys = history[0].keys()
    csv_path = f"{output_prefix}_data.csv"
    with open(csv_path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(history)

    # 2. Professional Plotting
    timesteps = [h['timestep'] for h in history]
    n_scouts = [h['n_scouts'] for h in history]
    n_workers = [h['n_workers'] for h in history]
    n_noise = [h.get('n_noise', 0) for h in history]

    # Use a cleaner style
    plt.style.use('seaborn-v0_8-paper') 
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Roles
    ax1.plot(timesteps, n_scouts, label='Scouts (Exploration)', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax1.plot(timesteps, n_workers, label='Workers (Exploitation)', color='#2ca02c', linewidth=2.5, marker='s', markersize=4)
    ax1.plot(timesteps, n_noise, label='Noise (Transition)', color='#d62728', linestyle='--', alpha=0.6)

    # Styling
    ax1.set_title('Evolution of Swarm Collective Intelligence\n(Emergent Role Differentiation via OPTICS)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Simulation Step', fontsize=12)
    ax1.set_ylabel('Robot Count', fontsize=12)
    ax1.set_ylim(-0.5, 10.5)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Add a second axis for Cluster Count if available
    n_clusters = [h.get('n_clusters', 1) for h in history]
    ax2 = ax1.twinx()
    ax2.step(timesteps, n_clusters, where='post', color='#ff7f0e', alpha=0.3, label='Cluster Count')
    ax2.set_ylabel('Detected Clusters', color='#ff7f0e', fontsize=12, alpha=0.6)
    ax2.set_ylim(0, 5)

    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, shadow=True)

    plt.tight_layout()
    plot_path = f"{output_prefix}_viz.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n--- Analysis Complete ---")
    print(f"Publication-ready Plot: {plot_path}")
    print(f"Data Spreadsheet: {csv_path}")

if __name__ == "__main__":
    process_results()
