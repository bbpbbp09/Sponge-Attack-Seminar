"""
Compare All Energy Attacks
==========================
Generates a comparative analysis (Table + Plots) of the best Energy-Adversarial inputs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
CONTEXT_LEN = 64
NUM_FEATURES = 9
FEATURE_NAMES = [
    'Power', 'Temp', 'Humidity', 'DewPoint', 
    'Wind10m', 'Wind100m', 'Dir10m', 'Dir100m', 'Gusts'
]

MODELS = {
    'Chronos': 'chronos_energy_sponge_best_input.npy',
    'DeepAR': 'deepar_energy_best_input.npy',
    'ACT-LSTM': 'act_energy_best_input.npy'
}

# Observed Results from Experiments (Hardcoded for visualization)
RESULTS = {
    'Chronos': "Base: 4.5J\nAdv: 4.5J\nDelta: ~0%",
    'DeepAR': "Base: 15.0J\nAdv: 16.2J\nDelta: +8%",
    'ACT-LSTM': "Base: 6.4J\nAdv: 8.9J\nDelta: +38%"
}

def load_input(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Returning zeros.")
        return np.zeros((CONTEXT_LEN, NUM_FEATURES))
    return np.load(filepath).reshape(CONTEXT_LEN, NUM_FEATURES)

def plot_comparison():
    plt.figure(figsize=(15, 12))
    
    # 1. Feature Statistics Table
    stats = []
    
    for model_name, path in MODELS.items():
        data = load_input(path)
        
        row = {
            'Model': model_name,
            'Min Val': data.min(),
            'Max Val': data.max(),
            'Mean': data.mean(),
            'Std Dev': data.std(),
            'L2 Norm': np.linalg.norm(data)
        }
        stats.append(row)

    # 1. Table Plot
    ax_table = plt.subplot(4, 1, 1)
    ax_table.axis('off')
    ax_table.set_title("Energy-Adversarial Input Statistics")
    
    df = pd.DataFrame(stats)
    # Round values for cleaner display
    df_display = df.copy()
    for col in ['Mean', 'Std Dev', 'L2 Norm']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.2e}")
    for col in ['Min Val', 'Max Val']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.2e}")
        
    table = plt.table(cellText=df_display.values,
                      colLabels=df_display.columns,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # 2. Heatmaps of full input features
    for i, (model_name, path) in enumerate(MODELS.items()):
        data = load_input(path)
        plt.subplot(4, 1, i + 2) # Start at plot 2
        plt.imshow(data.T, cmap="inferno", aspect='auto') # Inferno cmap for "Energy/Heat" vibe
        plt.colorbar(label='Value')
        plt.title(f"{model_name} - Energy Input Heatmap")
        plt.yticks(ticks=np.arange(NUM_FEATURES), labels=FEATURE_NAMES, rotation=0)
        plt.xlabel("Time Step")
        
        # Add Results Text
        res_text = RESULTS.get(model_name, "")
        plt.text(1.25, 0.5, res_text, transform=plt.gca().transAxes, 
                 verticalalignment='center', fontsize=12, fontweight='bold', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(right=0.75) # Make room for text
    plt.savefig("energy_attack_comparison_summary.png", dpi=150)
    print("Saved plot: energy_attack_comparison_summary.png")
    
    # Print Table
    print("\nEnergy-Adversarial Input Statistics:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    plot_comparison()
