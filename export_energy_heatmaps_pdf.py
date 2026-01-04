"""
Export Energy Attack Heatmaps to PDF
====================================
Generates separate PDF files for each model's energy attack heatmap with stats on the right.
"""

import os
import numpy as np
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

def export_single_heatmap_pdf(model_name, path, output_filename):
    """Export a single energy heatmap with stats to a PDF file."""
    data = load_input(path)
    
    # Create figure with specific size for good PDF quality
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot heatmap with inferno colormap for "energy/heat" vibe
    im = ax.imshow(data.T, cmap="inferno", aspect='auto')
    cbar = plt.colorbar(im, ax=ax, label='Value')
    
    ax.set_title(f"{model_name} - Energy Attack Heatmap (Features x Time)", fontsize=14, fontweight='bold')
    ax.set_yticks(np.arange(NUM_FEATURES))
    ax.set_yticklabels(FEATURE_NAMES, rotation=0)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    
    # Add Results Text on the right
    res_text = RESULTS.get(model_name, "")
    ax.text(1.25, 0.5, res_text, transform=ax.transAxes, 
            verticalalignment='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))
    
    # Adjust layout to make room for the stats text
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    # Save to PDF
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_filename}")

def main():
    """Export all three energy heatmaps as separate PDF files."""
    print("Exporting energy attack heatmaps to PDF...\n")
    
    for model_name, path in MODELS.items():
        # Create filename from model name (replace special chars)
        safe_name = model_name.lower().replace('-', '_').replace(' ', '_')
        output_filename = f"{safe_name}_energy_heatmap.pdf"
        export_single_heatmap_pdf(model_name, path, output_filename)
    
    print("\n✓ All energy heatmaps exported successfully!")
    print("\nGenerated files:")
    for model_name in MODELS.keys():
        safe_name = model_name.lower().replace('-', '_').replace(' ', '_')
        print(f"  - {safe_name}_energy_heatmap.pdf")

if __name__ == "__main__":
    main()
