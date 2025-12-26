"""
Analyze Adversarial Input Patterns for Energy Sponge Attack
============================================================
Compare high-power adversarial inputs vs baseline to understand
what patterns cause increased GPU power consumption.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Configuration
CONTEXT_LEN = 64
NUM_FEATURES = 9
FEATURE_COLS = [
    'Power', 'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 
    'windspeed_10m', 'windspeed_100m', 'winddirection_10m', 
    'winddirection_100m', 'windgusts_10m'
]

# =============================================================================
# LOAD DATA
# =============================================================================
print("="*70)
print("ADVERSARIAL INPUT PATTERN ANALYSIS")
print("="*70)

# Load baseline seed data
try:
    df = pd.read_csv("data/Location1.csv")
    df = df.sort_values('Time')
    data = df[FEATURE_COLS].values.astype(np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-5
    baseline = (data[:CONTEXT_LEN] - mean) / std
    print(f"✓ Loaded baseline: {baseline.shape}")
except:
    baseline = np.load("original_seed.npy").reshape(CONTEXT_LEN, NUM_FEATURES)
    print(f"✓ Loaded baseline from npy: {baseline.shape}")

# Load Hall of Fame adversarial inputs
hof_inputs = []
hof_powers = [74.2, 73.9, 75.7, 74.7, 74.6, 75.2, 74.4, 72.6, 75.0, 74.6]  # From run output

for i in range(1, 11):
    try:
        inp = np.load(f"chronos_energy_sponge_hof_{i}.npy").reshape(CONTEXT_LEN, NUM_FEATURES)
        hof_inputs.append(inp)
    except:
        pass

print(f"✓ Loaded {len(hof_inputs)} Hall of Fame inputs")

if not hof_inputs:
    print("No HoF inputs found!")
    exit(1)

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("STATISTICAL COMPARISON: Baseline vs Adversarial")
print("="*70)

# Stack all adversarial inputs
adversarial = np.stack(hof_inputs)  # (N_hof, 64, 9)
adv_mean = adversarial.mean(axis=0)  # Average adversarial pattern

print(f"\n{'Metric':<30} {'Baseline':>15} {'Adversarial':>15} {'Ratio':>10}")
print("-"*70)

# Overall statistics
base_flat = baseline.flatten()
adv_flat = adv_mean.flatten()

metrics = [
    ("Mean", np.mean(base_flat), np.mean(adv_flat)),
    ("Std Dev", np.std(base_flat), np.std(adv_flat)),
    ("Min", np.min(base_flat), np.min(adv_flat)),
    ("Max", np.max(base_flat), np.max(adv_flat)),
    ("Abs Mean", np.mean(np.abs(base_flat)), np.mean(np.abs(adv_flat))),
    ("Range", np.ptp(base_flat), np.ptp(adv_flat)),
    ("Variance", np.var(base_flat), np.var(adv_flat)),
]

for name, base_val, adv_val in metrics:
    ratio = adv_val / base_val if abs(base_val) > 1e-6 else float('inf')
    print(f"{name:<30} {base_val:>15.4f} {adv_val:>15.4f} {ratio:>10.2f}x")

# Per-feature analysis
print("\n" + "="*70)
print("PER-FEATURE ANALYSIS")
print("="*70)
print(f"\n{'Feature':<25} {'Base Std':>12} {'Adv Std':>12} {'Std Ratio':>12} {'Base Range':>12} {'Adv Range':>12}")
print("-"*85)

for i, feat in enumerate(FEATURE_COLS):
    base_feat = baseline[:, i]
    adv_feat = adv_mean[:, i]
    
    base_std = np.std(base_feat)
    adv_std = np.std(adv_feat)
    std_ratio = adv_std / base_std if base_std > 1e-6 else float('inf')
    
    base_range = np.ptp(base_feat)
    adv_range = np.ptp(adv_feat)
    
    print(f"{feat:<25} {base_std:>12.4f} {adv_std:>12.4f} {std_ratio:>12.2f}x {base_range:>12.4f} {adv_range:>12.4f}")

# =============================================================================
# PATTERN ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("PATTERN ANALYSIS")
print("="*70)

# 1. Temporal Gradient (changes between timesteps)
base_grad = np.diff(baseline, axis=0)
adv_grad = np.diff(adv_mean, axis=0)

print(f"\n1. TEMPORAL GRADIENTS (changes between timesteps):")
print(f"   Baseline avg gradient magnitude: {np.mean(np.abs(base_grad)):.4f}")
print(f"   Adversarial avg gradient magnitude: {np.mean(np.abs(adv_grad)):.4f}")
print(f"   Ratio: {np.mean(np.abs(adv_grad)) / np.mean(np.abs(base_grad)):.2f}x")

# 2. Sign changes (oscillations)
base_sign_changes = np.sum(np.diff(np.sign(baseline), axis=0) != 0)
adv_sign_changes = np.sum(np.diff(np.sign(adv_mean), axis=0) != 0)

print(f"\n2. SIGN CHANGES (oscillations):")
print(f"   Baseline sign changes: {base_sign_changes}")
print(f"   Adversarial sign changes: {adv_sign_changes}")
print(f"   Ratio: {adv_sign_changes / base_sign_changes if base_sign_changes > 0 else 0:.2f}x")

# 3. Value distribution (bin diversity for Chronos tokenization)
print(f"\n3. VALUE DISTRIBUTION:")

# Simulate Chronos binning (4096 bins over typical range)
def count_unique_bins(data, n_bins=4096):
    flat = data.flatten()
    # Normalize to [0, 1] for binning
    normalized = (flat - flat.min()) / (flat.max() - flat.min() + 1e-6)
    bins = (normalized * (n_bins - 1)).astype(int)
    return len(np.unique(bins))

base_bins = count_unique_bins(baseline)
adv_bins = count_unique_bins(adv_mean)

print(f"   Baseline unique bins: {base_bins}")
print(f"   Adversarial unique bins: {adv_bins}")
print(f"   Token diversity: {adv_bins / base_bins:.2f}x")

# 4. Extreme values
base_extreme = np.sum(np.abs(base_flat) > 3)  # > 3 std from mean
adv_extreme = np.sum(np.abs(adv_flat) > 3)

print(f"\n4. EXTREME VALUES (|x| > 3):")
print(f"   Baseline extreme values: {base_extreme} ({100*base_extreme/len(base_flat):.1f}%)")
print(f"   Adversarial extreme values: {adv_extreme} ({100*adv_extreme/len(adv_flat):.1f}%)")

# 5. Feature correlation (cross-feature patterns)
base_corr = np.corrcoef(baseline.T)
adv_corr = np.corrcoef(adv_mean.T)

print(f"\n5. FEATURE CORRELATIONS:")
print(f"   Baseline avg abs correlation: {np.mean(np.abs(np.triu(base_corr, 1))):.4f}")
print(f"   Adversarial avg abs correlation: {np.mean(np.abs(np.triu(adv_corr, 1))):.4f}")

# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "="*70)
print("KEY INSIGHTS - What Causes Higher GPU Power?")
print("="*70)

insights = []

# Check variance increase
var_ratio = np.var(adv_flat) / np.var(base_flat)
if var_ratio > 2:
    insights.append(f"VARIANCE: Adversarial inputs have {var_ratio:.1f}x higher variance → More diverse token embeddings")

# Check gradient increase
grad_ratio = np.mean(np.abs(adv_grad)) / np.mean(np.abs(base_grad))
if grad_ratio > 2:
    insights.append(f"GRADIENTS: {grad_ratio:.1f}x larger temporal changes → FFN processes more varied patterns")

# Check extreme values
if adv_extreme > base_extreme * 2:
    insights.append(f"EXTREMES: {adv_extreme}x more extreme values → Edge case processing")

# Check magnitude
mag_ratio = np.mean(np.abs(adv_flat)) / np.mean(np.abs(base_flat))
if mag_ratio > 1.5:
    insights.append(f"MAGNITUDE: {mag_ratio:.1f}x larger absolute values → Larger activations in ReLU/GeLU")

for i, insight in enumerate(insights, 1):
    print(f"  {i}. {insight}")

if not insights:
    print("  No clear pattern identified - attack may be exploiting subtle embedding space properties")

# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 1. Time series comparison (first feature)
ax1 = axes[0, 0]
ax1.plot(baseline[:, 0], 'b-', linewidth=2, label='Baseline', alpha=0.7)
ax1.plot(adv_mean[:, 0], 'r-', linewidth=2, label='Adversarial (mean)')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Power Feature')
ax1.set_title('Time Series: Baseline vs Adversarial')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Value distribution histogram
ax2 = axes[0, 1]
ax2.hist(base_flat, bins=50, alpha=0.5, label='Baseline', density=True)
ax2.hist(adv_flat, bins=50, alpha=0.5, label='Adversarial', density=True)
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.set_title('Value Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Gradient magnitude comparison
ax3 = axes[1, 0]
ax3.bar(['Baseline', 'Adversarial'], 
        [np.mean(np.abs(base_grad)), np.mean(np.abs(adv_grad))],
        color=['blue', 'red'], alpha=0.7)
ax3.set_ylabel('Mean Gradient Magnitude')
ax3.set_title('Temporal Change Rate')
ax3.grid(True, alpha=0.3)

# 4. Per-feature variance comparison
ax4 = axes[1, 1]
base_vars = np.var(baseline, axis=0)
adv_vars = np.var(adv_mean, axis=0)
x = np.arange(len(FEATURE_COLS))
width = 0.35
ax4.bar(x - width/2, base_vars, width, label='Baseline', alpha=0.7)
ax4.bar(x + width/2, adv_vars, width, label='Adversarial', alpha=0.7)
ax4.set_xlabel('Feature')
ax4.set_ylabel('Variance')
ax4.set_title('Per-Feature Variance')
ax4.set_xticks(x)
ax4.set_xticklabels([f.split('_')[0][:6] for f in FEATURE_COLS], rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Heatmap: Adversarial - Baseline difference
ax5 = axes[2, 0]
diff = adv_mean - baseline
im = ax5.imshow(diff.T, aspect='auto', cmap='RdBu_r', vmin=-np.percentile(np.abs(diff), 95), vmax=np.percentile(np.abs(diff), 95))
ax5.set_xlabel('Timestep')
ax5.set_ylabel('Feature')
ax5.set_yticks(range(len(FEATURE_COLS)))
ax5.set_yticklabels([f.split('_')[0][:8] for f in FEATURE_COLS])
ax5.set_title('Adversarial - Baseline Difference')
plt.colorbar(im, ax=ax5)

# 6. Top adversarial inputs overlay
ax6 = axes[2, 1]
for i, inp in enumerate(hof_inputs[:3]):
    alpha = 1.0 - i * 0.25
    ax6.plot(inp[:, 0], alpha=alpha, linewidth=1.5, label=f'HoF #{i+1} ({hof_powers[i]:.1f}W)')
ax6.plot(baseline[:, 0], 'k--', linewidth=2, label='Baseline')
ax6.set_xlabel('Timestep')
ax6.set_ylabel('Power Feature')
ax6.set_title('Top 3 High-Power Inputs vs Baseline')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.suptitle('Energy Sponge Attack - Adversarial Pattern Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('adversarial_pattern_analysis.png', dpi=150)
print(f"\n✓ Saved: adversarial_pattern_analysis.png")

print("\n" + "="*70)
print("Analysis Complete")
print("="*70)
