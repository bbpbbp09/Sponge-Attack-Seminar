"""
Energy Sponge Attack for Chronos Time-Series Transformer
=========================================================
Since Chronos uses fixed binning (1 hour = 1 token), it's immune to latency attacks.
This script targets GPU ENERGY consumption by forcing dense FFN activations.

Attack Vector: Force non-zero activations in ReLU/GeLU layers by crafting inputs
that, after binning and embedding, produce dense activation patterns.

Key Measurements:
- GPU Power (Watts) via nvidia-smi during inference
- Energy = Power × Time (Joules)
- CPU energy via psutil as secondary metric
"""

import time
import torch
import numpy as np
import pygad
import pandas as pd
import psutil
import argparse
import subprocess
import threading
import warnings
from collections import deque
warnings.filterwarnings("ignore")

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(description="Energy Sponge Attack")
parser.add_argument("--model", type=str, choices=["chronos", "deepar", "patchtst"], 
                    default="chronos", help="Model to attack")
parser.add_argument("--mode", type=str, choices=["constrained", "extreme"], 
                    default="extreme", help="Attack mode")
parser.add_argument("--generations", type=int, default=50, help="Number of generations")
args = parser.parse_args()

MODEL_TYPE = args.model
MODE = args.mode
NUM_GENERATIONS = args.generations

print("="*70)
print("ENERGY SPONGE ATTACK - Maximizing GPU Power Consumption")
print("="*70)
print(f"Model: {MODEL_TYPE.upper()} | Mode: {MODE.upper()}")
print(f"Generations: {NUM_GENERATIONS}")
print("="*70)

# =============================================================================
# CONFIGURATION
# =============================================================================
CONTEXT_LEN = 64
PREDICTION_LEN = 10
NUM_FEATURES = 9

# Measurement Config
ENERGY_SAMPLE_INTERVAL = 0.01  # 10ms power sampling during inference
REPS_PER_MEASUREMENT = 10     # Number of inferences per energy measurement
WARMUP_REPS = 5

# GA Config
POPULATION_SIZE = 20
NUM_PARENTS_MATING = 10
MUTATION_PERCENT = 35
KEEP_ELITISM = 3

CURRENT_PROCESS = psutil.Process()

# =============================================================================
# GPU POWER MONITORING
# =============================================================================
class GPUPowerMonitor:
    """Continuously samples GPU power during inference."""
    
    def __init__(self, sample_interval=0.01):
        self.sample_interval = sample_interval
        self.readings = []
        self._running = False
        self._thread = None
        self.gpu_available = self._check_gpu()
        
    def _check_gpu(self):
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            power = float(result.stdout.strip().split('\n')[0])
            print(f"✓ GPU Power Monitoring: Current draw = {power:.1f}W")
            return True
        except Exception as e:
            print(f"✗ GPU Power Monitoring unavailable: {e}")
            return False
    
    def _get_power(self):
        """Get current GPU power in watts."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            return float(result.stdout.strip().split('\n')[0])
        except:
            return 0.0
    
    def _sample_loop(self):
        """Background thread that samples power."""
        while self._running:
            power = self._get_power()
            timestamp = time.time()
            self.readings.append((timestamp, power))
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start power monitoring."""
        self.readings = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop power monitoring and return readings."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        return self.readings
    
    def get_energy_stats(self):
        """Calculate energy stats from readings."""
        if not self.readings or len(self.readings) < 2:
            return {'avg_power': 0, 'max_power': 0, 'energy_joules': 0}
        
        powers = [r[1] for r in self.readings]
        times = [r[0] for r in self.readings]
        duration = times[-1] - times[0]
        
        # Energy = integral of power over time (trapezoidal approximation)
        energy = 0
        for i in range(1, len(self.readings)):
            dt = times[i] - times[i-1]
            avg_power = (powers[i] + powers[i-1]) / 2
            energy += avg_power * dt
        
        return {
            'avg_power': np.mean(powers),
            'max_power': np.max(powers),
            'energy_joules': energy,
            'duration': duration,
            'num_samples': len(powers)
        }

power_monitor = GPUPowerMonitor(ENERGY_SAMPLE_INTERVAL)

# =============================================================================
# MODEL LOADING  
# =============================================================================
model = None
pipeline = None

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

if MODEL_TYPE == "chronos":
    from chronos import Chronos2Pipeline
    MODEL_NAME = "autogluon/chronos-2-small"
    print(f"Loading Chronos on {device}...")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map=device,
        torch_dtype=torch.float32,
    )
    
    # Try to access internal T5 model for activation analysis
    try:
        if hasattr(pipeline, 'model'):
            internal_model = pipeline.model
            print(f"Internal model: {type(internal_model)}")
            if hasattr(internal_model, 'encoder'):
                num_layers = len(internal_model.encoder.block)
                print(f"Encoder layers: {num_layers}")
    except Exception as e:
        print(f"Could not inspect internals: {e}")

elif MODEL_TYPE == "deepar":
    from autogluon.timeseries import TimeSeriesPredictor
    import os
    MODEL_PATH = "AutogluonDeepAR"
    if not os.path.exists(MODEL_PATH):
        print(f"Error: DeepAR model not found at {MODEL_PATH}.")
        exit(1)
    
    print(f"Loading DeepAR...")
    ag_predictor = TimeSeriesPredictor.load(MODEL_PATH)
    
    try:
        wrapper = ag_predictor._trainer.load_model("DeepAR")
        inner = wrapper.most_recent_model
        if hasattr(inner, 'gts_predictor'):
            gts_predictor = inner.gts_predictor
        elif hasattr(inner.model, 'gts_predictor'):
            gts_predictor = inner.model.gts_predictor
        else:
            raise ValueError("Could not find gts_predictor")
        print(f"Raw GluonTS Predictor: {type(gts_predictor)}")
    except Exception as e:
        print(f"Failed to extract predictor: {e}")
        exit(1)

elif MODEL_TYPE == "patchtst":
    from transformers import PatchTSTConfig, PatchTSTForPrediction
    
    print("Creating PatchTST model...")
    config = PatchTSTConfig(
        num_input_channels=NUM_FEATURES,
        context_length=CONTEXT_LEN,
        prediction_length=PREDICTION_LEN,
        patch_length=16,
        stride=8,
        d_model=128,
        num_attention_heads=4,
        num_hidden_layers=3,
        ffn_dim=256,
        dropout=0.1,
        scaling="std",
    )
    model = PatchTSTForPrediction(config).to(device)
    model.eval()

# =============================================================================
# LOAD SEED DATA
# =============================================================================
print("\nLoading seed data...")

FEATURE_COLS = [
    'Power', 'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 
    'windspeed_10m', 'windspeed_100m', 'winddirection_10m', 
    'winddirection_100m', 'windgusts_10m'
]

try:
    df = pd.read_csv("data/Location1.csv")
    df = df.sort_values('Time')
    data = df[FEATURE_COLS].values.astype(np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-5
    seed_data = (data[:CONTEXT_LEN] - mean) / std
    print(f"Loaded seed data: {seed_data.shape}")
except FileNotFoundError:
    print("Warning: Location1.csv not found. Using random seed.")
    seed_data = np.random.randn(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)

flat_seed = seed_data.flatten()

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def make_prediction(input_array):
    """Run single inference on input array."""
    if MODEL_TYPE == "chronos":
        # Create tensor on CPU first (avoid pinned memory issues)
        inp = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)
        pipeline.predict(inp, prediction_length=PREDICTION_LEN)
        
    elif MODEL_TYPE == "deepar":
        target = input_array[:, 0].astype(np.float32)
        feat_dynamic = input_array[:, 1:].T.astype(np.float32)
        start_date = pd.Period("2020-01-01 00:00:00", freq="H")
        entry = {"start": start_date, "target": target, "feat_dynamic_real": feat_dynamic}
        list(gts_predictor.predict([entry]))
        
    elif MODEL_TYPE == "patchtst":
        inp = torch.tensor(input_array).float().unsqueeze(0).to(device)
        with torch.no_grad():
            model(past_values=inp)

# Sanity check
print("Running sanity check...")
try:
    make_prediction(seed_data)
    print("✓ Sanity check passed.")
except Exception as e:
    print(f"✗ CRITICAL: Sanity check failed: {e}")
    exit(1)

# =============================================================================
# ENERGY MEASUREMENT
# =============================================================================
def measure_energy(input_array, num_reps=REPS_PER_MEASUREMENT):
    """
    Measure energy consumed during inference.
    Returns: dict with energy metrics
    """
    # Warmup
    for _ in range(WARMUP_REPS):
        try:
            make_prediction(input_array)
        except:
            return {'avg_power': 0, 'max_power': 0, 'energy_per_inference': 0, 'latency': 0.001}
    
    # Synchronize GPU
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Start power monitoring
    if power_monitor.gpu_available:
        power_monitor.start()
    
    cpu_start = CURRENT_PROCESS.cpu_times()
    cpu_start_total = cpu_start.user + cpu_start.system
    start_time = time.perf_counter()
    
    # Run inferences
    for _ in range(num_reps):
        try:
            make_prediction(input_array)
        except:
            if power_monitor.gpu_available:
                power_monitor.stop()
            return {'avg_power': 0, 'max_power': 0, 'energy_per_inference': 0, 'latency': 0.001}
    
    # Synchronize GPU
    if device == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    cpu_end = CURRENT_PROCESS.cpu_times()
    cpu_end_total = cpu_end.user + cpu_end.system
    
    # Stop power monitoring
    if power_monitor.gpu_available:
        power_monitor.stop()
        power_stats = power_monitor.get_energy_stats()
    else:
        power_stats = {'avg_power': 0, 'max_power': 0, 'energy_joules': 0}
    
    total_time = end_time - start_time
    total_cpu = cpu_end_total - cpu_start_total
    
    return {
        'avg_power': power_stats['avg_power'],
        'max_power': power_stats['max_power'],
        'energy_joules': power_stats.get('energy_joules', 0),
        'energy_per_inference': power_stats.get('energy_joules', 0) / num_reps,
        'cpu_time_per_inference': total_cpu / num_reps,
        'latency': total_time / num_reps,
        'num_samples': power_stats.get('num_samples', 0)
    }

# =============================================================================
# BASELINE MEASUREMENT
# =============================================================================
print("\nMeasuring baseline energy consumption...")
baseline_stats = measure_energy(seed_data, num_reps=20)
print(f"Baseline Stats:")
print(f"  Average GPU Power: {baseline_stats['avg_power']:.1f}W")
print(f"  Max GPU Power:     {baseline_stats['max_power']:.1f}W")
print(f"  Energy/Inference:  {baseline_stats['energy_per_inference']*1000:.3f} mJ")
print(f"  CPU Time/Inf:      {baseline_stats['cpu_time_per_inference']*1000:.2f} ms")
print(f"  Latency/Inf:       {baseline_stats['latency']*1000:.2f} ms")

BASELINE_ENERGY = baseline_stats['energy_per_inference']
BASELINE_POWER = baseline_stats['avg_power']
BASELINE_CPU = baseline_stats['cpu_time_per_inference']
BASELINE_LATENCY = baseline_stats['latency']

# =============================================================================
# TRACKING
# =============================================================================
generation_data = {
    'gen': [], 'max_fitness': [], 'avg_fitness': [],
    'best_power': [], 'best_energy': [], 'best_latency': [],
    # Global best tracking (monotonically increasing)
    'global_best_fitness': [], 'global_best_power': [], 'global_best_energy': []
}

hall_of_fame = []
current_gen_fitness = []
current_gen_solutions = []
global_best = {'fitness': 0, 'power': 0, 'energy': 0, 'latency': 0, 'solution': None}

# =============================================================================
# FITNESS FUNCTION (ENERGY-FOCUSED)
# =============================================================================
def fitness_func(ga_instance, solution, solution_idx):
    global current_gen_fitness, current_gen_solutions, global_best
    
    input_array = solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    
    # Measure energy for adversarial input
    adv_stats = measure_energy(input_array, num_reps=REPS_PER_MEASUREMENT)
    
    # Primary fitness: maximize power consumption
    # We want inputs that cause the GPU to draw more power
    power_ratio = adv_stats['avg_power'] / BASELINE_POWER if BASELINE_POWER > 0 else 1.0
    
    # Secondary: CPU time (proxy for dense computation)
    cpu_ratio = adv_stats['cpu_time_per_inference'] / BASELINE_CPU if BASELINE_CPU > 0 else 1.0
    
    # Combined fitness (weighted toward power)
    if power_monitor.gpu_available:
        fitness = 0.7 * power_ratio + 0.3 * cpu_ratio
    else:
        # No GPU monitoring - use CPU and latency as proxies
        lat_ratio = adv_stats['latency'] / BASELINE_LATENCY if BASELINE_LATENCY > 0 else 1.0
        fitness = 0.6 * cpu_ratio + 0.4 * lat_ratio
    
    # Track
    current_gen_fitness.append(fitness)
    current_gen_solutions.append({
        'fitness': fitness,
        'power': adv_stats['avg_power'],
        'energy': adv_stats['energy_per_inference'],
        'latency': adv_stats['latency'],
        'solution': solution.copy()
    })
    
    # Update global best
    if fitness > global_best['fitness']:
        global_best = {
            'fitness': fitness,
            'power': adv_stats['avg_power'],
            'energy': adv_stats['energy_per_inference'],
            'latency': adv_stats['latency'],
            'solution': solution.copy()
        }
    
    # Hall of Fame
    hall_of_fame.append({
        'fitness': fitness,
        'power': adv_stats['avg_power'],
        'energy': adv_stats['energy_per_inference'],
        'solution': solution.copy()
    })
    hall_of_fame.sort(key=lambda x: x['fitness'], reverse=True)
    while len(hall_of_fame) > 10:
        hall_of_fame.pop()
    
    return float(fitness)

# =============================================================================
# GENERATION CALLBACK
# =============================================================================
def on_generation(ga_instance):
    global current_gen_fitness, current_gen_solutions
    
    gen = ga_instance.generations_completed
    
    if current_gen_fitness:
        max_fit = max(current_gen_fitness)
        avg_fit = np.mean(current_gen_fitness)
        
        best = max(current_gen_solutions, key=lambda x: x['fitness'])
        
        generation_data['gen'].append(gen)
        generation_data['max_fitness'].append(max_fit)
        generation_data['avg_fitness'].append(avg_fit)
        generation_data['best_power'].append(best['power'])
        generation_data['best_energy'].append(best['energy'])
        generation_data['best_latency'].append(best['latency'])
        
        # Track global best (monotonically increasing)
        generation_data['global_best_fitness'].append(global_best['fitness'])
        generation_data['global_best_power'].append(global_best['power'])
        generation_data['global_best_energy'].append(global_best['energy'])
        
        power_change = ((best['power'] / BASELINE_POWER) - 1) * 100 if BASELINE_POWER > 0 else 0
        gb_power_change = ((global_best['power'] / BASELINE_POWER) - 1) * 100 if BASELINE_POWER > 0 else 0
        
        print(f"Gen {gen:3d}: ThisGen={max_fit:.4f} (Power {power_change:+.1f}%) | "
              f"GlobalBest={global_best['fitness']:.4f} ({gb_power_change:+.1f}%)")
    
    current_gen_fitness = []
    current_gen_solutions = []

# =============================================================================
# MUTATION (DENSE ACTIVATION TARGETING)
# =============================================================================
def energy_mutation(offspring, ga_instance):
    """
    Mutation strategy targeting dense FFN activations.
    Goal: Create inputs that produce non-zero outputs from ReLU/GeLU neurons.
    """
    for idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.random() < MUTATION_PERCENT / 100:
                mutation_type = np.random.random()
                
                if mutation_type < 0.25:
                    # MODERATE MAGNITUDE (avoid saturation, maximize ReLU gradient flow)
                    # ReLU is linear for positive values, but saturates at 0
                    # GELU has gradual transition - moderate values = maximum gradient
                    scale = np.random.choice([1, 2, 5, 10, 20])
                    offspring[idx, gene_idx] = np.abs(np.random.normal(0, scale))
                    
                elif mutation_type < 0.45:
                    # DIVERSITY PATTERN (different values = different activations)
                    # Force each embedding to activate different neurons
                    value_range = np.random.uniform(1, 50)
                    offspring[idx, gene_idx] = np.random.uniform(-value_range, value_range)
                    
                elif mutation_type < 0.60:
                    # BOUNDARY EXPLOITATION (Chronos bin boundaries)
                    # Values at bin boundaries may cause unstable embeddings
                    bin_boundary = np.random.uniform(-1, 1) * 1000
                    epsilon = np.random.uniform(-0.01, 0.01)
                    offspring[idx, gene_idx] = bin_boundary + epsilon
                    
                elif mutation_type < 0.75:
                    # ORTHOGONAL PATTERNS (maximize attention diversity)
                    # Adjacent features should be very different
                    if gene_idx > 0:
                        prev_val = offspring[idx, gene_idx - 1]
                        # Orthogonal: different sign and magnitude
                        offspring[idx, gene_idx] = -prev_val * np.random.uniform(0.5, 2.0)
                    else:
                        offspring[idx, gene_idx] = np.random.uniform(-100, 100)
                        
                elif mutation_type < 0.85:
                    # UNIFORM DISTRIBUTION (all bins activated equally)
                    # Map to full range of possible token values
                    offspring[idx, gene_idx] = np.random.uniform(-1e4, 1e4)
                    
                elif mutation_type < 0.92:
                    # NEAR-THRESHOLD VALUES (GeLU threshold ~0)
                    # GELU(x) ≈ x * sigmoid(1.702x)
                    # Maximum curvature (and potential energy) around x ≈ 0
                    offspring[idx, gene_idx] = np.random.normal(0, 0.5)
                    
                else:
                    # DENORMAL FLOATS (slow hardware path)
                    offspring[idx, gene_idx] = np.float32(np.random.choice([
                        1e-42, 1e-43, 1e-44, 1e-45,
                        -1e-42, -1e-43, -1e-44, -1e-45
                    ]))
    
    return offspring

# =============================================================================
# INITIAL POPULATION (ENERGY-FOCUSED)
# =============================================================================
print("\nInitializing population (energy-focused strategies)...")
initial_population = []
n = len(flat_seed)

if MODE == "constrained":
    for _ in range(POPULATION_SIZE):
        noise = np.random.normal(0, 0.5, n)
        initial_population.append(flat_seed + noise)
    gene_space = {'low': -10.0, 'high': 10.0}
    
elif MODE == "extreme":
    n_per_strategy = POPULATION_SIZE // 5
    
    # Strategy 1: UNIFORM MODERATE MAGNITUDE (densest activations)
    # Not too small (zero), not too large (saturation)
    for _ in range(n_per_strategy):
        x = np.random.uniform(-50, 50, n)
        initial_population.append(x)
    
    # Strategy 2: ALTERNATING SIGNS (maximize attention computation)
    for _ in range(n_per_strategy):
        x = np.zeros(n)
        x[::2] = np.random.uniform(10, 100, len(x[::2]))
        x[1::2] = np.random.uniform(-100, -10, len(x[1::2]))
        initial_population.append(x)
    
    # Strategy 3: NEAR-ZERO THRESHOLD (GeLU sweet spot)
    for _ in range(n_per_strategy):
        x = np.random.normal(0, 1, n)
        initial_population.append(x)
    
    # Strategy 4: RANDOM DIVERSE (evolutionary exploration)
    for _ in range(n_per_strategy):
        x = np.random.uniform(-1000, 1000, n)
        initial_population.append(x)
    
    # Strategy 5: Perturbed seed
    remaining = POPULATION_SIZE - 4 * n_per_strategy
    for _ in range(remaining):
        initial_population.append(flat_seed + np.random.normal(0, 10, n))
    
    gene_space = None

initial_population = np.array(initial_population)

# =============================================================================
# PYGAD INSTANCE
# =============================================================================
print(f"\nInitializing GA: {POPULATION_SIZE} pop, {NUM_GENERATIONS} gens, {MUTATION_PERCENT}% mutation")

ga_instance = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=NUM_PARENTS_MATING,
    fitness_func=fitness_func,
    sol_per_pop=POPULATION_SIZE,
    num_genes=len(flat_seed),
    parent_selection_type="tournament",
    K_tournament=3,
    crossover_type="two_points",
    mutation_type=energy_mutation,
    initial_population=initial_population,
    gene_space=gene_space,
    on_generation=on_generation,
    keep_elitism=KEEP_ELITISM,
    suppress_warnings=True
)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Starting Energy Sponge Attack")
    print("="*70 + "\n")
    
    start_time = time.time()
    ga_instance.run()
    total_time = time.time() - start_time
    
    print(f"\nTotal GA time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Best solution
    best_solution, best_fitness, _ = ga_instance.best_solution()
    
    # Re-verify with more measurements
    print("\nRe-verifying best solution with extended measurement...")
    adv_input = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    
    verify_adv = measure_energy(adv_input, num_reps=50)
    verify_base = measure_energy(seed_data, num_reps=50)
    
    power_change = ((verify_adv['avg_power'] / verify_base['avg_power']) - 1) * 100 if verify_base['avg_power'] > 0 else 0
    energy_change = ((verify_adv['energy_per_inference'] / verify_base['energy_per_inference']) - 1) * 100 if verify_base['energy_per_inference'] > 0 else 0
    cpu_change = ((verify_adv['cpu_time_per_inference'] / verify_base['cpu_time_per_inference']) - 1) * 100 if verify_base['cpu_time_per_inference'] > 0 else 0
    lat_change = ((verify_adv['latency'] / verify_base['latency']) - 1) * 100 if verify_base['latency'] > 0 else 0
    
    print("\n" + "="*70)
    print("FINAL VERIFIED RESULTS - ENERGY SPONGE ATTACK")
    print("="*70)
    print(f"{'Metric':<25} {'Baseline':>15} {'Adversarial':>15} {'Change':>12}")
    print("-"*70)
    print(f"{'GPU Power (W)':<25} {verify_base['avg_power']:>12.1f} W {verify_adv['avg_power']:>12.1f} W {power_change:>+10.1f}%")
    print(f"{'Max GPU Power (W)':<25} {verify_base['max_power']:>12.1f} W {verify_adv['max_power']:>12.1f} W")
    print(f"{'Energy/Inference (mJ)':<25} {verify_base['energy_per_inference']*1000:>12.3f} mJ {verify_adv['energy_per_inference']*1000:>12.3f} mJ {energy_change:>+10.1f}%")
    print(f"{'CPU Time (ms)':<25} {verify_base['cpu_time_per_inference']*1000:>12.2f} ms {verify_adv['cpu_time_per_inference']*1000:>12.2f} ms {cpu_change:>+10.1f}%")
    print(f"{'Latency (ms)':<25} {verify_base['latency']*1000:>12.2f} ms {verify_adv['latency']*1000:>12.2f} ms {lat_change:>+10.1f}%")
    print("="*70)
    
    # Key insight
    print("\n>>> ATTACK ANALYSIS <<<")
    if abs(lat_change) < 5:
        print(f"✓ LATENCY: Confirmed constant-time ({lat_change:+.1f}%) - Chronos binning blocks latency attacks")
    else:
        print(f"! LATENCY: Unexpected change ({lat_change:+.1f}%)")
    
    if power_change > 2:
        print(f"✓ ENERGY: Success! Power increased by {power_change:+.1f}%")
    elif cpu_change > 2:
        print(f"~ ENERGY: CPU time increased by {cpu_change:+.1f}% (power monitoring may be limited)")
    else:
        print(f"✗ ENERGY: No significant increase detected")
    
    # Save outputs
    prefix = f"{MODEL_TYPE}_energy_sponge"
    np.save(f"{prefix}_best_input.npy", best_solution)
    np.savez(f"{prefix}_generation_data.npz", **generation_data)
    print(f"\nSaved: {prefix}_best_input.npy")
    print(f"Saved: {prefix}_generation_data.npz")
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    gens = generation_data['gen']
    
    # Fitness Evolution
    ax1 = axes[0, 0]
    ax1.plot(gens, generation_data['max_fitness'], 'b-', linewidth=1, alpha=0.5, label='Per-Gen Max (noisy)')
    ax1.plot(gens, generation_data['global_best_fitness'], 'r-', linewidth=3, label='Global Best (monotonic)')
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (ratio=1)')
    ax1.set_ylabel('Fitness (Power Ratio)')
    ax1.set_title('Fitness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # GPU Power Over Generations
    ax2 = axes[0, 1]
    if any(generation_data['best_power']):
        ax2.plot(gens, generation_data['best_power'], 'b-', linewidth=1, alpha=0.5, label='Per-Gen Best (noisy)')
        ax2.plot(gens, generation_data['global_best_power'], 'r-', linewidth=3, label='Global Best (monotonic)')
        ax2.axhline(BASELINE_POWER, color='gray', linestyle='--', label=f'Baseline ({BASELINE_POWER:.1f}W)')
    ax2.set_ylabel('GPU Power (W)')
    ax2.set_title('GPU Power Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Energy Per Inference
    ax3 = axes[1, 0]
    if any(generation_data['best_energy']):
        ax3.plot(gens, [e*1000 for e in generation_data['best_energy']], 'b-', linewidth=1, alpha=0.5, label='Per-Gen Best (noisy)')
        ax3.plot(gens, [e*1000 for e in generation_data['global_best_energy']], 'r-', linewidth=3, label='Global Best (monotonic)')
        ax3.axhline(BASELINE_ENERGY * 1000, color='gray', linestyle='--', label=f'Baseline ({BASELINE_ENERGY*1000:.3f}mJ)')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Energy (mJ/inference)')
    ax3.set_title('Energy per Inference')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Latency (should be nearly flat)
    ax4 = axes[1, 1]
    ax4.plot(gens, [l*1000 for l in generation_data['best_latency']], 'c-', linewidth=2, label='Best Latency')
    ax4.axhline(BASELINE_LATENCY * 1000, color='gray', linestyle='--', label=f'Baseline ({BASELINE_LATENCY*1000:.2f}ms)')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title('Latency (Expected: Nearly Constant)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{MODEL_TYPE.upper()} Energy Sponge Attack Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{prefix}_results.png", dpi=150)
    print(f"Saved: {prefix}_results.png")
    
    # Hall of Fame
    print(f"\n--- Hall of Fame (Top {len(hall_of_fame)}) ---")
    for i, hof in enumerate(hall_of_fame):
        print(f"  #{i+1}: Fitness={hof['fitness']:.4f} | Power={hof['power']:.1f}W | Energy={hof['energy']*1000:.3f}mJ")
        np.save(f"{prefix}_hof_{i+1}.npy", hof['solution'])
    print(f"Saved: {prefix}_hof_*.npy")
    
    print("\n" + "="*70)
    print("Energy Sponge Attack Complete")
    print("="*70)
