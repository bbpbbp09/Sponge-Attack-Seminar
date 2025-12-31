"""
DeepAR-Style LSTM Latency Attack
=================================
Genetic algorithm attack maximizing inference latency.
Uses custom PyTorch LSTM (DeepAR architecture).
"""

import time
import torch
import torch.nn as nn
import numpy as np
import pygad
import pandas as pd
import psutil
import argparse
import subprocess
import threading
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(description="DeepAR Latency Attack")
parser.add_argument("--generations", type=int, default=50, help="Number of generations")
parser.add_argument("--mode", type=str, choices=["constrained", "extreme"], default="extreme")
args = parser.parse_args()

NUM_GENERATIONS = args.generations
MODE = args.mode

print("="*70)
print("DeepAR LATENCY ATTACK - Maximizing Inference Time")
print("="*70)
print(f"Generations: {NUM_GENERATIONS} | Mode: {MODE}")
print("="*70)

# =============================================================================
# CONFIGURATION
# =============================================================================
CONTEXT_LEN = 64
PREDICTION_LEN = 10
NUM_FEATURES = 9
HIDDEN_SIZE = 128
RNN_LAYERS = 2

MODEL_PATH = "deepar_model.pt"
DATA_PATH = "data/Location1.csv"

REPS_PER_MEASUREMENT = 20
WARMUP_REPS = 10

POPULATION_SIZE = 20
NUM_PARENTS_MATING = 10
MUTATION_PERCENT = 40
KEEP_ELITISM = 3

ALL_FEATURES = [
    'Power', 'temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 
    'windspeed_10m', 'windspeed_100m', 'winddirection_10m', 
    'winddirection_100m', 'windgusts_10m'
]

CURRENT_PROCESS = psutil.Process()

# =============================================================================
# MODEL DEFINITION (must match training)
# =============================================================================
class DeepARLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, 
                 output_size=10, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output
    
    @property
    def rnn(self):
        return self.lstm

# =============================================================================
# GPU POWER MONITORING
# =============================================================================
class GPUPowerMonitor:
    def __init__(self, sample_interval=0.01):
        self.sample_interval = sample_interval
        self.readings = []
        self._running = False
        self._thread = None
        self.gpu_available = self._check_gpu()
        
    def _check_gpu(self):
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
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            return float(result.stdout.strip().split('\n')[0])
        except:
            return 0.0
    
    def _sample_loop(self):
        while self._running:
            power = self._get_power()
            timestamp = time.time()
            self.readings.append((timestamp, power))
            time.sleep(self.sample_interval)
    
    def start(self):
        self.readings = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        return self.readings
    
    def get_energy_stats(self):
        if not self.readings or len(self.readings) < 2:
            return {'avg_power': 0, 'max_power': 0}
        powers = [r[1] for r in self.readings]
        return {'avg_power': np.mean(powers), 'max_power': np.max(powers)}

power_monitor = GPUPowerMonitor(0.01)

# =============================================================================
# LOAD MODEL
# =============================================================================
print("\nLoading DeepAR model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = DeepARLSTM(
    input_size=NUM_FEATURES,
    hidden_size=HIDDEN_SIZE,
    num_layers=RNN_LAYERS,
    output_size=PREDICTION_LEN,
    dropout=0.1
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# =============================================================================
# LOAD SEED DATA
# =============================================================================
print("\nLoading seed data...")
df = pd.read_csv(DATA_PATH)
df = df.sort_values('Time').reset_index(drop=True)

norm_params = checkpoint.get('norm_params', None)
if norm_params:
    mean = np.array(norm_params['mean'])
    std = np.array(norm_params['std'])
else:
    data = df[ALL_FEATURES].values.astype(np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-5

seed_data_raw = df[ALL_FEATURES].values[:CONTEXT_LEN].astype(np.float32)
seed_data = (seed_data_raw - mean) / std
flat_seed = seed_data.flatten()
print(f"Seed data shape: {seed_data.shape}")

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def make_prediction(input_array):
    x = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x)
        return output

print("Running sanity check...")
try:
    make_prediction(seed_data)
    print("✓ Sanity check passed.")
except Exception as e:
    print(f"✗ Sanity check failed: {e}")
    exit(1)

# =============================================================================
# LATENCY MEASUREMENT
# =============================================================================
def measure_latency(input_array, num_reps=REPS_PER_MEASUREMENT):
    for _ in range(WARMUP_REPS):
        try:
            make_prediction(input_array)
        except:
            return {'latency': 0.001, 'latency_std': 0, 'avg_power': 0}
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    if power_monitor.gpu_available:
        power_monitor.start()
    
    latencies = []
    for _ in range(num_reps):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        try:
            make_prediction(input_array)
        except:
            if power_monitor.gpu_available:
                power_monitor.stop()
            return {'latency': 0.001, 'latency_std': 0, 'avg_power': 0}
        
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append(end - start)
    
    if power_monitor.gpu_available:
        power_monitor.stop()
        power_stats = power_monitor.get_energy_stats()
    else:
        power_stats = {'avg_power': 0, 'max_power': 0}
    
    return {
        'latency': np.mean(latencies),
        'latency_std': np.std(latencies),
        'latency_max': np.max(latencies),
        'latency_min': np.min(latencies),
        'avg_power': power_stats['avg_power'],
        'max_power': power_stats.get('max_power', 0)
    }

# =============================================================================
# BASELINE MEASUREMENT
# =============================================================================
print("\nMeasuring baseline latency...")
baseline_stats = measure_latency(seed_data, num_reps=50)
print(f"Baseline Stats:")
print(f"  Mean Latency:  {baseline_stats['latency']*1000:.3f} ms")
print(f"  Std Latency:   {baseline_stats['latency_std']*1000:.3f} ms")
print(f"  GPU Power:     {baseline_stats['avg_power']:.1f}W")

BASELINE_LATENCY = baseline_stats['latency']
BASELINE_POWER = baseline_stats['avg_power']

# =============================================================================
# TRACKING
# =============================================================================
generation_data = {
    'gen': [], 'max_fitness': [], 'avg_fitness': [],
    'best_latency': [], 'best_latency_std': [], 'best_power': [],
    'global_best_fitness': [], 'global_best_latency': []
}

hall_of_fame = []
current_gen_fitness = []
current_gen_solutions = []
global_best = {'fitness': 0, 'latency': 0, 'power': 0, 'solution': None}

# =============================================================================
# FITNESS FUNCTION
# =============================================================================
def fitness_func(ga_instance, solution, solution_idx):
    global current_gen_fitness, current_gen_solutions, global_best
    
    input_array = solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    stats = measure_latency(input_array, num_reps=REPS_PER_MEASUREMENT)
    
    latency_ratio = stats['latency'] / BASELINE_LATENCY if BASELINE_LATENCY > 0 else 1.0
    variance_bonus = 1 + (stats['latency_std'] / stats['latency']) if stats['latency'] > 0 else 1.0
    variance_bonus = min(variance_bonus, 1.5)
    
    fitness = latency_ratio * variance_bonus
    
    current_gen_fitness.append(fitness)
    current_gen_solutions.append({
        'fitness': fitness, 'latency': stats['latency'],
        'latency_std': stats['latency_std'], 'power': stats['avg_power'],
        'solution': solution.copy()
    })
    
    if fitness > global_best['fitness']:
        global_best = {
            'fitness': fitness, 'latency': stats['latency'],
            'power': stats['avg_power'], 'solution': solution.copy()
        }
    
    hall_of_fame.append({
        'fitness': fitness, 'latency': stats['latency'],
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
        generation_data['best_latency'].append(best['latency'])
        generation_data['best_latency_std'].append(best['latency_std'])
        generation_data['best_power'].append(best['power'])
        generation_data['global_best_fitness'].append(global_best['fitness'])
        generation_data['global_best_latency'].append(global_best['latency'])
        
        lat_change = ((best['latency'] / BASELINE_LATENCY) - 1) * 100 if BASELINE_LATENCY > 0 else 0
        gb_lat_change = ((global_best['latency'] / BASELINE_LATENCY) - 1) * 100 if BASELINE_LATENCY > 0 else 0
        
        print(f"Gen {gen:3d}: Fitness={max_fit:.4f} (Latency {lat_change:+.1f}%) | "
              f"GlobalBest={global_best['fitness']:.4f} ({gb_lat_change:+.1f}%)")
    
    current_gen_fitness = []
    current_gen_solutions = []

# =============================================================================
# MUTATION (LATENCY-FOCUSED)
# =============================================================================
def latency_mutation(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.random() < MUTATION_PERCENT / 100:
                mutation_type = np.random.random()
                
                if mutation_type < 0.25:
                    # Denormal floats
                    denormal_values = [
                        1e-38, 1e-39, 1e-40, 1e-41, 1e-42, 1e-43, 1e-44, 1e-45,
                        -1e-38, -1e-39, -1e-40, -1e-41, -1e-42, -1e-43, -1e-44, -1e-45,
                    ]
                    offspring[idx, gene_idx] = np.float32(np.random.choice(denormal_values))
                elif mutation_type < 0.40:
                    # Extreme values
                    extreme_values = [1e30, -1e30, 1e35, -1e35]
                    offspring[idx, gene_idx] = np.float32(np.random.choice(extreme_values))
                elif mutation_type < 0.55:
                    # Mixed scales
                    if gene_idx > 0:
                        scale_ratio = np.random.choice([1e10, 1e-10, 1e15, 1e-15])
                        offspring[idx, gene_idx] = offspring[idx, gene_idx-1] * scale_ratio
                    else:
                        offspring[idx, gene_idx] = np.random.choice([1e20, 1e-20])
                elif mutation_type < 0.70:
                    # Near-max values
                    offspring[idx, gene_idx] = np.float32(np.random.choice([1e38, -1e38]))
                elif mutation_type < 0.85:
                    # Oscillating
                    base = np.random.uniform(1, 1000)
                    if gene_idx % 2 == 0:
                        offspring[idx, gene_idx] = base
                    else:
                        offspring[idx, gene_idx] = -base
                else:
                    # Random
                    offspring[idx, gene_idx] = np.random.uniform(-1e6, 1e6)
    
    offspring = np.clip(offspring, -1e38, 1e38)
    return offspring

# =============================================================================
# INITIAL POPULATION
# =============================================================================
print("\nInitializing population (latency-focused strategies)...")
initial_population = []
n = len(flat_seed)

if MODE == "constrained":
    for _ in range(POPULATION_SIZE):
        noise = np.random.normal(0, 0.5, n)
        initial_population.append(flat_seed + noise)
    gene_space = {'low': -10.0, 'high': 10.0}
else:
    n_per_strategy = POPULATION_SIZE // 5
    
    for _ in range(n_per_strategy):
        x = np.random.choice([1e-42, 1e-43, 1e-44, 1e-45, -1e-42, -1e-43], n).astype(np.float32)
        initial_population.append(x)
    for _ in range(n_per_strategy):
        x = np.random.uniform(-1e35, 1e35, n).astype(np.float32)
        initial_population.append(x)
    for _ in range(n_per_strategy):
        x = np.zeros(n)
        for i in range(n):
            x[i] = 10 ** np.random.uniform(-40, 35)
            if np.random.random() < 0.5:
                x[i] = -x[i]
        initial_population.append(x.astype(np.float32))
    for _ in range(n_per_strategy):
        x = np.zeros(n)
        x[::2] = 1e30
        x[1::2] = 1e-40
        initial_population.append(x.astype(np.float32))
    remaining = POPULATION_SIZE - 4 * n_per_strategy
    for _ in range(remaining):
        initial_population.append(flat_seed + np.random.normal(0, 10, n))
    gene_space = None

initial_population = np.array(initial_population, dtype=np.float32)

# =============================================================================
# PYGAD INSTANCE
# =============================================================================
print(f"Initializing GA: {POPULATION_SIZE} pop, {NUM_GENERATIONS} gens")

ga_instance = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=NUM_PARENTS_MATING,
    fitness_func=fitness_func,
    sol_per_pop=POPULATION_SIZE,
    num_genes=len(flat_seed),
    parent_selection_type="tournament",
    K_tournament=3,
    crossover_type="two_points",
    mutation_type=latency_mutation,
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
    print("Starting DeepAR Latency Attack")
    print("="*70 + "\n")
    
    start_time = time.time()
    ga_instance.run()
    total_time = time.time() - start_time
    
    print(f"\nTotal GA time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    best_solution, best_fitness, _ = ga_instance.best_solution()
    
    print("\nRe-verifying best solution with extended measurement...")
    adv_input = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    
    verify_adv = measure_latency(adv_input, num_reps=100)
    verify_base = measure_latency(seed_data, num_reps=100)
    
    lat_change = ((verify_adv['latency'] / verify_base['latency']) - 1) * 100 if verify_base['latency'] > 0 else 0
    power_change = ((verify_adv['avg_power'] / verify_base['avg_power']) - 1) * 100 if verify_base['avg_power'] > 0 else 0
    
    print("\n" + "="*70)
    print("FINAL RESULTS - DeepAR LATENCY ATTACK")
    print("="*70)
    print(f"{'Metric':<25} {'Baseline':>15} {'Adversarial':>15} {'Change':>12}")
    print("-"*70)
    print(f"{'Mean Latency (ms)':<25} {verify_base['latency']*1000:>12.3f} ms {verify_adv['latency']*1000:>12.3f} ms {lat_change:>+10.1f}%")
    print(f"{'Std Latency (ms)':<25} {verify_base['latency_std']*1000:>12.3f} ms {verify_adv['latency_std']*1000:>12.3f} ms")
    print(f"{'Max Latency (ms)':<25} {verify_base['latency_max']*1000:>12.3f} ms {verify_adv['latency_max']*1000:>12.3f} ms")
    print(f"{'GPU Power (W)':<25} {verify_base['avg_power']:>12.1f} W {verify_adv['avg_power']:>12.1f} W {power_change:>+10.1f}%")
    print("="*70)
    
    if lat_change > 10:
        print(f"\n✓ LATENCY ATTACK SUCCESS: {lat_change:+.1f}% increase in inference time!")
    elif lat_change > 0:
        print(f"\n~ LATENCY ATTACK: Modest {lat_change:+.1f}% increase detected")
    else:
        print(f"\n✗ LATENCY ATTACK: No significant increase detected")
    
    # Save outputs
    prefix = "deepar_latency"
    np.save(f"{prefix}_best_input.npy", best_solution)
    np.savez(f"{prefix}_generation_data.npz", **generation_data)
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    gens = generation_data['gen']
    
    ax1 = axes[0, 0]
    ax1.plot(gens, generation_data['max_fitness'], 'b-', linewidth=1, alpha=0.5, label='Per-Gen Max')
    ax1.plot(gens, generation_data['global_best_fitness'], 'r-', linewidth=3, label='Global Best')
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax1.set_ylabel('Fitness (Latency Ratio)')
    ax1.set_title('Fitness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(gens, [l*1000 for l in generation_data['best_latency']], 'b-', linewidth=1, alpha=0.5, label='Per-Gen Best')
    ax2.plot(gens, [l*1000 for l in generation_data['global_best_latency']], 'r-', linewidth=3, label='Global Best')
    ax2.axhline(BASELINE_LATENCY * 1000, color='gray', linestyle='--', label=f'Baseline ({BASELINE_LATENCY*1000:.2f}ms)')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(gens, [s*1000 for s in generation_data['best_latency_std']], 'g-', linewidth=2)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Latency Std Dev (ms)')
    ax3.set_title('Latency Variance')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    if any(generation_data['best_power']):
        ax4.plot(gens, generation_data['best_power'], 'orange', linewidth=2)
        ax4.axhline(BASELINE_POWER, color='gray', linestyle='--')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('GPU Power (W)')
    ax4.set_title('GPU Power')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('DeepAR Latency Attack Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{prefix}_results.png", dpi=150)
    print(f"\nSaved: {prefix}_results.png")
    
    # Hall of Fame
    print(f"\n--- Hall of Fame (Top {len(hall_of_fame)}) ---")
    for i, hof in enumerate(hall_of_fame):
        print(f"  #{i+1}: Fitness={hof['fitness']:.4f} | Latency={hof['latency']*1000:.3f}ms")
        np.save(f"{prefix}_hof_{i+1}.npy", hof['solution'])
    
    print("\n" + "="*70)
    print("DeepAR Latency Attack Complete")
    print("="*70)
