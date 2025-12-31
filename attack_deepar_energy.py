"""
DeepAR-Style LSTM Energy Attack
================================
Genetic algorithm attack maximizing GPU power consumption.
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
parser = argparse.ArgumentParser(description="DeepAR Energy Attack")
parser.add_argument("--generations", type=int, default=50, help="Number of generations")
parser.add_argument("--mode", type=str, choices=["constrained", "extreme"], default="extreme")
args = parser.parse_args()

NUM_GENERATIONS = args.generations
MODE = args.mode

print("="*70)
print("DeepAR ENERGY ATTACK - Maximizing GPU Power Consumption")
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

ENERGY_SAMPLE_INTERVAL = 0.01
REPS_PER_MEASUREMENT = 10
WARMUP_REPS = 5

POPULATION_SIZE = 20
NUM_PARENTS_MATING = 10
MUTATION_PERCENT = 35
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
            return {'avg_power': 0, 'max_power': 0, 'energy_joules': 0}
        
        powers = [r[1] for r in self.readings]
        times = [r[0] for r in self.readings]
        
        energy = 0
        for i in range(1, len(self.readings)):
            dt = times[i] - times[i-1]
            avg_power = (powers[i] + powers[i-1]) / 2
            energy += avg_power * dt
        
        return {
            'avg_power': np.mean(powers),
            'max_power': np.max(powers),
            'energy_joules': energy,
            'num_samples': len(powers)
        }

power_monitor = GPUPowerMonitor(ENERGY_SAMPLE_INTERVAL)

# =============================================================================
# LOAD MODEL
# =============================================================================
print("\nLoading DeepAR model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Create model
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

# Normalize using saved params
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
    """Run inference on LSTM model."""
    x = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x)
        return output

# Sanity check
print("Running sanity check...")
try:
    make_prediction(seed_data)
    print("✓ Sanity check passed.")
except Exception as e:
    print(f"✗ Sanity check failed: {e}")
    exit(1)

# =============================================================================
# ENERGY MEASUREMENT
# =============================================================================
def measure_energy(input_array, num_reps=REPS_PER_MEASUREMENT):
    for _ in range(WARMUP_REPS):
        try:
            make_prediction(input_array)
        except:
            return {'avg_power': 0, 'max_power': 0, 'energy_per_inference': 0, 'latency': 0.001}
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    if power_monitor.gpu_available:
        power_monitor.start()
    
    cpu_start = CURRENT_PROCESS.cpu_times()
    cpu_start_total = cpu_start.user + cpu_start.system
    start_time = time.perf_counter()
    
    for _ in range(num_reps):
        try:
            make_prediction(input_array)
        except:
            if power_monitor.gpu_available:
                power_monitor.stop()
            return {'avg_power': 0, 'max_power': 0, 'energy_per_inference': 0, 'latency': 0.001}
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    cpu_end = CURRENT_PROCESS.cpu_times()
    cpu_end_total = cpu_end.user + cpu_end.system
    
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
    'global_best_fitness': [], 'global_best_power': [], 'global_best_energy': []
}

hall_of_fame = []
current_gen_fitness = []
current_gen_solutions = []
global_best = {'fitness': 0, 'power': 0, 'energy': 0, 'latency': 0, 'solution': None}

# =============================================================================
# FITNESS FUNCTION
# =============================================================================
def fitness_func(ga_instance, solution, solution_idx):
    global current_gen_fitness, current_gen_solutions, global_best
    
    input_array = solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    adv_stats = measure_energy(input_array, num_reps=REPS_PER_MEASUREMENT)
    
    power_ratio = adv_stats['avg_power'] / BASELINE_POWER if BASELINE_POWER > 0 else 1.0
    cpu_ratio = adv_stats['cpu_time_per_inference'] / BASELINE_CPU if BASELINE_CPU > 0 else 1.0
    
    if power_monitor.gpu_available:
        fitness = 0.7 * power_ratio + 0.3 * cpu_ratio
    else:
        lat_ratio = adv_stats['latency'] / BASELINE_LATENCY if BASELINE_LATENCY > 0 else 1.0
        fitness = 0.6 * cpu_ratio + 0.4 * lat_ratio
    
    current_gen_fitness.append(fitness)
    current_gen_solutions.append({
        'fitness': fitness, 'power': adv_stats['avg_power'],
        'energy': adv_stats['energy_per_inference'], 'latency': adv_stats['latency'],
        'solution': solution.copy()
    })
    
    if fitness > global_best['fitness']:
        global_best = {
            'fitness': fitness, 'power': adv_stats['avg_power'],
            'energy': adv_stats['energy_per_inference'], 'latency': adv_stats['latency'],
            'solution': solution.copy()
        }
    
    hall_of_fame.append({
        'fitness': fitness, 'power': adv_stats['avg_power'],
        'energy': adv_stats['energy_per_inference'], 'solution': solution.copy()
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
# MUTATION
# =============================================================================
def energy_mutation(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.random() < MUTATION_PERCENT / 100:
                mutation_type = np.random.random()
                
                if mutation_type < 0.25:
                    scale = np.random.choice([1, 2, 5, 10, 20])
                    offspring[idx, gene_idx] = np.abs(np.random.normal(0, scale))
                elif mutation_type < 0.45:
                    value_range = np.random.uniform(1, 50)
                    offspring[idx, gene_idx] = np.random.uniform(-value_range, value_range)
                elif mutation_type < 0.60:
                    offspring[idx, gene_idx] = np.random.uniform(-1000, 1000)
                elif mutation_type < 0.75:
                    if gene_idx > 0:
                        prev_val = offspring[idx, gene_idx - 1]
                        offspring[idx, gene_idx] = -prev_val * np.random.uniform(0.5, 2.0)
                    else:
                        offspring[idx, gene_idx] = np.random.uniform(-100, 100)
                elif mutation_type < 0.85:
                    offspring[idx, gene_idx] = np.random.uniform(-1e4, 1e4)
                elif mutation_type < 0.92:
                    offspring[idx, gene_idx] = np.random.normal(0, 0.5)
                else:
                    offspring[idx, gene_idx] = np.float32(np.random.choice([
                        1e-42, 1e-43, 1e-44, 1e-45, -1e-42, -1e-43, -1e-44, -1e-45
                    ]))
    
    return offspring

# =============================================================================
# INITIAL POPULATION
# =============================================================================
print("\nInitializing population...")
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
        initial_population.append(np.random.uniform(-50, 50, n))
    for _ in range(n_per_strategy):
        x = np.zeros(n)
        x[::2] = np.random.uniform(10, 100, len(x[::2]))
        x[1::2] = np.random.uniform(-100, -10, len(x[1::2]))
        initial_population.append(x)
    for _ in range(n_per_strategy):
        initial_population.append(np.random.normal(0, 1, n))
    for _ in range(n_per_strategy):
        initial_population.append(np.random.uniform(-1000, 1000, n))
    remaining = POPULATION_SIZE - 4 * n_per_strategy
    for _ in range(remaining):
        initial_population.append(flat_seed + np.random.normal(0, 10, n))
    gene_space = None

initial_population = np.array(initial_population)

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
    print("Starting DeepAR Energy Attack")
    print("="*70 + "\n")
    
    start_time = time.time()
    ga_instance.run()
    total_time = time.time() - start_time
    
    print(f"\nTotal GA time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    best_solution, best_fitness, _ = ga_instance.best_solution()
    
    print("\nRe-verifying best solution...")
    adv_input = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    
    verify_adv = measure_energy(adv_input, num_reps=50)
    verify_base = measure_energy(seed_data, num_reps=50)
    
    power_change = ((verify_adv['avg_power'] / verify_base['avg_power']) - 1) * 100 if verify_base['avg_power'] > 0 else 0
    energy_change = ((verify_adv['energy_per_inference'] / verify_base['energy_per_inference']) - 1) * 100 if verify_base['energy_per_inference'] > 0 else 0
    lat_change = ((verify_adv['latency'] / verify_base['latency']) - 1) * 100 if verify_base['latency'] > 0 else 0
    
    print("\n" + "="*70)
    print("FINAL RESULTS - DeepAR ENERGY ATTACK")
    print("="*70)
    print(f"{'Metric':<25} {'Baseline':>15} {'Adversarial':>15} {'Change':>12}")
    print("-"*70)
    print(f"{'GPU Power (W)':<25} {verify_base['avg_power']:>12.1f} W {verify_adv['avg_power']:>12.1f} W {power_change:>+10.1f}%")
    print(f"{'Energy/Inference (mJ)':<25} {verify_base['energy_per_inference']*1000:>12.3f} mJ {verify_adv['energy_per_inference']*1000:>12.3f} mJ {energy_change:>+10.1f}%")
    print(f"{'Latency (ms)':<25} {verify_base['latency']*1000:>12.2f} ms {verify_adv['latency']*1000:>12.2f} ms {lat_change:>+10.1f}%")
    print("="*70)
    
    # Save outputs
    prefix = "deepar_energy"
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
    ax1.set_ylabel('Fitness (Power Ratio)')
    ax1.set_title('Fitness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    if any(generation_data['best_power']):
        ax2.plot(gens, generation_data['best_power'], 'b-', linewidth=1, alpha=0.5, label='Per-Gen Best')
        ax2.plot(gens, generation_data['global_best_power'], 'r-', linewidth=3, label='Global Best')
        ax2.axhline(BASELINE_POWER, color='gray', linestyle='--', label=f'Baseline ({BASELINE_POWER:.1f}W)')
    ax2.set_ylabel('GPU Power (W)')
    ax2.set_title('GPU Power Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    if any(generation_data['best_energy']):
        ax3.plot(gens, [e*1000 for e in generation_data['best_energy']], 'b-', linewidth=1, alpha=0.5)
        ax3.plot(gens, [e*1000 for e in generation_data['global_best_energy']], 'r-', linewidth=3)
        ax3.axhline(BASELINE_ENERGY * 1000, color='gray', linestyle='--')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Energy (mJ/inference)')
    ax3.set_title('Energy per Inference')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(gens, [l*1000 for l in generation_data['best_latency']], 'c-', linewidth=2)
    ax4.axhline(BASELINE_LATENCY * 1000, color='gray', linestyle='--')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Latency (ms)')
    ax4.set_title('Latency')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('DeepAR Energy Attack Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{prefix}_results.png", dpi=150)
    print(f"\nSaved: {prefix}_results.png")
    
    print("\n" + "="*70)
    print("DeepAR Energy Attack Complete")
    print("="*70)
