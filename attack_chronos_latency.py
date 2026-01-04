import time
import torch
import numpy as np
import pygad
import argparse
import matplotlib.pyplot as plt
from chronos import ChronosPipeline

from config import (CONTEXT_LEN, PREDICTION_LEN, NUM_FEATURES, DATA_PATH, ALL_FEATURES,
                    POPULATION_SIZE, NUM_PARENTS_MATING, MUTATION_PERCENT, KEEP_ELITISM,
                    REPS_PER_MEASUREMENT, WARMUP_REPS)
from utils.power_monitor import PowerMonitor
from utils.data_loader import load_seed_data
from utils.metrics import measure_latency

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, default=50)
parser.add_argument("--mode", type=str, choices=["constrained", "extreme"], default="extreme")
args = parser.parse_args()

NUM_GENERATIONS = args.generations
MODE = args.mode

print("="*70)
print("CHRONOS LATENCY ATTACK")
print("="*70)

device = "cuda" if torch.cuda.is_available() else "cpu"
power_monitor = PowerMonitor(0.001)

print("\nLoading Chronos model...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map=device,
    torch_dtype=torch.float32,
)

seed_data, mean, std = load_seed_data(DATA_PATH, CONTEXT_LEN)
flat_seed = seed_data.flatten()

def make_prediction(input_array):
    if len(input_array.shape) == 2:
        univariate = input_array[:, 0]
    else:
        univariate = input_array
    inp = torch.tensor(univariate, dtype=torch.float32)
    pipeline.predict(inp, prediction_length=PREDICTION_LEN)

print("\nMeasuring baseline...")
baseline_stats = measure_latency(make_prediction, seed_data, power_monitor, device, num_reps=50)
BASELINE_LATENCY = baseline_stats['latency']
BASELINE_POWER = baseline_stats['avg_power']
print(f"Baseline Latency: {BASELINE_LATENCY*1000:.3f}ms")

generation_data = {
    'gen': [], 'max_fitness': [], 'avg_fitness': [],
    'best_latency': [], 'best_power': [], 'best_cpu_percent': [],
    'global_best_fitness': [], 'global_max_latency': []
}
hall_of_fame = []
current_gen_fitness = []
current_gen_solutions = []
global_best = {'fitness': 0, 'latency': 0, 'power': 0, 'solution': None}
global_max_latency = 0

def fitness_func(ga_instance, solution, solution_idx):
    global current_gen_fitness, current_gen_solutions, global_best, global_max_latency
    input_array = solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    stats = measure_latency(make_prediction, input_array, power_monitor, device, REPS_PER_MEASUREMENT)
    latency_ratio = stats['latency'] / BASELINE_LATENCY if BASELINE_LATENCY > 0 else 1.0
    fitness = latency_ratio * 100
    current_gen_fitness.append(fitness)
    current_gen_solutions.append({
        'fitness': fitness, 'latency': stats['latency'],
        'power': stats['avg_power'], 'cpu_percent': stats['avg_cpu_percent'],
        'solution': solution.copy()
    })
    if fitness > global_best['fitness']:
        global_best = {'fitness': fitness, 'latency': stats['latency'], 'power': stats['avg_power'], 'solution': solution.copy()}
    if stats['latency'] > global_max_latency:
        global_max_latency = stats['latency']
    hall_of_fame.append({'fitness': fitness, 'latency': stats['latency'], 'solution': solution.copy()})
    hall_of_fame.sort(key=lambda x: x['fitness'], reverse=True)
    while len(hall_of_fame) > 10:
        hall_of_fame.pop()
    return float(fitness)

def on_generation(ga_instance):
    global current_gen_fitness, current_gen_solutions
    gen = ga_instance.generations_completed
    if current_gen_fitness:
        max_fit = max(current_gen_fitness)
        best = max(current_gen_solutions, key=lambda x: x['fitness'])
        generation_data['gen'].append(gen)
        generation_data['max_fitness'].append(max_fit)
        generation_data['avg_fitness'].append(np.mean(current_gen_fitness))
        generation_data['best_latency'].append(best['latency'])
        generation_data['best_power'].append(best['power'])
        generation_data['best_cpu_percent'].append(best.get('cpu_percent', 0))
        generation_data['global_best_fitness'].append(global_best['fitness'])
        generation_data['global_max_latency'].append(global_max_latency)
        lat_change = ((best['latency'] / BASELINE_LATENCY) - 1) * 100 if BASELINE_LATENCY > 0 else 0
        print(f"Gen {gen:3d}: Fitness={max_fit:.4f} (Latency {lat_change:+.1f}%)")
    current_gen_fitness = []
    current_gen_solutions = []

def latency_mutation(offspring, ga_instance):
    bin_hop_min = 0.1
    for idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.random() < MUTATION_PERCENT / 100:
                mutation_type = np.random.random()
                val = offspring[idx, gene_idx]
                if mutation_type < 0.50:
                    direction = np.random.choice([1, -1])
                    jump = np.random.uniform(bin_hop_min, 2.0)
                    offspring[idx, gene_idx] = val + (direction * jump)
                elif mutation_type < 0.75:
                    offspring[idx, gene_idx] = np.random.uniform(-1e4, 1e4)
                else:
                    offspring[idx, gene_idx] = np.random.normal(0, 0.5)
    return offspring

def time_slice_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        p1 = parents[idx % parents.shape[0], :].reshape(CONTEXT_LEN, NUM_FEATURES)
        p2 = parents[(idx + 1) % parents.shape[0], :].reshape(CONTEXT_LEN, NUM_FEATURES)
        crossover_pt = np.random.randint(1, CONTEXT_LEN)
        child = np.vstack([p1[:crossover_pt, :], p2[crossover_pt:, :]])
        offspring.append(child.flatten())
        idx += 1
    return np.array(offspring)

print("\nInitializing population...")
initial_population = []
n = len(flat_seed)
if MODE == "constrained":
    for _ in range(POPULATION_SIZE):
        initial_population.append(flat_seed + np.random.normal(0, 0.5, n))
    gene_space = {'low': -10.0, 'high': 10.0}
else:
    n_per = POPULATION_SIZE // 4
    for _ in range(n_per):
        initial_population.append(np.random.uniform(-50, 50, n))
    for _ in range(n_per):
        x = np.zeros(n)
        x[::2] = np.random.uniform(10, 100, len(x[::2]))
        x[1::2] = np.random.uniform(-100, -10, len(x[1::2]))
        initial_population.append(x)
    for _ in range(n_per):
        initial_population.append(np.random.normal(0, 1, n))
    for _ in range(POPULATION_SIZE - 3 * n_per):
        initial_population.append(flat_seed + np.random.normal(0, 10, n))
    gene_space = None
initial_population = np.array(initial_population)

ga_instance = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=NUM_PARENTS_MATING,
    fitness_func=fitness_func,
    sol_per_pop=POPULATION_SIZE,
    num_genes=len(flat_seed),
    parent_selection_type="tournament",
    K_tournament=3,
    crossover_type=time_slice_crossover,
    mutation_type=latency_mutation,
    initial_population=initial_population,
    gene_space=gene_space,
    on_generation=on_generation,
    keep_elitism=KEEP_ELITISM,
    suppress_warnings=True
)

if __name__ == "__main__":
    print("\nStarting Chronos Latency Attack\n")
    start_time = time.time()
    ga_instance.run()
    print(f"\nTotal GA time: {time.time() - start_time:.1f}s")

    best_solution, _, _ = ga_instance.best_solution()
    adv_input = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    verify_adv = measure_latency(make_prediction, adv_input, power_monitor, device, 100)
    verify_base = measure_latency(make_prediction, seed_data, power_monitor, device, 100)
    lat_change = ((verify_adv['latency'] / verify_base['latency']) - 1) * 100

    print("\n" + "="*70)
    print("FINAL RESULTS - CHRONOS LATENCY ATTACK")
    print("="*70)
    print(f"Latency: {verify_base['latency']*1000:.3f}ms -> {verify_adv['latency']*1000:.3f}ms ({lat_change:+.1f}%)")
    print("="*70)

    prefix = "chronos_latency"
    np.save(f"{prefix}_best_input.npy", best_solution)
    np.savez(f"{prefix}_generation_data.npz", **generation_data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    gens = generation_data['gen']
    axes[0, 0].plot(gens, generation_data['global_best_fitness'], 'r-', linewidth=2)
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].set_title('Fitness Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(gens, [l*1000 for l in generation_data['global_max_latency']], 'r-', linewidth=2)
    axes[0, 1].axhline(BASELINE_LATENCY * 1000, color='gray', linestyle='--')
    axes[0, 1].set_ylabel('Latency (ms)')
    axes[0, 1].set_title('Latency Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(gens, generation_data['best_power'], 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Power (W)')
    axes[1, 0].set_title('Power Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(gens, generation_data['best_cpu_percent'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('CPU %')
    axes[1, 1].set_title('CPU Utilization')
    axes[1, 1].grid(True, alpha=0.3)
    plt.suptitle('Chronos Latency Attack Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{prefix}_results.png", dpi=150)
    print(f"\nSaved: {prefix}_results.png")

    for i, hof in enumerate(hall_of_fame):
        np.save(f"{prefix}_hof_{i+1}.npy", hof['solution'])
