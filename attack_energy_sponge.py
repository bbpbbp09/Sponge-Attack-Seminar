import time
import torch
import numpy as np
import pygad
import argparse
import matplotlib.pyplot as plt

from config import (CONTEXT_LEN, PREDICTION_LEN, NUM_FEATURES, DATA_PATH, ALL_FEATURES,
                    HIDDEN_SIZE, RNN_LAYERS, DEEPAR_MODEL_PATH,
                    POPULATION_SIZE, NUM_PARENTS_MATING, MUTATION_PERCENT, KEEP_ELITISM,
                    REPS_PER_MEASUREMENT, WARMUP_REPS, ENERGY_SAMPLE_INTERVAL)
from utils.power_monitor import PowerMonitor
from utils.data_loader import load_seed_data
from utils.metrics import measure_energy

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["chronos", "deepar", "patchtst"], default="chronos")
parser.add_argument("--mode", type=str, choices=["constrained", "extreme"], default="extreme")
parser.add_argument("--generations", type=int, default=50)
args = parser.parse_args()

MODEL_TYPE = args.model
MODE = args.mode
NUM_GENERATIONS = args.generations

print("="*70)
print("ENERGY SPONGE ATTACK - Maximizing GPU Power Consumption")
print("="*70)
print(f"Model: {MODEL_TYPE.upper()} | Mode: {MODE.upper()}")

device = "cuda" if torch.cuda.is_available() else "cpu"
power_monitor = PowerMonitor(ENERGY_SAMPLE_INTERVAL)

model = None
pipeline = None

if MODEL_TYPE == "chronos":
    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-base", device_map=device, torch_dtype=torch.float32)
elif MODEL_TYPE == "deepar":
    from models.deepar import DeepARLSTM
    checkpoint = torch.load(DEEPAR_MODEL_PATH, map_location=device)
    model = DeepARLSTM(input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=RNN_LAYERS, output_size=PREDICTION_LEN).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

seed_data, mean, std = load_seed_data(DATA_PATH, CONTEXT_LEN)
flat_seed = seed_data.flatten()

def make_prediction(input_array):
    if MODEL_TYPE == "chronos":
        if len(input_array.shape) == 2:
            univariate = input_array[:, 0]
        else:
            univariate = input_array
        inp = torch.tensor(univariate, dtype=torch.float32)
        pipeline.predict(inp, prediction_length=PREDICTION_LEN)
    elif MODEL_TYPE == "deepar":
        x = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            model(x)

print("\nMeasuring baseline...")
baseline_stats = measure_energy(make_prediction, seed_data, power_monitor, device, num_reps=20)
BASELINE_POWER = baseline_stats['avg_power']
BASELINE_ENERGY = baseline_stats['energy_per_inference']
BASELINE_CPU = baseline_stats['cpu_time_per_inference']
BASELINE_LATENCY = baseline_stats['latency']
print(f"Baseline Power: {BASELINE_POWER:.1f}W, Energy: {BASELINE_ENERGY*1000:.3f}mJ")

generation_data = {
    'gen': [], 'max_fitness': [], 'avg_fitness': [],
    'best_power': [], 'best_energy': [], 'best_latency': [],
    'global_best_fitness': [], 'global_max_power': [], 'global_max_energy': []
}
hall_of_fame = []
current_gen_fitness = []
current_gen_solutions = []
global_best = {'fitness': 0, 'power': 0, 'energy': 0, 'latency': 0, 'solution': None}
global_max_power = 0
global_max_energy = 0

def fitness_func(ga_instance, solution, solution_idx):
    global current_gen_fitness, current_gen_solutions, global_best, global_max_power, global_max_energy
    input_array = solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    adv_stats = measure_energy(make_prediction, input_array, power_monitor, device, REPS_PER_MEASUREMENT)
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
        global_best = {'fitness': fitness, 'power': adv_stats['avg_power'], 'energy': adv_stats['energy_per_inference'], 'latency': adv_stats['latency'], 'solution': solution.copy()}
    if adv_stats['avg_power'] > global_max_power:
        global_max_power = adv_stats['avg_power']
    if adv_stats['energy_per_inference'] > global_max_energy:
        global_max_energy = adv_stats['energy_per_inference']
    hall_of_fame.append({'fitness': fitness, 'power': adv_stats['avg_power'], 'solution': solution.copy()})
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
        generation_data['best_power'].append(best['power'])
        generation_data['best_energy'].append(best['energy'])
        generation_data['best_latency'].append(best['latency'])
        generation_data['global_best_fitness'].append(global_best['fitness'])
        generation_data['global_max_power'].append(global_max_power)
        generation_data['global_max_energy'].append(global_max_energy)
        power_change = ((best['power'] / BASELINE_POWER) - 1) * 100 if BASELINE_POWER > 0 else 0
        print(f"Gen {gen:3d}: Fitness={max_fit:.4f} (Power {power_change:+.1f}%)")
    current_gen_fitness = []
    current_gen_solutions = []

def energy_mutation(offspring, ga_instance):
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
                elif mutation_type < 0.92:
                    offspring[idx, gene_idx] = np.random.normal(0, 0.5)
                else:
                    offspring[idx, gene_idx] = np.float32(np.random.choice([1e-42, 1e-43, 1e-44, 1e-45, -1e-42, -1e-43, -1e-44, -1e-45]))
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
    n_per = POPULATION_SIZE // 5
    for _ in range(n_per):
        initial_population.append(np.random.uniform(-50, 50, n))
    for _ in range(n_per):
        x = np.zeros(n)
        x[::2] = np.random.uniform(10, 100, len(x[::2]))
        x[1::2] = np.random.uniform(-100, -10, len(x[1::2]))
        initial_population.append(x)
    for _ in range(n_per):
        initial_population.append(np.random.normal(0, 1, n))
    for _ in range(n_per):
        initial_population.append(np.random.uniform(-1000, 1000, n))
    for _ in range(POPULATION_SIZE - 4 * n_per):
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
    mutation_type=energy_mutation,
    initial_population=initial_population,
    gene_space=gene_space,
    on_generation=on_generation,
    keep_elitism=KEEP_ELITISM,
    suppress_warnings=True
)

if __name__ == "__main__":
    print("\nStarting Energy Sponge Attack\n")
    start_time = time.time()
    ga_instance.run()
    print(f"\nTotal GA time: {time.time() - start_time:.1f}s")

    best_solution, _, _ = ga_instance.best_solution()
    adv_input = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    verify_adv = measure_energy(make_prediction, adv_input, power_monitor, device, 50)
    verify_base = measure_energy(make_prediction, seed_data, power_monitor, device, 50)

    power_change = ((verify_adv['avg_power'] / verify_base['avg_power']) - 1) * 100 if verify_base['avg_power'] > 0 else 0
    energy_change = ((verify_adv['energy_per_inference'] / verify_base['energy_per_inference']) - 1) * 100 if verify_base['energy_per_inference'] > 0 else 0

    print("\n" + "="*70)
    print("FINAL RESULTS - ENERGY SPONGE ATTACK")
    print("="*70)
    print(f"Power: {verify_base['avg_power']:.1f}W -> {verify_adv['avg_power']:.1f}W ({power_change:+.1f}%)")
    print(f"Energy: {verify_base['energy_per_inference']*1000:.3f}mJ -> {verify_adv['energy_per_inference']*1000:.3f}mJ ({energy_change:+.1f}%)")
    print("="*70)

    prefix = f"{MODEL_TYPE}_energy_sponge"
    np.save(f"{prefix}_best_input.npy", best_solution)
    np.savez(f"{prefix}_generation_data.npz", **generation_data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    gens = generation_data['gen']
    axes[0, 0].plot(gens, generation_data['global_best_fitness'], 'r-', linewidth=2)
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].set_title('Fitness Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(gens, generation_data['global_max_power'], 'r-', linewidth=2)
    axes[0, 1].axhline(BASELINE_POWER, color='gray', linestyle='--')
    axes[0, 1].set_ylabel('Power (W)')
    axes[0, 1].set_title('Power Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(gens, [e*1000 for e in generation_data['global_max_energy']], 'r-', linewidth=2)
    axes[1, 0].axhline(BASELINE_ENERGY * 1000, color='gray', linestyle='--')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Energy (mJ)')
    axes[1, 0].set_title('Energy Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(gens, [l*1000 for l in generation_data['best_latency']], 'c-', linewidth=2)
    axes[1, 1].axhline(BASELINE_LATENCY * 1000, color='gray', linestyle='--')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Latency (ms)')
    axes[1, 1].set_title('Latency (should be flat)')
    axes[1, 1].grid(True, alpha=0.3)
    plt.suptitle(f'{MODEL_TYPE.upper()} Energy Sponge Attack Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{prefix}_results.png", dpi=150)
    print(f"\nSaved: {prefix}_results.png")

    for i, hof in enumerate(hall_of_fame):
        np.save(f"{prefix}_hof_{i+1}.npy", hof['solution'])
