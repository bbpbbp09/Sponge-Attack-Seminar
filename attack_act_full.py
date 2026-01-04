import time
import torch
import numpy as np
import pygad
import pandas as pd
import matplotlib.pyplot as plt

from config import (CONTEXT_LEN, PREDICTION_LEN, NUM_FEATURES, ACT_HIDDEN_SIZE,
                    ACT_MODEL_PATH, DATA_PATH, ALL_FEATURES, MAX_PONDERS,
                    POPULATION_SIZE, NUM_PARENTS_MATING, MUTATION_PERCENT)
from models.act import ACTModel
from utils.power_monitor import PowerMonitor
from utils.data_loader import get_normalization_params
from utils.metrics import measure_energy

print("="*70)
print("ACT-LSTM FULL ATTACK & VISUALIZATION")
print("="*70)

device = "cuda" if torch.cuda.is_available() else "cpu"
power_monitor = PowerMonitor(0.01)

model = ACTModel(NUM_FEATURES, ACT_HIDDEN_SIZE, PREDICTION_LEN).to(device)
model.load_state_dict(torch.load(ACT_MODEL_PATH, map_location=device))
model.eval()

df = pd.read_csv(DATA_PATH).sort_values('Time')
data = df[ALL_FEATURES].values.astype(np.float32)
mean, std = get_normalization_params(data)
seed_data = (data[:CONTEXT_LEN] - mean) / std
flat_seed = seed_data.flatten()

generation_data = {'gen': [], 'best_fitness': [], 'best_latency': [], 'best_ponder': [], 'best_power': []}

def measure_inference(input_tensor, num_reps=20):
    power_monitor.start()
    latencies = []
    ponders = []
    for _ in range(num_reps):
        start = time.perf_counter()
        with torch.no_grad():
            _, ponder = model(input_tensor)
        end = time.perf_counter()
        latencies.append(end - start)
        ponders.append(ponder.item())
    power_monitor.stop()
    stats = power_monitor.get_energy_stats()
    return {
        'latency': np.mean(latencies),
        'ponder': np.mean(ponders),
        'power': stats['avg_power']
    }

def fitness_func(ga_instance, solution, solution_idx):
    input_tensor = torch.tensor(solution.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    stats = measure_inference(input_tensor)
    return float(stats['ponder'] ** 2)

def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best_solution, best_fitness, _ = ga_instance.best_solution()
    input_tensor = torch.tensor(best_solution.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    stats = measure_inference(input_tensor)
    generation_data['gen'].append(gen)
    generation_data['best_fitness'].append(best_fitness)
    generation_data['best_latency'].append(stats['latency'])
    generation_data['best_ponder'].append(stats['ponder'])
    generation_data['best_power'].append(stats['power'])
    print(f"Gen {gen:3d}: Ponder={stats['ponder']:.2f}/{MAX_PONDERS} | Latency={stats['latency']*1000:.3f}ms | Power={stats['power']:.1f}W")

if __name__ == "__main__":
    print("\nStarting ACT-LSTM Full Attack...")

    print("Warming up...", end="", flush=True)
    dummy = torch.randn(1, CONTEXT_LEN, NUM_FEATURES).to(device)
    for _ in range(50):
        model(dummy)
    print(" Done.")

    base_tensor = torch.tensor(seed_data.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    base_stats = measure_inference(base_tensor)
    print(f"Baseline: Ponder={base_stats['ponder']:.2f}, Latency={base_stats['latency']*1000:.3f}ms, Power={base_stats['power']:.1f}W")

    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=NUM_PARENTS_MATING,
        fitness_func=fitness_func,
        sol_per_pop=POPULATION_SIZE,
        num_genes=len(flat_seed),
        initial_population=[flat_seed + np.random.normal(0, 0.5, len(flat_seed)) for _ in range(POPULATION_SIZE)],
        on_generation=on_generation,
        suppress_warnings=True
    )

    ga_instance.run()

    best_solution, _, _ = ga_instance.best_solution()
    adv_tensor = torch.tensor(best_solution.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    adv_stats = measure_inference(adv_tensor)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Baseline Ponders: {base_stats['ponder']:.2f} | Attack Ponders: {adv_stats['ponder']:.2f}")
    print(f"Baseline Latency: {base_stats['latency']*1000:.3f}ms | Attack Latency: {adv_stats['latency']*1000:.3f}ms")
    print(f"Baseline Power: {base_stats['power']:.1f}W | Attack Power: {adv_stats['power']:.1f}W")

    slowdown = adv_stats['latency'] / base_stats['latency'] if base_stats['latency'] > 0 else 1.0
    print(f"Slowdown Factor: {slowdown:.2f}x")
    print("="*70)

    np.save("act_latency_best_input.npy", best_solution)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    gens = generation_data['gen']
    axes[0, 0].plot(gens, generation_data['best_ponder'], 'b-', linewidth=2)
    axes[0, 0].axhline(base_stats['ponder'], color='gray', linestyle='--')
    axes[0, 0].axhline(MAX_PONDERS, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_ylabel('Ponder Steps')
    axes[0, 0].set_title('Ponder Steps Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(gens, [l*1000 for l in generation_data['best_latency']], 'r-', linewidth=2)
    axes[0, 1].axhline(base_stats['latency'] * 1000, color='gray', linestyle='--')
    axes[0, 1].set_ylabel('Latency (ms)')
    axes[0, 1].set_title('Latency Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(gens, generation_data['best_power'], 'orange', linewidth=2)
    axes[1, 0].axhline(base_stats['power'], color='gray', linestyle='--')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Power (W)')
    axes[1, 0].set_title('Power Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(gens, generation_data['best_fitness'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Fitness')
    axes[1, 1].set_title('Fitness Evolution')
    axes[1, 1].grid(True, alpha=0.3)
    plt.suptitle('ACT-LSTM Full Attack Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("act_latency_results_full.png", dpi=150)
    print(f"\nSaved: act_latency_results_full.png")
