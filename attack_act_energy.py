import time
import torch
import numpy as np
import pygad
import pandas as pd
import matplotlib.pyplot as plt

from config import (CONTEXT_LEN, PREDICTION_LEN, NUM_FEATURES, ACT_HIDDEN_SIZE,
                    ACT_MODEL_PATH, DATA_PATH, ALL_FEATURES, POPULATION_SIZE,
                    NUM_PARENTS_MATING, MUTATION_PERCENT, ENERGY_SAMPLE_INTERVAL)
from models.act import ACTModel
from utils.power_monitor import PowerMonitor
from utils.data_loader import get_normalization_params
from utils.metrics import measure_energy

print("="*70)
print("ACT-LSTM ENERGY ATTACK")
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

def make_prediction(input_array):
    x = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x)

print("\nMeasuring baseline...")
baseline_stats = measure_energy(make_prediction, seed_data, power_monitor, device, num_reps=20)
BASELINE_ENERGY = baseline_stats['energy_per_inference']
BASELINE_POWER = baseline_stats['avg_power']
BASELINE_LATENCY = baseline_stats['latency']
print(f"Baseline Power: {BASELINE_POWER:.1f}W")

generation_data = {'gen': [], 'best_energy': [], 'best_power': [], 'best_latency': []}

def fitness_func(ga_instance, solution, solution_idx):
    input_array = solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    stats = measure_energy(make_prediction, input_array, power_monitor, device, 20)
    return stats['energy_per_inference'] / BASELINE_ENERGY if BASELINE_ENERGY > 0 else 1.0

def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best_solution, best_fitness, _ = ga_instance.best_solution()
    input_array = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    stats = measure_energy(make_prediction, input_array, power_monitor, device, 20)
    generation_data['gen'].append(gen)
    generation_data['best_energy'].append(stats['energy_per_inference'])
    generation_data['best_power'].append(stats['avg_power'])
    generation_data['best_latency'].append(stats['latency'])
    print(f"Gen {gen:3d}: Energy={stats['energy_per_inference']*1000:.3f}mJ")

if __name__ == "__main__":
    print("\nStarting ACT-LSTM Energy Attack...")

    print("Warming up...", end="", flush=True)
    dummy = torch.randn(1, CONTEXT_LEN, NUM_FEATURES).to(device)
    for _ in range(20):
        model(dummy)
    print(" Done.")

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
    adv_input = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    adv_tensor = torch.tensor(adv_input, dtype=torch.float32).unsqueeze(0).to(device)
    base_tensor = torch.tensor(seed_data, dtype=torch.float32).unsqueeze(0).to(device)

    adv_stats = measure_energy(make_prediction, adv_input, power_monitor, device, 20)
    base_stats = measure_energy(make_prediction, seed_data, power_monitor, device, 20)

    print("="*70)
    print(f"{'Metric':<20} {'Baseline':>15} {'Adversarial':>15} {'Change':>10}")
    print("-"*70)
    print(f"{'Energy (J)':<20} {base_stats['energy_per_inference']:.4f} {adv_stats['energy_per_inference']:.4f} {((adv_stats['energy_per_inference']/base_stats['energy_per_inference'])-1)*100:+.1f}%")
    print(f"{'Power (W)':<20} {base_stats['avg_power']:.1f} {adv_stats['avg_power']:.1f} {((adv_stats['avg_power']/base_stats['avg_power'])-1)*100:+.1f}%")
    print(f"{'Latency (ms)':<20} {base_stats['latency']*1000:.1f} {adv_stats['latency']*1000:.1f} {((adv_stats['latency']/base_stats['latency'])-1)*100:+.1f}%")
    print("="*70)

    np.save("act_energy_best_input.npy", best_solution)
