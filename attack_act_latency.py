import time
import torch
import numpy as np
import pygad
import pandas as pd

from config import (CONTEXT_LEN, PREDICTION_LEN, NUM_FEATURES, ACT_HIDDEN_SIZE,
                    ACT_MODEL_PATH, DATA_PATH, ALL_FEATURES, MAX_PONDERS,
                    POPULATION_SIZE, NUM_PARENTS_MATING, MUTATION_PERCENT)
from models.act import ACTModel
from utils.data_loader import get_normalization_params

print("="*70)
print("ACT-LSTM LATENCY ATTACK")
print("="*70)
print(f"Goal: Find inputs that trigger {MAX_PONDERS} ponder steps.")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ACTModel(NUM_FEATURES, ACT_HIDDEN_SIZE, PREDICTION_LEN).to(device)
model.load_state_dict(torch.load(ACT_MODEL_PATH, map_location=device))
model.eval()

df = pd.read_csv(DATA_PATH).sort_values('Time')
data = df[ALL_FEATURES].values.astype(np.float32)
mean, std = get_normalization_params(data)
seed_data = (data[:CONTEXT_LEN] - mean) / std
flat_seed = seed_data.flatten()

def measure_inference(input_tensor):
    start = time.perf_counter()
    with torch.no_grad():
        _, avg_ponder = model(input_tensor)
    end = time.perf_counter()
    return end - start, avg_ponder.item()

def fitness_func(ga_instance, solution, solution_idx):
    input_tensor = torch.tensor(solution.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    _, ponder_steps = measure_inference(input_tensor)
    return float(ponder_steps ** 2)

def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best_solution, _, _ = ga_instance.best_solution()
    input_tensor = torch.tensor(best_solution.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    latency, ponder = measure_inference(input_tensor)
    print(f"Gen {gen}: Ponder={ponder:.2f}/{MAX_PONDERS} | Latency={latency*1000:.3f}ms")

if __name__ == "__main__":
    print("\nStarting ACT-LSTM Latency Attack...")

    print("Warming up...", end="", flush=True)
    dummy = torch.randn(1, CONTEXT_LEN, NUM_FEATURES).to(device)
    for _ in range(50):
        model(dummy)
    print(" Done.")

    base_tensor = torch.tensor(seed_data.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    base_lat, base_ponder = measure_inference(base_tensor)
    print(f"Baseline: Ponders={base_ponder:.2f}, Latency={base_lat*1000:.3f}ms")

    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=NUM_PARENTS_MATING,
        fitness_func=fitness_func,
        sol_per_pop=POPULATION_SIZE,
        num_genes=len(flat_seed),
        initial_population=[flat_seed] * POPULATION_SIZE + np.random.normal(0, 0.5, (POPULATION_SIZE, len(flat_seed))),
        on_generation=on_generation,
        suppress_warnings=True
    )

    ga_instance.run()

    best_solution, _, _ = ga_instance.best_solution()
    adv_tensor = torch.tensor(best_solution.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    adv_lat, adv_ponder = measure_inference(adv_tensor)

    print("\n" + "="*70)
    print("ATTACK RESULTS")
    print("="*70)
    print(f"Baseline Ponders: {base_ponder:.2f}")
    print(f"Attack Ponders:   {adv_ponder:.2f} (Max: {MAX_PONDERS})")
    print(f"Baseline Latency: {base_lat*1000:.3f} ms")
    print(f"Attack Latency:   {adv_lat*1000:.3f} ms")

    slowdown = adv_lat / base_lat if base_lat > 0 else 1.0
    print(f"Slowdown Factor:  {slowdown:.2f}x")

    if slowdown > 1.5:
        print("SUCCESS: Logic-based Latency Attack Confirmed.")
    else:
        print("FAILURE: Could not trigger significant latency increase.")

    np.save("act_latency_best_input.npy", best_solution)
