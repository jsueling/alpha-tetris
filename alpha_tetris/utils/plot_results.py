"""Plot results from different training runs and compare them to a random baseline."""

import json
import numpy as np
import matplotlib.pyplot as plt

DIR = "./out/"

RANDOM_AGENT_FP = DIR + "f3_random_async_seed_3_results.npy"

def to_serialisable(val):
    """Convert numpy types to native Python types for JSON serialisation."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    return val

print(
    json.dumps(
        np.load(RANDOM_AGENT_FP, allow_pickle=True).item(),
        indent=2,
        default=to_serialisable
    )
)

# After running 1000 episodes with a randomly initialised model that does not learn,
# the running average benchmark episode score converged to approximately 0.87.
# This value serves as a baseline for comparison with other models
# and is plotted as a flat reference line in subsequent analyses.

regular = np.load(DIR + "f1_async_seed_3_results.npy", allow_pickle=True).item()
unsupervised = np.load(DIR + "f3_unsupervised_async_seed_3_results.npy", allow_pickle=True).item()
ensemble = np.load(DIR + "f2_ensemble_seed_0_results.npy", allow_pickle=True).item()
known = np.load(DIR + "f3_known_async_seed_3_results.npy", allow_pickle=True).item()

def score_per_benchmark_episode(lst):
    """Compute the average score per benchmark episode."""
    benchmark_episodes = 50
    return [x / benchmark_episodes for x in lst]

# Plot regular, unsupervised, and ensemble together
plt.figure(figsize=(10, 6))
plt.plot(score_per_benchmark_episode(regular['benchmark_scores']), label='Standard')
plt.plot(score_per_benchmark_episode(unsupervised['benchmark_scores']), label='Unsupervised')
plt.plot(score_per_benchmark_episode(ensemble['benchmark_scores']), label='Ensemble')
plt.axhline(y=0.87, color='r', linestyle='--', label='Random Baseline')
plt.xlabel('Iteration')
plt.ylabel('Avg score per benchmark episode')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot known separately
plt.figure(figsize=(10, 6))
plt.plot(score_per_benchmark_episode(known['benchmark_scores']), label='Known')
plt.axhline(y=0.87, color='r', linestyle='--', label='Random Baseline')
plt.xlabel('Iteration')
plt.ylabel('Avg score per benchmark episode')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
