from simulator import *
from pid_controller import *
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path

# run N episondes with m steps per episode
N = 10  # Number of episodes
sim = CoupledOscillators(config_file="experiment_config.json")
pid = PIDController(config_file="experiment_config.json")

dataset = []

# loop over episodes
print(f"Starting data collection: {N} episodes x 1000 steps")
for i in range(N):
    print(f"Episode {i+1}/{N}")
    data_buffer = []
    obs, info = sim.reset()

    # loop over steps
    for m in range(1000):
        current_position = obs[0]  # Extract scalar from observation array
        action = pid.update(0.0, current_position)
        next_obs, reward, term, trunc, info = sim.step(action)
        data_buffer.append([obs, action, next_obs, reward])
        
        # Update obs for next iteration
        obs = next_obs

    # buffer episode
    dataset.append(data_buffer)

# Create output directory
output_dir = Path("./archived_datasets")
output_dir.mkdir(parents=True, exist_ok=True)

# Generate timestamp for unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Convert dataset to numpy array for consistent format
dataset_array = np.array(dataset, dtype=object)

# Create metadata
metadata = {
    "timestamp": timestamp,
    "num_episodes": N,
    "steps_per_episode": 1000,
    "simulation_config": sim.get_system_info(),
    "pid_config": {
        "kp": pid.kp,
        "ki": pid.ki, 
        "kd": pid.kd,
        "output_limits": pid.output_limits,
        "integral_limits": pid.integral_limits
    },
    "data_format": "List of episodes, each episode contains [state, action, next_state, reward] tuples",
    "total_samples": sum(len(episode) for episode in dataset)
}


#Save as pickle (preserves exact Python objects)
base_filename = f"oscillator_dataset_{timestamp}"
pickle_file = output_dir / f"{base_filename}.pkl"
with open(pickle_file, 'wb') as f:
    pickle.dump({
        'dataset': dataset,
        'metadata': metadata
    }, f)

print(f"Dataset archived successfully!")
print(f"Output directory: {output_dir}")
print(f"Base filename: {base_filename}")
print(f"Total samples collected: {metadata['total_samples']}")
print(f"\nFiles created:")
print(f"  - {pickle_file}")
