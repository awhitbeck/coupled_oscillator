
# Coupled Oscillator Simulation

## 🚀 Quick Start Guide

### Prerequisites
```bash
pip install numpy matplotlib gymnasium
```

### 1. Run Basic Examples
```bash
python test.py
```

### 2. Interactive Python Usage
```python
from simulator import CoupledOscillators
from pid_controller import PIDController

# Create a simple 3-oscillator system
sim = CoupledOscillators(N=3, k=1.0, damping=0.02)
obs, info = sim.reset()

# Apply force and step simulation
force = 2.0  # Apply 2N force to first oscillator
obs, reward, term, trunc, info = sim.step(force)

# Plot results after running simulation
sim.plot_results()
```

### 3. Configuration-Based Setup
```python
# Use JSON configuration files for reproducible experiments
sim = CoupledOscillators(config_file="experiment_config.json")
pid = PIDController(config_file="experiment_config.json")

# Run PID-controlled simulation
obs, info = sim.reset()
for i in range(1000):
    control_force = pid.update(reference=0.0, actual=obs[0])
    obs, reward, term, trunc, info = sim.step(control_force)
```

### 4. Available Example Configurations
- **`example_config.json`**: Basic PID control setup
- **`experiment_config.json`**: Two-oscillator system with fixed end
- **`chirp_trajectory_config.json`**: Frequency sweep demonstration
- **`multi_sine_trajectory_config.json`**: Complex multi-frequency patterns
- **`custom_trajectory_config.json`**: Custom interpolated trajectories

## Using data as an environment

To support work that is either driven by simulation or driven by
prerecorded dataset, a few examples have been included to capture
datasets from the simulator and load a dataset to emulate the
simulated environment.  Emulation using data is particularly useful
for building surrogate models of complex systems. 

The archiver runs a configurable number of episodes and saves
data to a pickel file.  The format of the data is arranged
as a vetor of episodes, which is in turn a vector of data points
corresponding to an individual step.  

 Each data point is a 4-element tuple:
  - observation (array, shape: 2): [position, velocity] of observed oscillator
  - action (scalar): PID control force applied to first oscillator
  - next_observation (array, shape: 2): [position, velocity] after action
  - reward (scalar): Environment reward signal

Note, the reward may not be relevant for all use cases.  The next observation
is also a redundant with the observation of the previous step, but make
grabbing data for training slightly easier.

For different problems, the shape of the observations and actions may change,
but the overall structure should remain the same.


# Configuration System for Coupled Oscillator Simulation

This document describes how to use configuration files to set up experiments with the coupled oscillator simulation.

## Overview

The configuration system allows you to define all experiment parameters in JSON or YAML files, making it easy to:
- Reproduce experiments
- Share experimental setups
- Run parameter sweeps
- Maintain consistent configurations

## Configuration File Structure

Configuration files have three main sections:

### 1. Simulation Parameters (`simulation`)
Controls the physics simulation environment:

```json
{
  "simulation": {
    "N": 3,                          // Number of oscillators
    "k": 1.0,                        // Coupling constant (scalar or matrix)
    "m": 1.0,                        // Mass(es) (scalar or array)
    "damping": 0.02,                 // Damping coefficient
    "dt": 0.01,                      // Time step
    "max_force": 5.0,                // Maximum applied force
    "max_episode_steps": 1000,       // Maximum simulation steps
    "target_frequency": 1.0,         // Target frequency for rewards
    "fixed_end_oscillator": false,   // Enable fixed end oscillator
    "end_frequency": 1.0,            // Fixed end oscillator frequency
    "end_amplitude": 1.0,            // Fixed end oscillator amplitude
    "end_phase": 0.0,                // Fixed end oscillator phase
    "observation_noise_std": 0.0     // Gaussian noise std for observations
  }
}
```

### 2. PID Controller Parameters (`pid_controller`)
Controls the feedback controller:

```json
{
  "pid_controller": {
    "kp": 1.0,                       // Proportional gain
    "ki": 0.0,                       // Integral gain
    "kd": 0.0,                       // Derivative gain
    "output_limits": [-10.0, 10.0],  // Force output limits [min, max]
    "integral_limits": [-1.0, 1.0]   // Integral term limits [min, max]
  }
}
```

### 3. Experiment Parameters (`experiment`)
Controls the experiment execution:

```json
{
  "experiment": {
    "name": "my_experiment",
    "description": "Description of the experiment",
    "duration_steps": 1000,          // Total simulation steps
    "update_factor": 1,              // PID update frequency (every N steps)
    "reference_trajectory": {        // Reference signal definition
      "type": "constant",            // See trajectory types below
      "value": 0.0
    },
    "data_collection": {
      "save_results": false,         // Save data to file
      "output_directory": "./results", // Output directory
      "plot_results": true           // Show plots
    }
  }
}
```

## Reference Trajectory Types

### Constant
```json
{
  "type": "constant",
  "value": 0.5
}
```

### Sinusoidal
```json
{
  "type": "sinusoidal",
  "amplitude": 1.0,
  "frequency": 1.0,
  "phase": 0.0,
  "offset": 0.0
}
```

### Step
```json
{
  "type": "step",
  "step_time": 2.0,
  "initial_value": 0.0,
  "final_value": 1.0
}
```

### Ramp
```json
{
  "type": "ramp",
  "start_time": 1.0,
  "end_time": 3.0,
  "initial_value": 0.0,
  "final_value": 1.0
}
```

### Custom
```json
{
  "type": "custom",
  "values": [0.0, 0.5, 1.0, 0.5, 0.0]
}
```

## Advanced Configuration Examples

### Complex Coupling Matrix
```json
{
  "simulation": {
    "N": 4,
    "k": [
      [0.0, 1.0, 0.5, 0.1],
      [1.0, 0.0, 1.5, 0.3],
      [0.5, 1.5, 0.0, 3.0],
      [0.1, 0.3, 3.0, 0.0]
    ]
  }
}
```

### Individual Masses
```json
{
  "simulation": {
    "N": 3,
    "m": [0.5, 1.0, 2.0]
  }
}
```

### With Observation Noise
```json
{
  "simulation": {
    "observation_noise_std": 0.01
  }
}
```

## Usage Examples

### Command Line
```bash
# Run with default config
python3 test_config.py

# Run with specific config
python3 test_config.py my_experiment.json

# Validate config only
python3 test_config.py --validate-only my_experiment.json
```

### Python Code
```python
from simulator import CoupledOscillators
from pid_controller import PIDController

# Create components from config file
oscillators = CoupledOscillators(config_file="my_config.json")
pid = PIDController(config_file="my_config.json")

# Or load config separately
from config import ExperimentConfig
config = ExperimentConfig("my_config.json")
sim_params = config.get_simulation_config()
oscillators = CoupledOscillators(**sim_params)
```

## File Formats

Both JSON and YAML formats are supported:

- `.json` files: Standard JSON format
- `.yaml` or `.yml` files: YAML format (requires PyYAML: `pip install pyyaml`)

## Example Configuration Files

- `example_config.json`: Basic example with PID control
- `advanced_config.json`: Complex system with coupling matrix and noise
- `experiment_config.yaml`: YAML format example

## Error Handling

The configuration system validates parameters and provides clear error messages for:
- Missing required parameters
- Invalid parameter values
- Malformed configuration files
- Type mismatches

## Tips

1. Start with example configurations and modify as needed
2. Use `--validate-only` to check configurations before running
3. Save successful configurations for reproducibility
4. Use descriptive experiment names and descriptions
5. Set appropriate output directories for different experiments


# Complex Trajectory System for Coupled Oscillator Simulation

This document describes the advanced trajectory system that allows the last oscillator to follow complex, time-varying motion patterns instead of simple sinusoidal motion.

## Overview

The trajectory system extends the coupled oscillator simulation to support sophisticated end oscillator behaviors including:
- Multi-frequency sinusoidal patterns
- Frequency sweeps (chirps)  
- Polynomial trajectories
- Custom interpolated patterns
- Step functions with smooth transitions
- And more...

## Available Trajectory Types

### 1. Constant Trajectory
Maintains a fixed position.

```json
{
  "type": "constant",
  "value": 0.5
}
```

### 2. Sinusoidal Trajectory
Classic sinusoidal motion: `A*sin(2πft + φ) + offset`

```json
{
  "type": "sinusoidal",
  "amplitude": 1.0,
  "frequency": 0.5,
  "phase": 0.0,
  "offset": 0.0
}
```

### 3. Multi-Sine Trajectory
Sum of multiple sinusoidal components for complex periodic patterns.

```json
{
  "type": "multi_sine",
  "components": [
    {
      "amplitude": 0.8,
      "frequency": 0.5,
      "phase": 0.0
    },
    {
      "amplitude": 0.3,
      "frequency": 1.5,
      "phase": 1.57
    },
    {
      "amplitude": 0.15,
      "frequency": 3.0,
      "phase": 0.0
    }
  ],
  "offset": 0.0
}
```

### 4. Chirp Trajectory (Frequency Sweep)
Linear or logarithmic frequency sweep over time.

```json
{
  "type": "chirp",
  "amplitude": 1.0,
  "start_frequency": 0.5,
  "end_frequency": 3.0,
  "duration": 20.0,
  "method": "linear",
  "offset": 0.0
}
```

**Methods:**
- `"linear"`: Linear frequency sweep
- `"logarithmic"`: Exponential frequency sweep

### 5. Step Trajectory
Step function with optional smooth transition.

```json
{
  "type": "step",
  "step_time": 10.0,
  "initial_value": 0.0,
  "final_value": 1.0,
  "transition_time": 2.0
}
```

### 6. Ramp Trajectory
Linear ramp between two values over a specified time interval.

```json
{
  "type": "ramp",
  "start_time": 2.0,
  "end_time": 8.0,
  "initial_value": 0.0,
  "final_value": 1.5
}
```

### 7. Polynomial Trajectory
Polynomial function: `∑ aᵢ(t/T)ⁱ`

```json
{
  "type": "polynomial",
  "coefficients": [0.0, 0.2, -0.05, 0.002],
  "time_scale": 10.0
}
```

### 8. Custom Trajectory
Interpolated trajectory from data points.

```json
{
  "type": "custom",
  "time_points": [0.0, 2.0, 5.0, 8.0, 12.0, 15.0],
  "position_points": [0.0, 0.5, 1.2, 0.8, -0.3, 0.0],
  "extrapolation": "constant"
}
```

**Extrapolation options:**
- `"constant"`: Use boundary values
- `"linear"`: Linear extrapolation
- `"zero"`: Return zero outside range

## Usage Examples

### In Configuration Files

Add the `end_trajectory` parameter to the simulation configuration:

```json
{
  "simulation": {
    "N": 3,
    "fixed_end_oscillator": true,
    "end_trajectory": {
      "type": "chirp",
      "amplitude": 1.0,
      "start_frequency": 0.5,
      "end_frequency": 3.0,
      "duration": 20.0,
      "method": "linear"
    }
  }
}
```

### In Python Code

```python
from simulator import CoupledOscillators

# Using configuration file
oscillators = CoupledOscillators(config_file="chirp_config.json")

# Direct specification
trajectory_config = {
    'type': 'multi_sine',
    'components': [
        {'amplitude': 0.8, 'frequency': 0.5},
        {'amplitude': 0.3, 'frequency': 2.0}
    ]
}

oscillators = CoupledOscillators(
    N=3,
    fixed_end_oscillator=True,
    end_trajectory=trajectory_config
)
```

### Using Trajectory Generator Directly

```python
from trajectory import create_trajectory
import numpy as np

# Create trajectory generator
config = {
    'type': 'chirp',
    'amplitude': 1.0,
    'start_frequency': 0.1,
    'end_frequency': 2.0,
    'duration': 10.0
}

traj_gen = create_trajectory(config)

# Generate trajectory data
time = np.linspace(0, 10, 1000)
positions = [traj_gen.get_position(t) for t in time]
velocities = [traj_gen.get_velocity(t) for t in time]
```

## Advanced Features

### Automatic Velocity Calculation
All trajectory generators automatically compute velocities using analytical derivatives where possible, or numerical differentiation for complex patterns.

### Configuration Validation
The system validates trajectory configurations to ensure:
- Required parameters are present
- Parameter values are physically reasonable
- Type-specific constraints are met

### Backward Compatibility
The system is fully backward compatible. Existing configurations using simple sinusoidal motion continue to work unchanged.

## Example Configuration Files

### Chirp Trajectory Example
File: `chirp_trajectory_config.json`
- Demonstrates frequency sweep from 0.5 Hz to 3.0 Hz
- Linear chirp over 20 seconds
- 3-oscillator system with PID control

### Multi-Sine Trajectory Example  
File: `multi_sine_trajectory_config.json`
- Complex periodic motion with 3 frequency components
- 4-oscillator system with coupling matrix
- Step reference trajectory for PID

### Custom Trajectory Example
File: `custom_trajectory_config.json`
- Arbitrary trajectory from interpolated points
- Demonstrates custom motion patterns
- Ramp reference trajectory

### Polynomial Trajectory Example
File: `polynomial_trajectory_config.json`
- 4th-order polynomial trajectory
- Time-scaled coordinates
- Multi-sine reference trajectory

## Testing and Visualization

### Test Script
Use `test_trajectories.py` to explore trajectory capabilities:

```bash
# Demonstrate all trajectory types
python3 test_trajectories.py --demo-types

# Compare trajectory effects on system response
python3 test_trajectories.py --compare

# Test specific configuration
python3 test_trajectories.py --config chirp_trajectory_config.json

# Run all demonstrations
python3 test_trajectories.py --all
```

### Validation
Validate trajectory configurations:

```bash
python3 test_config.py --validate-only chirp_trajectory_config.json
```

## Physical Considerations

### Realistic Motion
All trajectory types generate physically reasonable motion with:
- Continuous position and velocity
- Smooth transitions (where applicable)
- Bounded accelerations

### System Response
Different trajectory types produce distinct system responses:
- **Chirp**: Reveals frequency response characteristics
- **Multi-sine**: Tests superposition and nonlinear effects  
- **Step**: Shows transient response
- **Polynomial**: Demonstrates response to complex acceleration profiles

### Parameter Selection
Guidelines for parameter selection:
- **Frequencies**: Should be comparable to system natural frequencies for resonance effects
- **Amplitudes**: Large amplitudes may cause nonlinear behavior
- **Duration**: Should be long enough to observe system settling
- **Transition times**: Affect smoothness vs. sharpness of response

