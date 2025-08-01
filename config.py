#!/usr/bin/env python3
"""
Configuration management for coupled oscillator simulation.

Supports loading simulation and PID controller parameters from JSON or YAML files.
"""

import json
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigurationError(Exception):
    """Raised when there's an error in configuration file or parameters."""
    pass


class ExperimentConfig:
    """
    Configuration manager for coupled oscillator experiments.
    
    Handles loading and validation of configuration parameters for both
    the simulation environment and PID controller.
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Parameters:
        -----------
        config_file : str, Path, or None
            Path to configuration file (JSON or YAML). If None, uses defaults.
        """
        self.config_file = Path(config_file) if config_file else None
        self.config = self._load_default_config()
        
        if self.config_file:
            self.load_from_file(self.config_file)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration parameters."""
        return {
            "simulation": {
                "N": 3,
                "k": 1.0,
                "m": 1.0,
                "damping": 0.02,
                "dt": 0.01,
                "max_force": 5.0,
                "max_episode_steps": 1000,
                "target_frequency": 1.0,
                "fixed_end_oscillator": False,
                "end_frequency": 1.0,
                "end_amplitude": 1.0,
                "end_phase": 0.0,
                "observation_noise_std": 0.0,
                "end_trajectory": None
            },
            "pid_controller": {
                "kp": 1.0,
                "ki": 0.0,
                "kd": 0.0,
                "output_limits": None,
                "integral_limits": None
            },
            "experiment": {
                "name": "default_experiment",
                "description": "Default coupled oscillator experiment",
                "duration_steps": 1000,
                "update_factor": 1,
                "reference_trajectory": {
                    "type": "constant",
                    "value": 0.0
                },
                "data_collection": {
                    "save_results": False,
                    "output_directory": "./results",
                    "plot_results": True
                }
            }
        }
    
    def load_from_file(self, config_file: Union[str, Path]):
        """
        Load configuration from file.
        
        Parameters:
        -----------
        config_file : str or Path
            Path to configuration file (JSON or YAML)
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    if not YAML_AVAILABLE:
                        raise ConfigurationError("PyYAML is required to load YAML files. Install with: pip install pyyaml")
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported file format: {config_path.suffix}. Use .json, .yaml, or .yml")
            
            # Merge with defaults
            self._deep_update(self.config, file_config)
            self.config_file = config_path
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigurationError(f"Error parsing configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration file: {e}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """Get simulation configuration parameters."""
        sim_config = self.config["simulation"].copy()
        
        # Convert coupling matrix if provided as list
        if isinstance(sim_config.get("k"), list):
            sim_config["k"] = np.array(sim_config["k"], dtype=np.float32)
        
        # Convert masses if provided as list
        if isinstance(sim_config.get("m"), list):
            sim_config["m"] = np.array(sim_config["m"], dtype=np.float32)
        
        return sim_config
    
    def get_pid_config(self) -> Dict[str, Any]:
        """Get PID controller configuration parameters."""
        pid_config = self.config["pid_controller"].copy()
        
        # Convert limits to tuples if provided as lists
        if isinstance(pid_config.get("output_limits"), list):
            pid_config["output_limits"] = tuple(pid_config["output_limits"])
        
        if isinstance(pid_config.get("integral_limits"), list):
            pid_config["integral_limits"] = tuple(pid_config["integral_limits"])
        
        return pid_config
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration parameters."""
        return self.config["experiment"].copy()
    
    def get_reference_trajectory(self, time_steps: np.ndarray) -> np.ndarray:
        """
        Generate reference trajectory based on configuration.
        
        Parameters:
        -----------
        time_steps : np.ndarray
            Array of time values
            
        Returns:
        --------
        np.ndarray
            Reference trajectory values
        """
        ref_config = self.config["experiment"]["reference_trajectory"]
        trajectory_type = ref_config.get("type", "constant")
        
        if trajectory_type == "constant":
            value = ref_config.get("value", 0.0)
            return np.full_like(time_steps, value)
        
        elif trajectory_type == "sinusoidal":
            amplitude = ref_config.get("amplitude", 1.0)
            frequency = ref_config.get("frequency", 1.0)
            phase = ref_config.get("phase", 0.0)
            offset = ref_config.get("offset", 0.0)
            return amplitude * np.sin(2 * np.pi * frequency * time_steps + phase) + offset
        
        elif trajectory_type == "step":
            step_time = ref_config.get("step_time", 1.0)
            initial_value = ref_config.get("initial_value", 0.0)
            final_value = ref_config.get("final_value", 1.0)
            return np.where(time_steps >= step_time, final_value, initial_value)
        
        elif trajectory_type == "ramp":
            start_time = ref_config.get("start_time", 0.0)
            end_time = ref_config.get("end_time", 2.0)
            initial_value = ref_config.get("initial_value", 0.0)
            final_value = ref_config.get("final_value", 1.0)
            
            trajectory = np.full_like(time_steps, initial_value)
            mask = (time_steps >= start_time) & (time_steps <= end_time)
            ramp_progress = (time_steps[mask] - start_time) / (end_time - start_time)
            trajectory[mask] = initial_value + ramp_progress * (final_value - initial_value)
            trajectory[time_steps > end_time] = final_value
            return trajectory
        
        elif trajectory_type == "custom":
            # Support for custom array of values
            values = ref_config.get("values", [0.0])
            if len(values) == len(time_steps):
                return np.array(values)
            else:
                # Interpolate if lengths don't match
                return np.interp(time_steps, 
                               np.linspace(time_steps[0], time_steps[-1], len(values)), 
                               values)
        
        else:
            raise ConfigurationError(f"Unknown reference trajectory type: {trajectory_type}")
    
    def validate_config(self):
        """Validate configuration parameters."""
        sim_config = self.config["simulation"]
        pid_config = self.config["pid_controller"]
        exp_config = self.config["experiment"]
        
        # Validate simulation parameters
        if sim_config["N"] < 1:
            raise ConfigurationError("Number of oscillators (N) must be at least 1")
        
        if sim_config["dt"] <= 0:
            raise ConfigurationError("Time step (dt) must be positive")
        
        if sim_config["damping"] < 0:
            raise ConfigurationError("Damping must be non-negative")
        
        if sim_config["observation_noise_std"] < 0:
            raise ConfigurationError("Observation noise standard deviation must be non-negative")
        
        # Validate PID parameters
        output_limits = pid_config.get("output_limits")
        if output_limits and len(output_limits) == 2:
            if output_limits[0] >= output_limits[1]:
                raise ConfigurationError("Output limits: lower bound must be less than upper bound")
        
        integral_limits = pid_config.get("integral_limits")
        if integral_limits and len(integral_limits) == 2:
            if integral_limits[0] >= integral_limits[1]:
                raise ConfigurationError("Integral limits: lower bound must be less than upper bound")
        
        # Validate experiment parameters
        if exp_config["duration_steps"] < 1:
            raise ConfigurationError("Experiment duration must be at least 1 step")
        
        if exp_config["update_factor"] < 1:
            raise ConfigurationError("Update factor must be at least 1")
        
        # Validate trajectory configuration if present
        end_trajectory = sim_config.get("end_trajectory")
        if end_trajectory is not None:
            self._validate_trajectory_config(end_trajectory)
    
    def _validate_trajectory_config(self, trajectory_config: Dict[str, Any]):
        """Validate trajectory configuration parameters."""
        trajectory_type = trajectory_config.get('type')
        if not trajectory_type:
            raise ConfigurationError("Trajectory configuration must specify 'type'")
        
        valid_types = ['constant', 'sinusoidal', 'multi_sine', 'step', 'ramp', 'chirp', 'polynomial', 'custom']
        if trajectory_type not in valid_types:
            raise ConfigurationError(f"Invalid trajectory type '{trajectory_type}'. Valid types: {valid_types}")
        
        # Type-specific validation
        if trajectory_type == 'sinusoidal':
            if trajectory_config.get('frequency', 1.0) <= 0:
                raise ConfigurationError("Sinusoidal trajectory frequency must be positive")
        
        elif trajectory_type == 'multi_sine':
            components = trajectory_config.get('components', [])
            if not components:
                raise ConfigurationError("Multi-sine trajectory must have at least one component")
            for i, comp in enumerate(components):
                if not isinstance(comp, dict):
                    raise ConfigurationError(f"Multi-sine component {i} must be a dictionary")
                if 'amplitude' not in comp or 'frequency' not in comp:
                    raise ConfigurationError(f"Multi-sine component {i} must have 'amplitude' and 'frequency'")
                if comp['frequency'] <= 0:
                    raise ConfigurationError(f"Multi-sine component {i} frequency must be positive")
        
        elif trajectory_type == 'step':
            step_time = trajectory_config.get('step_time', 1.0)
            transition_time = trajectory_config.get('transition_time', 0.0)
            if step_time < 0:
                raise ConfigurationError("Step trajectory step_time must be non-negative")
            if transition_time < 0:
                raise ConfigurationError("Step trajectory transition_time must be non-negative")
        
        elif trajectory_type == 'ramp':
            start_time = trajectory_config.get('start_time', 0.0)
            end_time = trajectory_config.get('end_time', 2.0)
            if end_time <= start_time:
                raise ConfigurationError("Ramp trajectory end_time must be greater than start_time")
        
        elif trajectory_type == 'chirp':
            start_freq = trajectory_config.get('start_frequency', 1.0)
            end_freq = trajectory_config.get('end_frequency', 5.0)
            duration = trajectory_config.get('duration', 10.0)
            method = trajectory_config.get('method', 'linear')
            
            if start_freq <= 0 or end_freq <= 0:
                raise ConfigurationError("Chirp trajectory frequencies must be positive")
            if duration <= 0:
                raise ConfigurationError("Chirp trajectory duration must be positive")
            if method not in ['linear', 'logarithmic']:
                raise ConfigurationError("Chirp trajectory method must be 'linear' or 'logarithmic'")
        
        elif trajectory_type == 'polynomial':
            coefficients = trajectory_config.get('coefficients', [])
            if not coefficients:
                raise ConfigurationError("Polynomial trajectory must have at least one coefficient")
            time_scale = trajectory_config.get('time_scale', 1.0)
            if time_scale <= 0:
                raise ConfigurationError("Polynomial trajectory time_scale must be positive")
        
        elif trajectory_type == 'custom':
            time_points = trajectory_config.get('time_points', [])
            position_points = trajectory_config.get('position_points', [])
            if len(time_points) != len(position_points):
                raise ConfigurationError("Custom trajectory time_points and position_points must have same length")
            if len(time_points) < 2:
                raise ConfigurationError("Custom trajectory must have at least 2 points")
            extrapolation = trajectory_config.get('extrapolation', 'constant')
            if extrapolation not in ['constant', 'linear', 'zero']:
                raise ConfigurationError("Custom trajectory extrapolation must be 'constant', 'linear', or 'zero'")
    
    def save_to_file(self, filename: Union[str, Path], format: str = "yaml"):
        """
        Save current configuration to file.
        
        Parameters:
        -----------
        filename : str or Path
            Output filename
        format : str
            File format ("yaml" or "json")
        """
        filepath = Path(filename)
        
        if format.lower() == "yaml":
            if not YAML_AVAILABLE:
                raise ConfigurationError("PyYAML is required to save YAML files")
            with open(filepath, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            # Convert numpy arrays to lists for JSON serialization
            json_config = self._convert_numpy_to_list(self.config)
            with open(filepath, 'w') as f:
                json.dump(json_config, f, indent=2)
        else:
            raise ConfigurationError(f"Unsupported format: {format}. Use 'yaml' or 'json'")
    
    def _convert_numpy_to_list(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    def print_config(self):
        """Print current configuration in a readable format."""
        print("Current Configuration:")
        print("=" * 50)
        self._print_dict(self.config, indent=0)
    
    def _print_dict(self, d: Dict, indent: int = 0):
        """Recursively print dictionary with indentation."""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")


def load_config(config_file: Union[str, Path]) -> ExperimentConfig:
    """
    Convenience function to load configuration from file.
    
    Parameters:
    -----------
    config_file : str or Path
        Path to configuration file
        
    Returns:
    --------
    ExperimentConfig
        Loaded configuration object
    """
    return ExperimentConfig(config_file)


if __name__ == "__main__":
    # Example usage
    config = ExperimentConfig()
    config.print_config()
    print("\nSaving example configuration...")
    config.save_to_file("example_config.yaml", "yaml")
    config.save_to_file("example_config.json", "json")
    print("Example configuration files created!")