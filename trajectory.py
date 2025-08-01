#!/usr/bin/env python3
"""
Trajectory generation for coupled oscillator simulation.

Provides flexible trajectory generation capabilities for driving oscillators
with complex time-varying signals.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod


class TrajectoryGenerator(ABC):
    """
    Abstract base class for trajectory generators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trajectory generator.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary for the trajectory
        """
        self.config = config.copy()
    
    @abstractmethod
    def get_position(self, time: float) -> float:
        """
        Get position at given time.
        
        Parameters:
        -----------
        time : float
            Current time
            
        Returns:
        --------
        float
            Position value
        """
        pass
    
    @abstractmethod
    def get_velocity(self, time: float) -> float:
        """
        Get velocity at given time.
        
        Parameters:
        -----------
        time : float
            Current time
            
        Returns:
        --------
        float
            Velocity value
        """
        pass
    
    def get_acceleration(self, time: float, dt: float = 1e-6) -> float:
        """
        Get acceleration at given time using numerical differentiation.
        
        Parameters:
        -----------
        time : float
            Current time
        dt : float
            Small time step for numerical differentiation
            
        Returns:
        --------
        float
            Acceleration value
        """
        v1 = self.get_velocity(time - dt/2)
        v2 = self.get_velocity(time + dt/2)
        return (v2 - v1) / dt


class ConstantTrajectory(TrajectoryGenerator):
    """Constant position trajectory."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.value = config.get('value', 0.0)
    
    def get_position(self, time: float) -> float:
        return self.value
    
    def get_velocity(self, time: float) -> float:
        return 0.0


class SinusoidalTrajectory(TrajectoryGenerator):
    """Sinusoidal trajectory: A*sin(2*pi*f*t + phi) + offset."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.amplitude = config.get('amplitude', 1.0)
        self.frequency = config.get('frequency', 1.0)
        self.phase = config.get('phase', 0.0)
        self.offset = config.get('offset', 0.0)
        self.omega = 2 * np.pi * self.frequency
    
    def get_position(self, time: float) -> float:
        return self.amplitude * np.sin(self.omega * time + self.phase) + self.offset
    
    def get_velocity(self, time: float) -> float:
        return self.amplitude * self.omega * np.cos(self.omega * time + self.phase)


class MultiSineTrajectory(TrajectoryGenerator):
    """Multiple sinusoidal components: sum of A_i*sin(2*pi*f_i*t + phi_i)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.components = config.get('components', [])
        self.offset = config.get('offset', 0.0)
        
        # Validate components
        for i, comp in enumerate(self.components):
            if not all(key in comp for key in ['amplitude', 'frequency']):
                raise ValueError(f"Component {i} missing required keys (amplitude, frequency)")
    
    def get_position(self, time: float) -> float:
        position = self.offset
        for comp in self.components:
            amplitude = comp['amplitude']
            frequency = comp['frequency']
            phase = comp.get('phase', 0.0)
            omega = 2 * np.pi * frequency
            position += amplitude * np.sin(omega * time + phase)
        return position
    
    def get_velocity(self, time: float) -> float:
        velocity = 0.0
        for comp in self.components:
            amplitude = comp['amplitude']
            frequency = comp['frequency']
            phase = comp.get('phase', 0.0)
            omega = 2 * np.pi * frequency
            velocity += amplitude * omega * np.cos(omega * time + phase)
        return velocity


class StepTrajectory(TrajectoryGenerator):
    """Step trajectory with transition time."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.step_time = config.get('step_time', 1.0)
        self.initial_value = config.get('initial_value', 0.0)
        self.final_value = config.get('final_value', 1.0)
        self.transition_time = config.get('transition_time', 0.0)  # Smooth transition
    
    def get_position(self, time: float) -> float:
        if time < self.step_time:
            return self.initial_value
        elif self.transition_time > 0 and time < self.step_time + self.transition_time:
            # Smooth transition using cosine
            progress = (time - self.step_time) / self.transition_time
            smoothed = 0.5 * (1 - np.cos(np.pi * progress))
            return self.initial_value + smoothed * (self.final_value - self.initial_value)
        else:
            return self.final_value
    
    def get_velocity(self, time: float) -> float:
        if self.transition_time > 0 and self.step_time <= time < self.step_time + self.transition_time:
            progress = (time - self.step_time) / self.transition_time
            # Derivative of smooth transition
            velocity_factor = (np.pi / (2 * self.transition_time)) * np.sin(np.pi * progress)
            return velocity_factor * (self.final_value - self.initial_value)
        return 0.0


class RampTrajectory(TrajectoryGenerator):
    """Linear ramp trajectory."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.start_time = config.get('start_time', 0.0)
        self.end_time = config.get('end_time', 2.0)
        self.initial_value = config.get('initial_value', 0.0)
        self.final_value = config.get('final_value', 1.0)
        
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        
        self.duration = self.end_time - self.start_time
        self.slope = (self.final_value - self.initial_value) / self.duration
    
    def get_position(self, time: float) -> float:
        if time < self.start_time:
            return self.initial_value
        elif time > self.end_time:
            return self.final_value
        else:
            return self.initial_value + self.slope * (time - self.start_time)
    
    def get_velocity(self, time: float) -> float:
        if self.start_time <= time <= self.end_time:
            return self.slope
        return 0.0


class ChirpTrajectory(TrajectoryGenerator):
    """Frequency sweep (chirp) trajectory."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.amplitude = config.get('amplitude', 1.0)
        self.f0 = config.get('start_frequency', 1.0)
        self.f1 = config.get('end_frequency', 5.0)
        self.duration = config.get('duration', 10.0)
        self.method = config.get('method', 'linear')  # 'linear' or 'logarithmic'
        self.offset = config.get('offset', 0.0)
    
    def _get_instantaneous_frequency(self, time: float) -> float:
        """Get instantaneous frequency at given time."""
        if time < 0 or time > self.duration:
            return self.f0 if time < 0 else self.f1
        
        if self.method == 'linear':
            return self.f0 + (self.f1 - self.f0) * time / self.duration
        elif self.method == 'logarithmic':
            if self.f0 <= 0 or self.f1 <= 0:
                raise ValueError("Frequencies must be positive for logarithmic chirp")
            ratio = self.f1 / self.f0
            return self.f0 * (ratio ** (time / self.duration))
        else:
            raise ValueError(f"Unknown chirp method: {self.method}")
    
    def get_position(self, time: float) -> float:
        if time < 0 or time > self.duration:
            return self.offset
        
        if self.method == 'linear':
            # For linear chirp: phi(t) = 2*pi * (f0*t + (f1-f0)*t^2/(2*T))
            phase = 2 * np.pi * (self.f0 * time + (self.f1 - self.f0) * time**2 / (2 * self.duration))
        elif self.method == 'logarithmic':
            # For logarithmic chirp: phi(t) = 2*pi * f0 * T / ln(f1/f0) * (ratio^(t/T) - 1)
            ratio = self.f1 / self.f0
            log_ratio = np.log(ratio)
            phase = 2 * np.pi * self.f0 * self.duration / log_ratio * (ratio**(time / self.duration) - 1)
        
        return self.amplitude * np.sin(phase) + self.offset
    
    def get_velocity(self, time: float) -> float:
        if time < 0 or time > self.duration:
            return 0.0
        
        freq = self._get_instantaneous_frequency(time)
        omega = 2 * np.pi * freq
        
        # Phase calculation (same as in get_position)
        if self.method == 'linear':
            phase = 2 * np.pi * (self.f0 * time + (self.f1 - self.f0) * time**2 / (2 * self.duration))
        elif self.method == 'logarithmic':
            ratio = self.f1 / self.f0
            log_ratio = np.log(ratio)
            phase = 2 * np.pi * self.f0 * self.duration / log_ratio * (ratio**(time / self.duration) - 1)
        
        return self.amplitude * omega * np.cos(phase)


class PolynomialTrajectory(TrajectoryGenerator):
    """Polynomial trajectory: sum of a_i * t^i."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.coefficients = np.array(config.get('coefficients', [0.0]))
        self.time_scale = config.get('time_scale', 1.0)  # Scale time axis
    
    def get_position(self, time: float) -> float:
        scaled_time = time / self.time_scale
        return np.polyval(self.coefficients[::-1], scaled_time)
    
    def get_velocity(self, time: float) -> float:
        if len(self.coefficients) <= 1:
            return 0.0
        
        # Derivative coefficients
        deriv_coeffs = np.array([i * self.coefficients[i] for i in range(1, len(self.coefficients))])
        scaled_time = time / self.time_scale
        return np.polyval(deriv_coeffs[::-1], scaled_time) / self.time_scale


class CustomTrajectory(TrajectoryGenerator):
    """Custom trajectory from interpolated data points."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.time_points = np.array(config.get('time_points', [0.0, 1.0]))
        self.position_points = np.array(config.get('position_points', [0.0, 0.0]))
        self.extrapolation = config.get('extrapolation', 'constant')  # 'constant', 'linear', 'zero'
        
        if len(self.time_points) != len(self.position_points):
            raise ValueError("time_points and position_points must have same length")
        
        # Sort by time
        sort_idx = np.argsort(self.time_points)
        self.time_points = self.time_points[sort_idx]
        self.position_points = self.position_points[sort_idx]
    
    def get_position(self, time: float) -> float:
        if len(self.time_points) == 1:
            return self.position_points[0]
        
        # Handle extrapolation
        if time < self.time_points[0]:
            if self.extrapolation == 'constant':
                return self.position_points[0]
            elif self.extrapolation == 'linear':
                if len(self.time_points) >= 2:
                    slope = (self.position_points[1] - self.position_points[0]) / (self.time_points[1] - self.time_points[0])
                    return self.position_points[0] + slope * (time - self.time_points[0])
                else:
                    return self.position_points[0]
            else:  # 'zero'
                return 0.0
        
        if time > self.time_points[-1]:
            if self.extrapolation == 'constant':
                return self.position_points[-1]
            elif self.extrapolation == 'linear':
                if len(self.time_points) >= 2:
                    slope = (self.position_points[-1] - self.position_points[-2]) / (self.time_points[-1] - self.time_points[-2])
                    return self.position_points[-1] + slope * (time - self.time_points[-1])
                else:
                    return self.position_points[-1]
            else:  # 'zero'
                return 0.0
        
        # Interpolate
        return np.interp(time, self.time_points, self.position_points)
    
    def get_velocity(self, time: float, dt: float = 1e-4) -> float:
        # Numerical differentiation
        return (self.get_position(time + dt/2) - self.get_position(time - dt/2)) / dt


def create_trajectory(trajectory_config: Dict[str, Any]) -> TrajectoryGenerator:
    """
    Factory function to create trajectory generators.
    
    Parameters:
    -----------
    trajectory_config : dict
        Configuration dictionary containing trajectory type and parameters
        
    Returns:
    --------
    TrajectoryGenerator
        Instantiated trajectory generator
    """
    trajectory_type = trajectory_config.get('type', 'constant')
    
    trajectory_classes = {
        'constant': ConstantTrajectory,
        'sinusoidal': SinusoidalTrajectory,
        'multi_sine': MultiSineTrajectory,
        'step': StepTrajectory,
        'ramp': RampTrajectory,
        'chirp': ChirpTrajectory,
        'polynomial': PolynomialTrajectory,
        'custom': CustomTrajectory,
    }
    
    if trajectory_type not in trajectory_classes:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}. "
                        f"Available types: {list(trajectory_classes.keys())}")
    
    return trajectory_classes[trajectory_type](trajectory_config)


if __name__ == "__main__":
    # Example usage and testing
    import matplotlib.pyplot as plt
    
    # Test different trajectory types
    time = np.linspace(0, 10, 1000)
    
    # Sinusoidal
    sin_config = {'type': 'sinusoidal', 'amplitude': 1.0, 'frequency': 0.5, 'phase': 0.0, 'offset': 0.0}
    sin_traj = create_trajectory(sin_config)
    
    # Multi-sine
    multi_config = {
        'type': 'multi_sine',
        'components': [
            {'amplitude': 1.0, 'frequency': 0.5},
            {'amplitude': 0.3, 'frequency': 2.0, 'phase': np.pi/4}
        ],
        'offset': 0.0
    }
    multi_traj = create_trajectory(multi_config)
    
    # Chirp
    chirp_config = {
        'type': 'chirp',
        'amplitude': 1.0,
        'start_frequency': 0.1,
        'end_frequency': 2.0,
        'duration': 10.0,
        'method': 'linear'
    }
    chirp_traj = create_trajectory(chirp_config)
    
    # Plot trajectories
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    positions = [sin_traj.get_position(t) for t in time]
    velocities = [sin_traj.get_velocity(t) for t in time]
    plt.plot(time, positions, 'b-', label='Position')
    plt.plot(time, velocities, 'r--', label='Velocity')
    plt.title('Sinusoidal Trajectory')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    positions = [multi_traj.get_position(t) for t in time]
    velocities = [multi_traj.get_velocity(t) for t in time]
    plt.plot(time, positions, 'b-', label='Position')
    plt.plot(time, velocities, 'r--', label='Velocity')
    plt.title('Multi-Sine Trajectory')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    positions = [chirp_traj.get_position(t) for t in time]
    velocities = [chirp_traj.get_velocity(t) for t in time]
    plt.plot(time, positions, 'b-', label='Position')
    plt.plot(time, velocities, 'r--', label='Velocity')
    plt.title('Chirp Trajectory')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Trajectory examples completed!")