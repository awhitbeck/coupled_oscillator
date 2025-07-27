import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional, List, Tuple, Dict, Any

class CoupledOscillators(gym.Env):
    """
    Coupled oscillators simulation as a Gymnasium environment.
    
    Simulates N oscillators connected by springs where the user applies
    force to oscillator 0, and all oscillators respond dynamically.
    Optionally, the last oscillator can move at a fixed frequency.
    """
    
    def __init__(self, N: int = 3, k=None, m=None, 
                 damping: float = 0.02, dt: float = 0.01, max_force: float = 5.0,
                 max_episode_steps: int = 1000, target_frequency: float = 1.0,
                 fixed_end_oscillator: bool = False, end_frequency: float = 1.0,
                 end_amplitude: float = 1.0, end_phase: float = 0.0):
        """
        Initialize the coupled oscillators system as a Gymnasium environment.
        
        Parameters:
        -----------
        N : int
            Number of oscillators
        k : float, np.ndarray, or None
            Coupling constants. Can be:
            - float: uniform coupling between adjacent oscillators only
            - np.ndarray (N, N): full coupling matrix between all oscillators
            - None: defaults to uniform coupling of 1.0 between adjacent oscillators
        m : float, np.ndarray, or None
            Masses. Can be:
            - float: uniform mass for all oscillators
            - np.ndarray (N,): individual masses for each oscillator
            - None: defaults to uniform mass of 1.0
        damping : float
            Damping coefficient (applied uniformly to all oscillators)
        dt : float
            Time step for numerical integration
        max_force : float
            Maximum absolute drive force applied to oscillator 0
        max_episode_steps : int
            Maximum number of steps per episode
        target_frequency : float
            Target frequency for reward calculation
        fixed_end_oscillator : bool
            If True, the last oscillator moves at a fixed frequency (infinite mass)
        end_frequency : float
            Frequency of the fixed end oscillator (Hz)
        end_amplitude : float
            Amplitude of the fixed end oscillator motion
        end_phase : float
            Phase offset of the fixed end oscillator motion (radians)
        """
        super().__init__()
        
        self.N = N
        self.damping = damping
        self.dt = dt
        self.max_force = max_force
        self.max_episode_steps = max_episode_steps
        self.target_frequency = target_frequency
        
        # Fixed end oscillator parameters
        self.fixed_end_oscillator = fixed_end_oscillator
        self.end_frequency = end_frequency
        self.end_amplitude = end_amplitude
        self.end_phase = end_phase
        
        # Setup masses
        if m is None:
            self.masses = np.ones(N, dtype=np.float32)
        elif isinstance(m, (int, float)):
            self.masses = np.full(N, float(m), dtype=np.float32)
        else:
            self.masses = np.array(m, dtype=np.float32)
            if len(self.masses) != N:
                raise ValueError(f"Mass array length {len(self.masses)} must equal N={N}")
        
        # Setup coupling matrix
        if k is None:
            # Default: adjacent coupling with k=1.0
            self.coupling_matrix = np.zeros((N, N), dtype=np.float32)
            for i in range(N-1):
                self.coupling_matrix[i, i+1] = 1.0
                self.coupling_matrix[i+1, i] = 1.0
        elif isinstance(k, (int, float)):
            # Uniform adjacent coupling
            self.coupling_matrix = np.zeros((N, N), dtype=np.float32)
            for i in range(N-1):
                self.coupling_matrix[i, i+1] = float(k)
                self.coupling_matrix[i+1, i] = float(k)
        else:
            # Full coupling matrix
            self.coupling_matrix = np.array(k, dtype=np.float32)
            if self.coupling_matrix.shape != (N, N):
                raise ValueError(f"Coupling matrix shape {self.coupling_matrix.shape} must be ({N}, {N})")
            # Ensure symmetry
            if not np.allclose(self.coupling_matrix, self.coupling_matrix.T):
                print("Warning: Coupling matrix is not symmetric. Making it symmetric.")
                self.coupling_matrix = (self.coupling_matrix + self.coupling_matrix.T) / 2
        
        # Gymnasium spaces
        # Action space: continuous drive force applied to oscillator 0
        self.action_space = spaces.Box(
            low=-max_force, high=max_force, shape=(1,), dtype=np.float32
        )
        
        # Observation space: position and velocity of the observation oscillator
        # If end oscillator is fixed, observe the second-to-last oscillator
        self.obs_oscillator_idx = N - 2 if fixed_end_oscillator and N > 1 else N - 1
        obs_dim = 2  # oscillator position + velocity
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state arrays
        self.positions = None
        self.velocities = None
        self.time = None
        self.step_count = None
        
        # Data storage for analysis
        self.time_history = []
        self.position_history = []
        self.drive_history = []
        
        # Reset to initialize
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the simulation to initial conditions (Gymnasium interface)."""
        super().reset(seed=seed)
        
        self.positions = np.zeros(self.N, dtype=np.float32)
        self.velocities = np.zeros(self.N, dtype=np.float32)
        self.time = 0.0
        self.step_count = 0
        self.time_history = []
        self.position_history = []
        self.drive_history = []
        
        # Add small random perturbations for varied initial conditions
        if seed is not None:
            np.random.seed(seed)
        self.positions += np.random.normal(0, 0.01, self.N).astype(np.float32)
        self.velocities += np.random.normal(0, 0.01, self.N).astype(np.float32)
        
        # Initialize fixed end oscillator position and velocity if applicable
        if self.fixed_end_oscillator:
            self.positions[-1] = self._get_fixed_end_position(self.time)
            self.velocities[-1] = self._get_fixed_end_velocity(self.time)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for Gymnasium interface."""
        # Return position and velocity of the observation oscillator
        obs = np.array([
            self.positions[self.obs_oscillator_idx],  # Observation oscillator position
            self.velocities[self.obs_oscillator_idx]  # Observation oscillator velocity
        ], dtype=np.float32)
        return obs
    
    def _get_fixed_end_position(self, time: float) -> float:
        """Calculate the position of the fixed end oscillator at given time."""
        if not self.fixed_end_oscillator:
            return 0.0
        return self.end_amplitude * np.sin(2 * np.pi * self.end_frequency * time + self.end_phase)
    
    def _get_fixed_end_velocity(self, time: float) -> float:
        """Calculate the velocity of the fixed end oscillator at given time."""
        if not self.fixed_end_oscillator:
            return 0.0
        omega = 2 * np.pi * self.end_frequency
        return self.end_amplitude * omega * np.cos(omega * time + self.end_phase)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary for Gymnasium interface."""
        # Calculate kinetic energy using individual masses
        kinetic_energy = np.sum(0.5 * self.masses * self.velocities**2)
        
        # Calculate potential energy from coupling springs
        potential_energy = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.coupling_matrix[i, j] != 0:
                    dx = self.positions[i] - self.positions[j]
                    potential_energy += 0.5 * self.coupling_matrix[i, j] * dx**2
        
        return {
            'time': self.time,
            'step_count': self.step_count,
            'last_oscillator_position': self.positions[-1] if len(self.positions) > 0 else 0.0,
            'first_oscillator_position': self.positions[0] if len(self.positions) > 0 else 0.0,
            'drive_force': self.drive_history[-1] if self.drive_history else 0.0,
            'kinetic_energy': float(kinetic_energy),
            'potential_energy': float(potential_energy),
            'total_energy': float(kinetic_energy + potential_energy)
        }
    
    def _calculate_reward(self, drive_force: float) -> float:
        """Calculate reward based on oscillator behavior."""
        # Reward based on how well the last oscillator follows a target pattern
        # For now, reward smooth oscillations at target frequency
        last_pos = self.positions[-1]
        
        # Penalty for extreme positions
        position_penalty = -0.1 * (last_pos**2)
        
        # Reward for maintaining reasonable oscillation amplitude
        target_amplitude = 0.5
        amplitude_reward = -0.05 * abs(abs(last_pos) - target_amplitude)
        
        # Penalty for excessive force (energy efficiency)
        force_penalty = -0.001 * (drive_force**2)
        
        # Small step reward to encourage longer episodes
        step_reward = 0.001
        
        return position_penalty + amplitude_reward + force_penalty + step_reward
    
    def calculate_forces(self, drive_force: float = 0.0) -> np.ndarray:
        """
        Calculate forces on each oscillator using the coupling matrix.
        
        Parameters:
        -----------
        drive_force : float
            External force applied to oscillator 0
            
        Returns:
        --------
        forces : np.ndarray
            Array of forces acting on each oscillator
        """
        forces = np.zeros(self.N, dtype=np.float32)
        
        # Get fixed end oscillator position if applicable
        if self.fixed_end_oscillator:
            end_pos = self._get_fixed_end_position(self.time)
        
        # Spring forces from coupling matrix
        for i in range(self.N):
            # Skip force calculation for fixed end oscillator
            if self.fixed_end_oscillator and i == self.N - 1:
                continue
                
            for j in range(self.N):
                if i != j and self.coupling_matrix[i, j] != 0:
                    if self.fixed_end_oscillator and j == self.N - 1:
                        # Force from fixed end oscillator
                        forces[i] += self.coupling_matrix[i, j] * (end_pos - self.positions[i])
                    else:
                        # Force from regular oscillator
                        forces[i] += self.coupling_matrix[i, j] * (self.positions[j] - self.positions[i])
            
            # Damping force proportional to velocity
            forces[i] -= self.damping * self.velocities[i]
        
        # Add external drive force to oscillator 0
        forces[0] += drive_force
        
        return forces
    
    def step(self, action):
        """
        Advance the simulation by one time step (Gymnasium interface).
        
        Parameters:
        -----------
        action : np.ndarray or float
            Drive force applied to oscillator 0
        """
        # Handle action input
        if isinstance(action, (np.ndarray, list)):
            drive_force = float(action[0])
        else:
            drive_force = float(action)
        
        # Clip action to valid range
        drive_force = np.clip(drive_force, -self.max_force, self.max_force)
        
        # Calculate forces on all oscillators including external drive force
        forces = self.calculate_forces(drive_force)
        
        # Update velocities and positions for all oscillators using individual masses
        for i in range(self.N):
            if self.fixed_end_oscillator and i == self.N - 1:
                # Fixed end oscillator: set position and velocity from fixed motion
                self.positions[i] = self._get_fixed_end_position(self.time + self.dt)
                self.velocities[i] = self._get_fixed_end_velocity(self.time + self.dt)
            else:
                # Regular oscillator: update using forces
                acceleration = forces[i] / self.masses[i]
                self.velocities[i] += acceleration * self.dt
                self.positions[i] += self.velocities[i] * self.dt
        
        # Update time and step count
        self.time += self.dt
        self.step_count += 1
        
        # Store data for analysis
        self.time_history.append(self.time)
        self.position_history.append(self.positions.copy())
        self.drive_history.append(drive_force)  # Now storing force instead of position
        
        # Calculate reward
        reward = self._calculate_reward(drive_force)
        
        # Check if episode is done
        terminated = False  # This environment doesn't have a natural termination
        truncated = self.step_count >= self.max_episode_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    
    
    def get_last_oscillator_response(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the response of the last oscillator."""
        if not self.time_history:
            return np.array([]), np.array([])
        
        times = np.array(self.time_history)
        last_positions = np.array(self.position_history)[:, -1]
        return times, last_positions
    
    def plot_results(self, show_all: bool = False):
        """
        Plot simulation results.
        
        Parameters:
        -----------
        show_all : bool
            If True, plot all oscillators. If False, plot only drive and last.
        """
        if not self.time_history:
            print("No data to plot. Run simulation first.")
            return
        
        times = np.array(self.time_history)
        positions = np.array(self.position_history)
        drives = np.array(self.drive_history)
        
        plt.figure(figsize=(12, 8))
        
        # Plot drive signal
        plt.subplot(2, 1, 1)
        plt.plot(times, drives, 'r-', alpha=0.1,linewidth=2, label='Drive Signal')
        plt.plot(times, positions[:, 0], 'r--', alpha=0.7, label='Oscillator 0 (Driven)')
        if show_all:
            for i in range(1, self.N-1):
                plt.plot(times, positions[:, i], alpha=0.5, label=f'Oscillator {i}')
        plt.plot(times, positions[:, -1], 'b-', linewidth=2, label=f'Oscillator {self.N-1} (Last)')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Coupled Oscillators Response')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot phase relationship
        plt.subplot(2, 1, 2)
        plt.plot(drives, positions[:, -1], 'g-', alpha=0.7)
        plt.xlabel('Drive Position')
        plt.ylabel('Last Oscillator Position')
        plt.title('Phase Plot: Drive vs Last Oscillator')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def render(self, mode='human'):
        """Render the environment (Gymnasium interface)."""
        if mode == 'human':
            # Simple text-based rendering
            if self.positions is not None:
                print(f"Time: {self.time:.3f}, Step: {self.step_count}, "
                      f"Drive Force: {self.drive_history[-1] if self.drive_history else 0:.3f}, "
                      f"Last Osc: {self.positions[-1]:.3f}")
        elif mode == 'rgb_array':
            # Could implement matplotlib-based rendering here
            # For now, return None
            return None
    
    def close(self):
        """Close the environment (Gymnasium interface)."""
        # Clean up any resources if needed
        plt.close('all')
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system configuration."""
        return {
            'N': self.N,
            'masses': self.masses.copy(),
            'coupling_matrix': self.coupling_matrix.copy(),
            'damping': self.damping,
            'dt': self.dt,
            'max_force': self.max_force,
            'fixed_end_oscillator': self.fixed_end_oscillator,
            'end_frequency': self.end_frequency if self.fixed_end_oscillator else None,
            'end_amplitude': self.end_amplitude if self.fixed_end_oscillator else None,
            'end_phase': self.end_phase if self.fixed_end_oscillator else None,
            'obs_oscillator_idx': self.obs_oscillator_idx
        }
    
    def set_coupling(self, i: int, j: int, k_value: float):
        """Set coupling between oscillators i and j."""
        if 0 <= i < self.N and 0 <= j < self.N:
            self.coupling_matrix[i, j] = k_value
            self.coupling_matrix[j, i] = k_value  # Maintain symmetry
        else:
            raise ValueError(f"Oscillator indices must be between 0 and {self.N-1}")
    
    def set_mass(self, i: int, mass: float):
        """Set mass of oscillator i."""
        if 0 <= i < self.N:
            self.masses[i] = mass
        else:
            raise ValueError(f"Oscillator index must be between 0 and {self.N-1}")
    
    def get_normal_modes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate normal modes of the system (linearized analysis).
        Returns eigenfrequencies and eigenvectors.
        """
        # Create the system matrix for small oscillations
        # M^-1 * K where M is mass matrix and K is coupling matrix
        M_inv = np.diag(1.0 / self.masses)
        
        # For small oscillations, the coupling matrix acts as a stiffness matrix
        # But we need to convert it to the proper form for eigenvalue analysis
        K = np.zeros_like(self.coupling_matrix)
        for i in range(self.N):
            K[i, i] = np.sum(self.coupling_matrix[i, :])  # Diagonal: sum of couplings
            for j in range(self.N):
                if i != j:
                    K[i, j] = -self.coupling_matrix[i, j]  # Off-diagonal: negative coupling
        
        # Solve eigenvalue problem: omega^2 * v = M^-1 * K * v
        eigenvalues, eigenvectors = np.linalg.eigh(M_inv @ K)
        
        # Convert to frequencies (omega = sqrt(eigenvalue))
        frequencies = np.sqrt(np.maximum(eigenvalues, 0))  # Avoid sqrt of negative
        
        return frequencies, eigenvectors


if __name__ == "__main__":
    """
    For comprehensive examples and validation tests, run:
    python validation.py
    
    For PID controller usage, import from pid_controller:
    from pid_controller import PIDController
    """
    print("CoupledOscillators simulation system")
    print("For examples and validation tests, run: python validation.py")
    print("For PID controller, import: from pid_controller import PIDController")
