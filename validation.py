#!/usr/bin/env python3
"""
Validation and testing scripts for the coupled oscillator simulation.

This file contains comprehensive examples and tests demonstrating all features
of the CoupledOscillators simulation system and PID controller.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulator import CoupledOscillators
from pid_controller import PIDController


def example_1_force_sinusoidal():
    """Example 1: Force-based sinusoidal driving."""
    print("Example 1: Force-based Sinusoidal Drive")
    print("-" * 35)
    
    # Create a custom coupling matrix
    coupling_matrix = np.array([
        [0.0, 1.0, 0.001],  # Oscillator 0 couplings
        [1.0, 0.0, 3.5],    # Oscillator 1 couplings  
        [0.001, 3.5, 0.0]   # Oscillator 2 couplings
    ]) / 10
    
    masses = [1., 1., 10.]
    oscillators = CoupledOscillators(N=3, k=coupling_matrix, m=masses, damping=0.02, dt=0.01)
    
    oscillators.reset()
    for i in range(1000):  # 10 seconds at dt=0.01
        t = oscillators.time
        # Sinusoidal force
        force = 2.0 * np.sin(2 * np.pi * 10.0 * t)  # 2N amplitude, 10 Hz
        obs, reward, term, trunc, info = oscillators.step(force)
        
        if i % 200 == 0:  # Print every 2 seconds
            print(f"t={t:.2f}s, force={force:.3f}N, last_osc={obs[0]:.4f}")
    
    print(f"Simulation completed: {len(oscillators.time_history)} time steps")
    print(f"Final time: {oscillators.time:.2f} s")
    print(f"Last oscillator final position: {obs[0]:.4f}")
    print()
    
    oscillators.plot_results(show_all=True)
    return oscillators


def example_2_step_force():
    """Example 2: Step force driving."""
    print("Example 2: Step Force Drive")
    print("-" * 25)
    
    oscillators = CoupledOscillators(N=3, k=1.0, m=1.0, damping=0.02, dt=0.01)
    oscillators.reset()
    
    for i in range(800):  # 8 seconds at dt=0.01 
        t = oscillators.time
        # Step force: 5N between t=2 and t=4
        force = 5.0 if 2.0 <= t <= 4.0 else 0.0
        obs, reward, term, trunc, info = oscillators.step(force)
        
        if i % 100 == 0 or (1.9 <= t <= 4.1 and i % 50 == 0):  # More frequent during step
            print(f"t={t:.2f}s, force={force:.1f}N, last_osc={obs[0]:.4f}")
    
    print()
    return oscillators


def example_3_chirp_signal():
    """Example 3: Force-based chirp signal."""
    print("Example 3: Force-based Chirp Signal")
    print("-" * 32)
    
    oscillators = CoupledOscillators(N=3, k=1.0, m=1.0, damping=0.02, dt=0.01)
    oscillators.reset()
    
    # Manually control each time step with force
    for i in range(1000):  # 10 seconds at dt=0.01
        t = oscillators.time
        
        # Custom driving function: chirp force signal
        if t < 5.0:
            freq = 0.5 + 0.2 * t  # Increasing frequency
            force = 3.0 * np.sin(2 * np.pi * freq * t)  # Force instead of position
        else:
            force = 0.0  # No driving after 5 seconds
        
        obs, reward, term, trunc, info = oscillators.step(force)
        
        # Print progress every 100 steps
        if i % 100 == 0:
            print(f"t={t:.2f}s, force={force:.3f}N, last_osc={obs[0]:.3f}")
    
    print()
    print("Force-based simulation completed!")
    return oscillators


def example_4_analysis():
    """Example 4: Analysis functions."""
    print("Example 4: Analysis")
    print("-" * 15)
    
    oscillators = CoupledOscillators(N=3, k=1.0, m=1.0, damping=0.02, dt=0.01)
    oscillators.reset()
    
    # Generate some data first
    for i in range(500):
        t = oscillators.time
        force = 2.0 * np.sin(2 * np.pi * 1.0 * t)
        obs, reward, term, trunc, info = oscillators.step(force)
    
    # Calculate transfer function (amplitude ratio) - now force to position
    drive_force_amplitude = np.max(np.abs(oscillators.drive_history))
    last_amplitude = np.max(np.abs(np.array(oscillators.position_history)[:, -1]))
    transfer_ratio = last_amplitude / drive_force_amplitude if drive_force_amplitude > 0 else 0
    
    print(f"Drive force amplitude: {drive_force_amplitude:.4f}N")
    print(f"Last oscillator amplitude: {last_amplitude:.4f}")
    print(f"Transfer ratio (pos/force): {transfer_ratio:.4f} m/N")
    
    # Calculate phase delay (simple cross-correlation)
    drive_signal = np.array(oscillators.drive_history)
    last_signal = np.array(oscillators.position_history)[:, -1]
    
    if len(drive_signal) > 100:  # Only if we have enough data
        correlation = np.correlate(last_signal, drive_signal, mode='full')
        delay_samples = np.argmax(correlation) - len(drive_signal) + 1
        phase_delay = delay_samples * oscillators.dt
        print(f"Phase delay: {phase_delay:.4f} s")
    
    return oscillators


def example_5_gymnasium_random():
    """Example 5: Gymnasium environment usage with random actions."""
    print("\n" + "="*50)
    print("Example 5: Gymnasium Environment Usage")
    print("="*50)
    
    # Create environment
    env = CoupledOscillators(N=4, max_episode_steps=200)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a simple episode with random actions
    obs, info = env.reset(seed=42)
    total_reward = 0
    
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    for step in range(50):  # Short episode for demo
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"Step {step}: action={action[0]:.3f}, reward={reward:.4f}, "
                  f"last_osc={info['last_oscillator_position']:.3f}")
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode finished after {step+1} steps")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final observation: {obs[:5]}...")  # Show first 5 elements
    
    env.close()
    return env


def example_6_advanced_configuration():
    """Example 6: Advanced configuration with different masses and couplings."""
    print("\n" + "="*60)
    print("Example 6: Advanced Configuration - Different Masses & Couplings")
    print("="*60)
    
    # Create a system with different masses
    masses = [0.5, 1.0, 2.0, 1.5]  # Different masses for each oscillator
    
    # Create a custom coupling matrix (all-to-all with different strengths)
    coupling_matrix = np.array([
        [0.0, 2.0, 0.5, 0.1],  # Oscillator 0 couplings
        [2.0, 0.0, 1.5, 0.3],  # Oscillator 1 couplings  
        [0.5, 1.5, 0.0, 3.0],  # Oscillator 2 couplings
        [0.1, 0.3, 3.0, 0.0]   # Oscillator 3 couplings
    ]) / 10
    
    # Create advanced system
    advanced_env = CoupledOscillators(N=4, m=masses, k=coupling_matrix, max_force=15.0)
    
    print("System configuration:")
    sys_info = advanced_env.get_system_info()
    print(f"Masses: {sys_info['masses']}")
    print(f"Coupling matrix shape: {sys_info['coupling_matrix'].shape}")
    print("Coupling matrix:")
    for i, row in enumerate(sys_info['coupling_matrix']):
        print(f"  Osc {i}: {row}")
    
    # Calculate normal modes
    try:
        frequencies, modes = advanced_env.get_normal_modes()
        print(f"\nNormal mode frequencies: {frequencies}")
        print("Normal mode shapes (first 3 components):")
        for i, mode in enumerate(modes.T[:3]):
            print(f"  Mode {i}: {mode[:3]}")
    except Exception as e:
        print(f"Normal mode calculation failed: {e}")
    
    # Test the advanced system
    obs, info = advanced_env.reset(seed=123)
    print(f"\nInitial state - Total energy: {info['total_energy']:.4f}")
    
    # Apply different forces and observe response
    forces = [5.0, -3.0, 8.0, 0.0, -5.0]
    for i, force in enumerate(forces):
        obs, reward, term, trunc, info = advanced_env.step(force)
        print(f"Step {i+1}: F={force:4.1f}N, Last osc: pos={obs[0]:.3f}, vel={obs[1]:.3f}, "
              f"Energy={info['total_energy']:.4f}")

    advanced_env.plot_results()
    advanced_env.close()
    return advanced_env


def example_7_fixed_end_oscillator():
    """Example 7: Fixed end oscillator."""
    print("\n" + "="*50)
    print("Example 7: Fixed End Oscillator")
    print("="*50)
    
    # Create system with fixed end oscillator
    fixed_env = CoupledOscillators(
        N=4, 
        k=2.0,  # Adjacent coupling
        fixed_end_oscillator=True,
        end_frequency=2.0,  # 2 Hz oscillation
        end_amplitude=0.8,
        end_phase=0.0
    )
    
    print("System with fixed end oscillator:")
    sys_info = fixed_env.get_system_info()
    print(f"N = {sys_info['N']}")
    print(f"Fixed end oscillator: {sys_info['fixed_end_oscillator']}")
    print(f"End frequency: {sys_info['end_frequency']} Hz")
    print(f"End amplitude: {sys_info['end_amplitude']}")
    print(f"Observing oscillator {sys_info['obs_oscillator_idx']} (second-to-last)")
    
    # Test the system
    obs, info = fixed_env.reset(seed=456)
    print(f"\nInitial observation: pos={obs[0]:.3f}, vel={obs[1]:.3f}")
    print(f"End oscillator initial pos: {info['last_oscillator_position']:.3f}")
    
    # Apply forces and observe system response
    for i in range(10):
        force = 3.0 * np.sin(i * 0.5)  # Variable force
        obs, reward, term, trunc, info = fixed_env.step(force)
        
        if i % 3 == 0:  # Print every 3rd step
            print(f"Step {i+1}: F={force:5.2f}N, obs_osc: pos={obs[0]:.3f}, vel={obs[1]:.3f}, "
                  f"end_osc: pos={info['last_oscillator_position']:.3f}")

    fixed_env.plot_results()
    fixed_env.close()
    return fixed_env


def example_8_pid_control():
    """Example 8: PID controller for position control."""
    print("\n" + "="*50)
    print("Example 8: PID Controller for Position Control")
    print("="*50)
    
    # Create oscillator system  
    pid_env = CoupledOscillators(N=4, k=1.5, damping=0.05, dt=0.01, max_force=20.0)
    
    # Create PID controller to control first oscillator position
    pid_controller = PIDController(
        kp=25.0,     # Proportional gain
        ki=5.0,      # Integral gain  
        kd=2.0,      # Derivative gain
        dt=0.01,     # Must match simulation dt
        output_limits=(-20.0, 20.0),    # Force limits
        integral_limits=(-2.0, 2.0)     # Prevent integral windup
    )
    
    print("PID Controller Configuration:")
    print(f"Kp={pid_controller.kp}, Ki={pid_controller.ki}, Kd={pid_controller.kd}")
    print(f"Output limits: {pid_controller.output_limits}")
    
    # Reset system and controller
    obs, info = pid_env.reset(seed=789)
    pid_controller.reset()
    
    # Reference trajectory: step input, then sinusoidal
    pid_data = {'time': [], 'reference': [], 'actual': [], 'force': [], 'error': []}
    
    print(f"\nPID Control Test - Step Response then Sinusoidal Tracking")
    for i in range(1000):  # 10 seconds at dt=0.01
        t = pid_env.time
        
        # Reference trajectory
        # if t < 3.0:
        #     reference = 0.8  # Step to 0.8
        # else:
        #     reference = 0.8 + 0.3 * np.sin(2 * np.pi * 0.5 * (t - 3.0))  # Sine wave

        reference = 0.8
        
        # Current position of first oscillator (what we're controlling)
        current_position = info['first_oscillator_position']
        
        # PID control calculation
        control_force = pid_controller.update(reference, current_position)
        
        # Apply control force to first oscillator
        obs, reward, term, trunc, info = pid_env.step(control_force)
        
        # Store data for analysis
        if i % 10 == 0:  # Store every 10th point
            pid_data['time'].append(t)
            pid_data['reference'].append(reference)
            pid_data['actual'].append(current_position)
            pid_data['force'].append(control_force)
            pid_data['error'].append(reference - current_position)
        
        # Print progress
        if i % 200 == 0:  # Every 2 seconds
            error = reference - current_position
            print(f"t={t:.1f}s: ref={reference:.3f}, actual={current_position:.3f}, "
                  f"error={error:.3f}, force={control_force:.1f}N")
    
    # Analysis
    final_error = abs(pid_data['reference'][-1] - pid_data['actual'][-1])
    max_error = max(abs(e) for e in pid_data['error'])
    print(f"\nPID Performance:")
    print(f"Final tracking error: {final_error:.4f}")
    print(f"Maximum error: {max_error:.4f}")
    print(f"Final control force: {pid_data['force'][-1]:.2f}N")
    
    # Plot PID control results
    if len(pid_data['time']) > 0:
        plt.figure(figsize=(12, 10))
        
        # Position tracking
        plt.subplot(3, 1, 1)
        plt.plot(pid_data['time'], pid_data['reference'], 'r--', linewidth=2, label='Reference')
        plt.plot(pid_data['time'], pid_data['actual'], 'b-', linewidth=2, label='Actual')
        plt.ylabel('Position')
        plt.title('PID Position Control')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Control force
        plt.subplot(3, 1, 2)
        plt.plot(pid_data['time'], pid_data['force'], 'g-', linewidth=2, label='Control Force')
        plt.ylabel('Force (N)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Tracking error
        plt.subplot(3, 1, 3)
        plt.plot(pid_data['time'], pid_data['error'], 'r-', linewidth=2, label='Tracking Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    pid_env.close()
    return pid_env, pid_controller, pid_data


def run_all_examples():
    """Run all validation examples."""
    print("Coupled Oscillators Simulation - Validation Examples")
    print("=" * 60)
    
    # Run all examples
    example_1_force_sinusoidal()
    example_2_step_force()
    example_3_chirp_signal()
    example_4_analysis()
    example_5_gymnasium_random()
    example_6_advanced_configuration()
    example_7_fixed_end_oscillator()
    example_8_pid_control()
    
    print("\n" + "="*60)
    print("All validation examples completed successfully!")
    print("\nTo use this simulation:")
    print("Force-based Gymnasium Environment with optional PID control:")
    print("   # Regular system")
    print("   env = CoupledOscillators(N=4, m=[1,2,1,3], k=coupling_matrix)")
    print("   # With fixed end oscillator")
    print("   env = CoupledOscillators(N=5, fixed_end_oscillator=True, end_frequency=1.5)")
    print("   # With PID controller")
    print("   pid = PIDController(kp=10, ki=1, kd=0.5, dt=env.dt)")
    print("   obs, info = env.reset()")
    print("   force = pid.update(reference, actual_position)")
    print("   obs, reward, term, trunc, info = env.step(force)")


if __name__ == "__main__":
    run_all_examples()
