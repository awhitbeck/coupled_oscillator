import numpy as np
from simulator import CoupledOscillators
from pid_controller import PIDController
# Example usage and demonstration
if __name__ == "__main__":
    # Create oscillator system
    
    # Create a custom coupling matrix (all-to-all with different strengths)
    coupling_matrix = np.array([
        [0.0,  1.0],#, 0.001],  # Oscillator 0 couplings
        [1.0,  0.0]#, 0.0],  # Oscillator 1 couplings  
        #[0.001,0.0, 0.0]  # Oscillator 2 couplings
    ])
    coupling_matrix = coupling_matrix/10

    masses = [0.01,0.001]#,0.0001]
    
    oscillators = CoupledOscillators(N=2, k=coupling_matrix, m=masses, damping=0.002, dt=0.002,
                                     fixed_end_oscillator=True,end_amplitude=1.0)

    update_factor = 5
    
    print("Coupled Oscillators Simulation")
    print("=" * 40)
    print(f"Number of oscillators: {oscillators.N}")
    print(f"Coupling constant: {oscillators.coupling_matrix}")
    print(f"Mass: {oscillators.masses}")
    print(f"Damping: {oscillators.damping}")
    print(f"Time step: {oscillators.dt}")
    print()

    # Create PID controller to control first oscillator position
    pid_controller = PIDController(
        kp=25.0,     # Proportional gain
        ki=5.0,      # Integral gain  
        kd=2.0,      # Derivative gain
        dt=oscillators.dt*update_factor,     # Must match simulation dt
        output_limits=(-20.0, 20.0),    # Force limits
        integral_limits=(-2.0, 2.0)     # Prevent integral windup
    )
    
    print("PID Controller Configuration:")
    print(f"Kp={pid_controller.kp}, Ki={pid_controller.ki}, Kd={pid_controller.kd}")
    print(f"Output limits: {pid_controller.output_limits}")
    
    print("-" * 35)
    
    obs, info = oscillators.reset()
    for i in range(5000):  # 10 seconds at dt=0.01
        t = oscillators.time
        # Sinusoidal force
        #force = -2.0 * np.sin(2 * np.pi * 1.0 * t-np.pi/4)  # 2N amplitude, 1 Hz
        # PID control
        if i % update_factor == 0:
            force = pid_controller.update(0.0,obs[0])
        obs, reward, term, trunc, info = oscillators.step(force)
        #print(force)
        if i % 200 == 0:  # Print every 2 seconds
            print(f"t={t:.2f}s, force={force:.3f}N, last_osc={obs[0]:.4f}")
    
    print(f"Simulation completed: {len(oscillators.time_history)} time steps")
    print(f"Final time: {oscillators.time:.2f} s")
    print(f"Last oscillator final position: {obs[0]:.4f}")
    print()
    
    # Calculate square of mean position over mean of position squared
    positions = np.array(oscillators.position_history)  # Shape: (time_steps, N_oscillators)
    
    # For each oscillator, calculate the requested ratio
    for osc_idx in range(oscillators.N):
        osc_positions = positions[:, osc_idx]
        
        mean_position = np.mean(osc_positions)
        mean_position_squared = np.mean(osc_positions**2)
        
        # Square of mean position over mean of position squared
        if mean_position_squared != 0:
            ratio = (mean_position**2) / mean_position_squared
        else:
            ratio = 0.0
        
        print(f"Oscillator {osc_idx}:")
        print(f"  Mean position: {mean_position:.6f}")
        print(f"  Mean of position squared: {mean_position_squared:.6f}")
        print(f"  (Mean position)² / Mean(position²): {ratio:.6f}")
    
    print()
    oscillators.plot_results(show_all=True)
    
