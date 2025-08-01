import numpy as np
from typing import Optional, Tuple, Dict, Union
from pathlib import Path


class PIDController:
    """
    PID (Proportional-Integral-Derivative) controller for position control.
    
    Computes the control force needed to bring an oscillator's actual position
    to a desired reference position using PID feedback control.
    """
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, dt: float = 0.01, 
                 output_limits: Optional[Tuple[float, float]] = None,
                 integral_limits: Optional[Tuple[float, float]] = None,
                 config_file: Optional[Union[str, Path]] = None):
        """
        Initialize PID controller.
        
        Parameters:
        -----------
        kp : float
            Proportional gain
        ki : float
            Integral gain  
        kd : float
            Derivative gain
        dt : float
            Time step (should match simulation dt)
        output_limits : tuple, optional
            (min, max) limits for controller output force
        integral_limits : tuple, optional
            (min, max) limits for integral term to prevent windup
        config_file : str, Path, or None
            Path to configuration file (JSON or YAML). If provided, overrides other parameters.
        """
        # Load configuration from file if provided
        if config_file is not None:
            try:
                from .config import ExperimentConfig
            except ImportError:
                from config import ExperimentConfig
            
            config = ExperimentConfig(config_file)
            pid_config = config.get_pid_config()
            
            # Override parameters with values from config file
            kp = pid_config.get('kp', kp)
            ki = pid_config.get('ki', ki)
            kd = pid_config.get('kd', kd)
            output_limits = pid_config.get('output_limits', output_limits)
            integral_limits = pid_config.get('integral_limits', integral_limits)
            
            # Get dt from simulation config if not explicitly set in PID config
            sim_config = config.get_simulation_config()
            dt = sim_config.get('dt', dt)
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        
        # Internal state
        self.reset()
    
    def reset(self):
        """Reset controller internal state."""
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = None
    
    def update(self, reference: float, actual: float) -> float:
        """
        Compute PID control output.
        
        Parameters:
        -----------
        reference : float
            Desired position (setpoint)
        actual : float
            Current actual position
            
        Returns:
        --------
        float
            Control force to apply
        """
        # Calculate error
        error = reference - actual
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * self.dt
        if self.integral_limits:
            self.integral = np.clip(self.integral, *self.integral_limits)
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.previous_error) / self.dt
        
        # Calculate total output
        output = proportional + integral + derivative
        
        # Apply output limits
        if self.output_limits:
            output = np.clip(output, *self.output_limits)
        
        # Store for next iteration
        self.previous_error = error
        
        return output
    
    def get_terms(self, reference: float, actual: float) -> Dict[str, float]:
        """
        Get individual PID terms for debugging/analysis.
        
        Parameters:
        -----------
        reference : float
            Desired position (setpoint)
        actual : float
            Current actual position
            
        Returns:
        --------
        dict
            Dictionary with 'error', 'proportional', 'integral', 'derivative', 'output'
        """
        error = reference - actual
        proportional = self.kp * error
        integral_term = self.ki * self.integral
        derivative = self.kd * (error - self.previous_error) / self.dt
        output = proportional + integral_term + derivative
        
        if self.output_limits:
            output_clamped = np.clip(output, *self.output_limits)
        else:
            output_clamped = output
        
        return {
            'error': error,
            'proportional': proportional,
            'integral': integral_term,
            'derivative': derivative,
            'output': output,
            'output_clamped': output_clamped
        }
    
    def set_gains(self, kp: float, ki: float, kd: float):
        """Update PID gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
    
    def set_limits(self, output_limits: Optional[Tuple[float, float]] = None,
                   integral_limits: Optional[Tuple[float, float]] = None):
        """Update controller limits."""
        self.output_limits = output_limits
        self.integral_limits = integral_limits