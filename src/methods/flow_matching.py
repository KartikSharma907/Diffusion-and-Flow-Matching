"""
Flow Matching method implementation.

Flow Matching uses a continuous-time formulation where we learn a velocity field
that transforms a source distribution (noise) to a target distribution (data).

Key concepts:
- Forward path: X_t = (1-t) * X_0 + t * X_1 (CondOT/linear interpolation)
  where X_0 ~ N(0,I) is noise and X_1 is data
- Target velocity: dX_t/dt = X_1 - X_0
- Training: Minimize MSE between predicted velocity and target velocity
- Sampling: Solve ODE dX/dt = v_theta(X_t, t) from t=0 to t=1
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class FlowMatching(BaseMethod):
    """
    Flow Matching with Conditional Optimal Transport (CondOT) path.
    
    Uses linear interpolation path: X_t = (1-t) * X_0 + t * X_1
    where X_0 is noise and X_1 is data.
    
    The model learns to predict the velocity field v(X_t, t) ≈ X_1 - X_0.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int = 1000,
        sigma_min: float = 1e-4,
    ):
        """
        Initialize FlowMatching.
        
        Args:
            model: Neural network that predicts velocity v(x, t)
            device: Device to run computations on
            num_timesteps: Number of discretization steps for sampling
            sigma_min: Small value to avoid numerical issues at t=0
        """
        super().__init__(model, device)
        
        self.num_timesteps = int(num_timesteps)
        self.sigma_min = float(sigma_min)
        self.device = device

    # =========================================================================
    # Forward process (interpolation path)
    # =========================================================================

    def forward_process(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the conditional probability path using CondOT (linear) scheduler.
        
        Path: X_t = (1-t) * X_0 + t * X_1
        Velocity: dX_t/dt = X_1 - X_0
        
        Args:
            x_0: Source samples (noise) of shape (batch_size, channels, height, width)
            x_1: Target samples (data) of shape (batch_size, channels, height, width)
            t: Time steps of shape (batch_size,) in [0, 1]
        
        Returns:
            x_t: Interpolated samples at time t
            velocity: Target velocity (X_1 - X_0)
        """
        # Expand t for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1)
        t_expanded = t.view(-1, *([1] * (x_1.ndim - 1)))
        
        # CondOT path: X_t = (1-t) * X_0 + t * X_1
        # sigma_t = 1 - t, alpha_t = t
        sigma_t = 1.0 - t_expanded
        alpha_t = t_expanded
        
        x_t = sigma_t * x_0 + alpha_t * x_1
        
        # Target velocity: dX_t/dt = d_sigma_t * X_0 + d_alpha_t * X_1 = -X_0 + X_1 = X_1 - X_0
        velocity = x_1 - x_0
        
        return x_t, velocity

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Flow Matching loss (velocity matching).
        
        Loss = E_{t, X_0, X_1} [ || v_theta(X_t, t) - (X_1 - X_0) ||^2 ]
        
        Args:
            x_0: Clean data samples (X_1 in FM notation) of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging
        """
        # Note: In this codebase, x_0 is the input data (which is X_1 in FM notation)
        # We rename for clarity
        x_1 = x_0  # Target data
        B = x_1.shape[0]
        
        # Sample source noise X_0 ~ N(0, I)
        noise = torch.randn_like(x_1)
        
        # Sample time uniformly t ~ U[0, 1]
        # Use sigma_min to avoid numerical issues near t=0
        t = torch.rand(B, device=x_1.device) * (1.0 - self.sigma_min) + self.sigma_min
        
        # Get interpolated sample and target velocity
        x_t, target_velocity = self.forward_process(noise, x_1, t)
        
        # Predict velocity using the model
        # The model expects discrete timesteps, so we scale t to [0, num_timesteps-1]
        t_discrete = (t * (self.num_timesteps - 1)).long()
        predicted_velocity = self.model(x_t, t_discrete)
        
        # MSE loss between predicted and target velocity
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        metrics = {
            'loss': loss.item(),
            't_mean': float(t.float().mean().detach().cpu()),
            't_std': float(t.float().std().detach().cpu()),
        }
        
        return loss, metrics

    # =========================================================================
    # Reverse process (sampling via ODE)
    # =========================================================================
    
    @torch.no_grad()
    def euler_step(self, x_t: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """
        Perform one Euler step for the ODE: dX/dt = v_theta(X, t)
        
        Args:
            x_t: Current samples at time t
            t: Current time (scalar in [0, 1])
            dt: Time step size
        
        Returns:
            x_next: Samples at time t + dt
        """
        B = x_t.shape[0]
        
        # Convert continuous time to discrete timestep for model
        t_discrete = torch.full((B,), int(t * (self.num_timesteps - 1)), 
                                device=x_t.device, dtype=torch.long)
        
        # Predict velocity
        velocity = self.model(x_t, t_discrete)
        
        # Euler update: X_{t+dt} = X_t + dt * v(X_t, t)
        x_next = x_t + dt * velocity
        
        return x_next

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        method: str = "euler",
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples by solving the ODE from t=0 (noise) to t=1 (data).
        
        ODE: dX/dt = v_theta(X, t), X(0) ~ N(0, I)
        
        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            num_steps: Number of ODE solver steps (defaults to num_timesteps)
            method: ODE solver method ('euler' or 'midpoint')
            **kwargs: Additional arguments
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        
        C, H, W = image_shape
        
        # Start from noise at t=0
        x_t = torch.randn(batch_size, C, H, W, device=self.device)
        
        # Number of steps for ODE integration
        if num_steps is None:
            num_steps = self.num_timesteps
        
        # Time step size
        dt = 1.0 / num_steps
        
        # Integrate from t=0 to t=1
        for step in range(num_steps):
            t = step / num_steps
            
            if method == "euler":
                x_t = self.euler_step(x_t, t, dt)
            elif method == "midpoint":
                # Midpoint method for better accuracy
                x_mid = self.euler_step(x_t, t, dt / 2)
                t_mid = t + dt / 2
                
                B = x_t.shape[0]
                t_discrete = torch.full((B,), int(t_mid * (self.num_timesteps - 1)), 
                                        device=x_t.device, dtype=torch.long)
                velocity_mid = self.model(x_mid, t_discrete)
                x_t = x_t + dt * velocity_mid
            else:
                raise ValueError(f"Unknown method: {method}. Use 'euler' or 'midpoint'.")
        
        return x_t

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "FlowMatching":
        """Move this module to the specified device."""
        nn.Module.to(self, device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        """Return state dict including config parameters."""
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        state["sigma_min"] = self.sigma_min
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "FlowMatching":
        """
        Create FlowMatching instance from config dict.
        
        Args:
            model: Neural network for velocity prediction
            config: Configuration dictionary
            device: Device to use
        
        Returns:
            FlowMatching instance
        """
        # Support both 'flow_matching' and 'ddpm' config keys for compatibility
        fm_config = config.get("flow_matching", config.get("ddpm", config))
        return cls(
            model=model,
            device=device,
            num_timesteps=fm_config.get("num_timesteps", 1000),
            sigma_min=fm_config.get("sigma_min", 1e-4),
        )
