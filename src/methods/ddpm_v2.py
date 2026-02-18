"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM_V2(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.device = device
        # Linear beta schedule
        self.get_betas = lambda t: self.beta_start + t * (self.beta_end - self.beta_start) / (self.num_timesteps - 1)

        # Precompute schedule buffers (create on target device/dtype)
        t_all = torch.arange(self.num_timesteps, dtype=torch.float32, device=self.device)
        betas = self.get_betas(t_all).clamp(1e-8, 0.999)
        self.register_buffer("betas", betas)

        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        ones_prev = torch.ones(1, dtype=alphas_cumprod.dtype, device=alphas_cumprod.device)
        self.register_buffer("alphas_cumprod_prev", torch.cat([ones_prev, alphas_cumprod[:-1]], dim=0))

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # posterior q(x_{t-1} | x_t, x_0) variance
        posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.register_buffer("posterior_log_variance_clipped", torch.log(self.posterior_variance))

        # posterior mean coefficients
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Extract values from a 1-D buffer 'a' at indices 't' and reshape for broadcasting to x_shape.
        """
        if t.dtype != torch.long:
            t = t.long()
        out = a.gather(0, t.clamp(0, self.num_timesteps - 1))
        return out.view(out.shape[0], *([1] * (len(x_shape) - 1)))


    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            t: Time steps of shape (batch_size,)
        
        Returns:
            x_t: Noisy samples at time t
            noise: The noise added to x_0
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x_0.device)
        x_t, _ = self.forward_process(x_0, t)
        
        x0_pred = self.model(x_t, t) # predict x0 directly
        mse = F.mse_loss(x0_pred, x_0)
        metric = {'loss': mse.item(),
                  't_mean': float(t.float().mean().detach().cpu()),
                  't_std': float(t.float().std().detach().cpu())
                  }
        return mse, metric


    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments
        
        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        if t.dtype != torch.long:
            t = t.long()
        
        # predict x0
        x0_pred = self.model(x_t, t)
        x0_pred = x0_pred.clamp(-1.0, 1.0)
        
        # compute mean of p(x_{t-1} | x_t)
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0_pred + coef2 * x_t
        

        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        # no noise when t == 0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))

        x_prev = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
        return x_prev
        
        

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()

        C, H, W = image_shape
        x_t = torch.randn(batch_size, C, H, W, device=self.device)

        if num_steps is None or num_steps == self.num_timesteps:
            timesteps = range(self.num_timesteps - 1, -1, -1)
        else:
            # Create a sequence of timesteps linearly spaced from T-1 to 0
            timesteps = torch.linspace(self.num_timesteps - 1, 0, steps=num_steps, device=self.device).long().tolist()

        for ti in timesteps:
            t = torch.full((batch_size,), int(ti), device=self.device, dtype=torch.long)
            x_t = self.reverse_process(x_t, t)

        return x_t

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        # Move this module (buffers and submodules) to `device`.
        # Use `nn.Module.to` directly so buffers (schedules) are moved as well.
        nn.Module.to(self, device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        state["beta_start"] = self.beta_start
        state["beta_end"] = self.beta_end
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
        )
