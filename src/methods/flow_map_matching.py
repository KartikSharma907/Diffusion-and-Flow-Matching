"""
Flow Map Matching method implementation.

Flow Map Matching learns flow maps X_{s,t}(x) that satisfy the Lagrangian PDE:
    ∂_t X_{s,t}(x) = b_t(X_{s,t}(x)),  X_{s,s}(x) = x

The flow map is parameterized as:
    X_{s,t}(x) = x + (t-s) * v_θ(s, t, x)

This enforces the boundary condition X_{s,s}(x) = x.

Training objective (Prop. 3.11 from paper):
    L = E[|∂_t X_{s,t}(X_{t,s}(I_t)) - I_dot_t|²] + E[|X_{s,t}(X_{t,s}(I_t)) - I_t|²]

where I_t is the deterministic linear interpolant I_t = (1-t)*x_0 + t*x_1.

Note: This implementation uses w_{s,t} = 1 (uniform weighting over [0,1]²).

Reference: https://arxiv.org/abs/2406.07507
"""

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp, vmap

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .base import BaseMethod


class FlowMapMatching(BaseMethod):
    """
    Flow Map Matching with diagonal annealing.

    Uses deterministic linear interpolation path:
        I_t = (1-t) * x_0 + t * x_1
    where x_0 is noise and x_1 is data.

    The model learns velocity-like predictions v_θ(s, t, x) where:
        X_{s,t}(x) = x + (t-s) * v_θ(s, t, x)

    Training enforces the Lagrangian PDE via composition loss (Prop. 3.11):
        Derivative matching: E[|∂_t X_{s,t}(X_{t,s}(I_t)) - I_dot_t|²]
        Reconstruction: E[|X_{s,t}(X_{t,s}(I_t)) - I_t|²]
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int = 1000,
        sigma_min: float = 1e-4,
        annealing_schedule: str = "linear",
        initial_gap: float = 0.1,
        warmup_steps: int = 10000,
        reconstruction_weight: float = 1.0,
    ):
        """
        Initialize FlowMapMatching.

        Args:
            model: Neural network (DualTimeUNet) that predicts v_θ(s, t, x)
            device: Device to run computations on
            num_timesteps: Number of discretization steps for sampling
            sigma_min: Small value to avoid numerical issues at boundaries
            annealing_schedule: Schedule for diagonal annealing ("linear", "cosine", or "none")
            initial_gap: Initial maximum value for |t-s| during training
            warmup_steps: Number of steps over which to anneal from initial_gap to 1.0
            reconstruction_weight: Weight for reconstruction term (default 1.0)
        """
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.sigma_min = float(sigma_min)
        self.annealing_schedule = annealing_schedule
        self.initial_gap = float(initial_gap)
        self.warmup_steps = int(warmup_steps)
        self.reconstruction_weight = float(reconstruction_weight)
        self.device = device

        # Track training step for annealing
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

    # =========================================================================
    # Interpolant (deterministic linear path)
    # =========================================================================
    def _func_model(self):
        """
        Return the underlying module for torch.func transforms.

        If model is wrapped in DDP, return .module. Otherwise return model.
        """
        return self.model.module if hasattr(self.model, "module") else self.model

    def interpolant_I_t(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the deterministic linear interpolant I_t = (1-t) * x0 + t * x1.

        Args:
            t: Time tensor of shape (batch_size,) in [0, 1]
            x0: Source samples (noise) of shape (batch_size, C, H, W)
            x1: Target samples (data) of shape (batch_size, C, H, W)

        Returns:
            I_t: Interpolant at time t
        """
        t_expanded = t.view(-1, 1, 1, 1)
        return (1.0 - t_expanded) * x0 + t_expanded * x1

    def interpolant_I_t_dot(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute time derivative of interpolant: dI_t/dt = -x0 + x1.

        Args:
            t: Time tensor (not used for linear interpolant)
            x0: Source samples
            x1: Target samples

        Returns:
            dI_t/dt: Time derivative (constant for linear path)
        """
        return x1 - x0

    # =========================================================================
    # Flow map: X_{s,t}(x) = x + (t-s) * v_θ(s, t, x)
    # =========================================================================

    def flow_map(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flow map X_{s,t}(x) = x + (t-s) * v_θ(s, t, x).

        Enforces boundary condition X_{s,s}(x) = x.

        Args:
            s: Source time (batch_size,)
            t: Target time (batch_size,)
            x: State (batch_size, C, H, W)

        Returns:
            X_{s,t}(x): Transformed state
        """
        v_theta = self.model(x, s, t)
        delta = (t - s).view(-1, 1, 1, 1)
        return x + delta * v_theta

    # =========================================================================
    # Time pair sampling with diagonal annealing
    # =========================================================================

    def get_time_gap(self, step: int) -> float:
        """
        Compute current maximum |t-s| based on training progress.

        Curriculum learning: start small, increase to 1.0.

        Args:
            step: Current training step

        Returns:
            Maximum gap for this step
        """
        progress = min(1.0, step / self.warmup_steps)

        if self.annealing_schedule == "linear":
            return self.initial_gap + (1.0 - self.initial_gap) * progress
        elif self.annealing_schedule == "cosine":
            return self.initial_gap + (1.0 - self.initial_gap) * (1 - math.cos(progress * math.pi)) / 2
        else:
            return 1.0

    def sample_time_pairs(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        current_gap = self.get_time_gap(self._step.item())

        sigma = self.sigma_min
        max_gap = 1.0 - 2.0 * sigma
        gap = min(current_gap, max_gap)

        # We sample delta with density p(delta) ∝ (max_gap - delta) on [sigma, gap]
        # (uniform over the feasible region in (s,t))
        u = torch.rand(batch_size, device=self.device)

        a = sigma
        b = gap
        # Inverse-CDF for p(δ) ∝ (b - δ) on [a,b]:
        # CDF(δ) = 1 - ((b-δ)^2 / (b-a)^2)  =>  δ = b - (b-a)*sqrt(1-u)
        delta = b - (b - a) * torch.sqrt(1.0 - u)

        # Now sample s uniformly from feasible interval given delta
        s = sigma + torch.rand(batch_size, device=self.device) * (max_gap - delta)
        t = s + delta

        if self.training:
            self._step += 1

        return s, t
    # =========================================================================
    # Loss computation (Prop. 3.11)
    # =========================================================================

    def compute_loss(self, x_1: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Flow Map Matching loss from Prop. 3.11.

        Combines:
        1. Derivative matching: |∂_t X_{s,t}(X_{t,s}(I_t)) - I_dot_t|²
        2. Reconstruction: |X_{s,t}(X_{t,s}(I_t)) - I_t|²

        Note: Uses w_{s,t} = 1 (uniform weighting).

        Args:
            x_1: Clean data samples (batch_size, C, H, W)

        Returns:
            loss: Total loss
            metrics: Logging dict
        """
        B = x_1.shape[0]

        x0 = torch.randn_like(x_1)
        s, t = self.sample_time_pairs(B)

        I_t = self.interpolant_I_t(t, x0, x_1)
        I_t_dot = self.interpolant_I_t_dot(t, x0, x_1)

        loss, loss_dict = self.lagrangian_loss(s, t, I_t, I_t_dot)

        metrics = {
            'loss': loss.item(),
            's_mean': s.mean().item(),
            't_mean': t.mean().item(),
            'delta_t_mean': (t - s).mean().item(),
            'current_gap': self.get_time_gap(self._step.item()),
            **loss_dict
        }

        return loss, metrics

    def lagrangian_loss(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        I_t: torch.Tensor,
        I_t_dot: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Lagrangian loss via composition and JVP (Prop. 3.11).

        Optimized: Uses analytical derivative ∂_t X = v_θ + (t-s) * ∂_t v_θ
        This only requires JVP through v_θ, not the entire flow map.

        Steps:
        1. Compute X_{t,s}(I_t) - reverse-direction learned map
        2. Compute v_θ(s, t, X_{t,s}(I_t)) and ∂_t v_θ using JVP
        3. Compute ∂_t X analytically: v_θ + (t-s) * ∂_t v_θ
        4. Derivative loss: |∂_t X_{s,t}(X_{t,s}(I_t)) - I_t_dot|²
        5. Reconstruction loss: |X_{s,t}(X_{t,s}(I_t)) - I_t|²

        Args:
            s: Source time (batch_size,)
            t: Target time (batch_size,)
            I_t: Interpolant at t (batch_size, C, H, W)
            I_t_dot: Time derivative (batch_size, C, H, W)

        Returns:
            total_loss, loss_components
        """
        B = I_t.shape[0]

        # Step 1: Reverse-direction learned map X_{t,s}(I_t)
        # Can use AMP here (no JVP)
        X_ts_It = self.flow_map(t, s, I_t)

        # Detach for Lagrangian frame (evaluate at fixed point)
        X_ts_It_fixed = X_ts_It.detach()

        # Step 2: Compute v_θ(s, t, X_{t,s}(I_t)) and ∂_t v_θ using JVP
        # IMPORTANT: JVP must be in fp32 for stability with torch.func
        # Temporarily disable autocast for this section
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Ensure inputs are fp32 for JVP
            s_fp32 = s.float()
            t_fp32 = t.float()
            X_ts_It_fp32 = X_ts_It_fixed.float()
            
        m = self._func_model()
        
        def model_single(t_single: torch.Tensor, s_single: torch.Tensor, x_single: torch.Tensor) -> torch.Tensor:
            """
            Model prediction for single sample as function of t.

            Returns v_θ(s, t, x) of shape (C, H, W)
            """
            s_batch = s_single.unsqueeze(0)
            t_batch = t_single.unsqueeze(0)
            x_batch = x_single.unsqueeze(0)

            # v_theta = self.model(x_batch, s_batch, t_batch)
            v_theta = m(x_batch, s_batch, t_batch)
            return v_theta.squeeze(0)

        def compute_v_and_dv_dt(s_i: torch.Tensor, t_i: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute v_θ and ∂_t v_θ for single sample.

            Returns:
                v_θ: Model prediction
                ∂_t v_θ: Time derivative of prediction
            """
            def fn(t_val):
                return model_single(t_val, s_i, x_i)

            primals = (t_i,)
            tangents = (torch.ones_like(t_i),)

            v_theta, dv_dt = jvp(fn, primals, tangents)
            return v_theta, dv_dt

        # Vectorize over batch
        # batched_jvp = vmap(compute_v_and_dv_dt, in_dims=(0, 0, 0))
        was_training = m.training
        m.eval()
        try:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                # Vectorize over batch (JVP path)
                batched_jvp = vmap(compute_v_and_dv_dt, in_dims=(0, 0, 0), randomness="different")
                v_theta, dv_dt = batched_jvp(s_fp32, t_fp32, X_ts_It_fp32)

                # --- NEW: delta in fp32 (matches v_theta/dv_dt dtype) ---
                delta = (t_fp32 - s_fp32).view(-1, 1, 1, 1)

                # Step 3: Analytical derivative
                partial_t_X_st = v_theta + delta * dv_dt

                # Step 4: Derivative matching loss
                derivative_loss = F.mse_loss(partial_t_X_st, I_t_dot.float())

                # Step 5: Reconstruction loss (use SAME model handle `m`)
                # backprop through X_ts_It (NOT fixed/detached)
                v_theta_rec = m(X_ts_It.float(), s_fp32, t_fp32)
                X_st_X_ts_It = X_ts_It.float() + delta * v_theta_rec
                reconstruction_loss = F.mse_loss(X_st_X_ts_It, I_t.float())

        finally:
            m.train(was_training)

        # Total loss (Prop. 3.11)
        total_loss = derivative_loss + self.reconstruction_weight * reconstruction_loss

        loss_components = {
            'derivative_loss': derivative_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
        }

        return total_loss, loss_components

    # =========================================================================
    # Sampling via sequential composition
    # =========================================================================

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples by sequential composition.

        x_final = X_{t_{N-1},t_N}( ... X_{t_0,t_1}(x_0) ... )

        Args:
            batch_size: Number of samples
            image_shape: (C, H, W)
            num_steps: Number of steps (default: num_timesteps)

        Returns:
            Generated samples
        """
        self.eval_mode()

        C, H, W = image_shape
        x = torch.randn(batch_size, C, H, W, device=self.device)

        if num_steps is None:
            num_steps = self.num_timesteps

        # Use [sigma_min, 1-sigma_min] for consistency with training
        ts = torch.linspace(self.sigma_min, 1.0 - self.sigma_min, num_steps + 1, device=self.device)

        for i in range(num_steps):
            s = ts[i].expand(batch_size)
            t = ts[i + 1].expand(batch_size)
            x = self.flow_map(s, t, x)

        return x

    # =========================================================================
    # Device / state management
    # =========================================================================

    def to(self, device: torch.device) -> "FlowMapMatching":
        """Move to device."""
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        """
        Return state dict (tensors only, as per PyTorch convention).

        Only includes _step buffer. Config params are saved separately in checkpoint.
        """
        state = super().state_dict()
        # _step is already included as a buffer, no need to add manually
        return state

    def load_state_dict(self, state_dict: Dict, strict: bool = True):
        """
        Load state dict.

        Properly handles _step buffer without breaking device/buffer semantics.
        """
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "FlowMapMatching":
        """Create from config."""
        fm_config = config.get("flow_map_matching", {})
        return cls(
            model=model,
            device=device,
            num_timesteps=fm_config.get("num_timesteps", 1000),
            sigma_min=fm_config.get("sigma_min", 1e-4),
            annealing_schedule=fm_config.get("annealing_schedule", "linear"),
            initial_gap=fm_config.get("initial_gap", 0.1),
            warmup_steps=fm_config.get("warmup_steps", 10000),
            reconstruction_weight=fm_config.get("reconstruction_weight", 1.0),
        )
