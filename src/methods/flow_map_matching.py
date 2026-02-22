"""
Flow Map Matching method implementation.

Flow Map Matching learns flow maps X_{s,t}(x) that satisfy the Lagrangian PDE:
    ∂_t X_{s,t}(x) = b_t(X_{s,t}(x)),  X_{s,s}(x) = x

The flow map is parameterized as:
    X_{s,t}(x) = x + (t-s) * net_θ(s, t, x)

where net_θ outputs either:
  - v_θ(s, t, x)  — instantaneous velocity (original, predict_x0=False)
  - u_θ(s, t, x)  — average velocity derived from x̂_θ (x̂-head, predict_x0=True):
        x̂_θ = net_θ(s, t, x)
        u_θ  = (x̂_θ - x) / denom(t-s)   ← pulls x toward x̂_θ
        denom uses a sign-safe epsilon so backward maps (t-s < 0) never cross zero:
          forward  (t-s > 0): denom = (t-s) + ε
          backward (t-s < 0): denom = (t-s) - ε
        NOTE: Pixel MeanFlow (arXiv 2601.22158) uses absolute t as denominator; this
        implementation ties ε to the step size (t-s) for scale consistency across (s,t)
        pairs, which is a design deviation from the paper.

This enforces the boundary condition X_{s,s}(x) = x.

Training objective (Prop. 3.11 from paper):
    L = E[|∂_t X_{s,t}(X_{t,s}(I_t)) - I_dot_t|²] + E[|X_{s,t}(X_{t,s}(I_t)) - I_t|²]

where I_t is the deterministic linear interpolant I_t = (1-t)*x_0 + t*x_1.

Optional semigroup consistency loss over triples (s < t < u):
    L_sg = E[||X_{s,u}(I_s) - X_{t,u}(X_{s,t}(I_s))||²]

Note: This implementation uses w_{s,t} = 1 (uniform weighting over [0,1]²).

References:
    FMM:  https://arxiv.org/abs/2406.07507
    pMF:  https://arxiv.org/abs/2601.22158
"""

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp, vmap

from torch.nn.parallel import DistributedDataParallel as DDP

from .base import BaseMethod


class FlowMapMatching(BaseMethod):
    """
    Flow Map Matching with diagonal annealing, optional semigroup consistency loss,
    and optional pMF-style x̂ reparameterization.

    Uses deterministic linear interpolation path:
        I_t = (1-t) * x_0 + t * x_1
    where x_0 is noise and x_1 is data.

    When predict_x0=False (default):
        Model predicts v_θ(s, t, x) and X_{s,t}(x) = x + (t-s) * v_θ.

    When predict_x0=True (x̂ head):
        Model predicts x̂_θ(s, t, x) (denoised image estimate).
        Average velocity: u_θ = (x̂_θ - x) / denom(t-s)   [sign-safe ε, see flow_map]
        Flow map:         X_{s,t}(x) = x + (t-s) * u_θ

    Training enforces the Lagrangian PDE via composition loss (Prop. 3.11):
        Derivative matching: E[|∂_t X_{s,t}(X_{t,s}(I_t)) - I_dot_t|²]
        Reconstruction:      E[|X_{s,t}(X_{t,s}(I_t)) - I_t|²]
        Semigroup (opt.):    E[||X_{s,u}(I_s) - X_{t,u}(X_{s,t}(I_s))||²]
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
        semigroup_weight: float = 0.0,
        predict_x0: bool = False,
        step_epsilon: float = 1e-3,
        clamp_x: bool = False,
        sg_ramp_steps: int = 50000,
        sg_min_seg: float = 0.05,
        sg_delay_steps: int = 20000,
    ):
        """
        Initialize FlowMapMatching.

        Args:
            model: Neural network (DualTimeUNet) that predicts v_θ(s, t, x) or x̂_θ(s, t, x)
            device: Device to run computations on
            num_timesteps: Number of discretization steps for sampling
            sigma_min: Small value to avoid numerical issues at boundaries
            annealing_schedule: Schedule for diagonal annealing ("linear", "cosine", or "none")
            initial_gap: Initial maximum value for |t-s| during training
            warmup_steps: Number of steps over which to anneal from initial_gap to 1.0
            reconstruction_weight: Weight for reconstruction term (default 1.0)
            semigroup_weight: Weight for semigroup consistency loss over triples (0 = disabled).
                              Adds L_sg = ||X_{s,u}(I_s) - X_{t,u}(X_{s,t}(I_s))||² to the loss.
            predict_x0: If True, network predicts x̂_θ (denoised image) and the average velocity
                        u_θ = (x - x̂_θ) / denom is derived from it. If False (default),
                        network predicts v_θ directly (original behaviour).
                        NOTE: Pixel MeanFlow uses absolute t as denominator; this design
                        uses (t-s) for scale consistency — results may differ from the paper.
            step_epsilon: Numerical stability constant ε for the x̂-head denominator.
                          Applied with sign-safety: forward denom = (t-s)+ε, backward = (t-s)-ε.
                          This ensures backward maps (flow_map(t, s, x)) never produce a
                          denominator that crosses zero. Increase to 5e-3 if loss is noisy.
        """
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.sigma_min = float(sigma_min)
        self.annealing_schedule = annealing_schedule
        self.initial_gap = float(initial_gap)
        self.warmup_steps = int(warmup_steps)
        self.reconstruction_weight = float(reconstruction_weight)
        self.semigroup_weight = float(semigroup_weight)
        self.predict_x0 = bool(predict_x0)
        self.clamp_x = bool(clamp_x)
        self.step_epsilon = float(step_epsilon)
        self.sg_ramp_steps = int(sg_ramp_steps)
        self.sg_min_seg = float(sg_min_seg)
        self.sg_delay_steps = int(sg_delay_steps)
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
    # Flow map: X_{s,t}(x) = x + (t-s) * net_θ(s, t, x)
    # =========================================================================

    def flow_map(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
        clamp_x: bool = False
    ) -> torch.Tensor:
        """
        Compute flow map X_{s,t}(x).

        Enforces boundary condition X_{s,s}(x) = x (since delta = t-s = 0).

        When predict_x0=False:
            X_{s,t}(x) = x + (t-s) * v_θ(s, t, x)

        When predict_x0=True (x̂ head):
            x̂_θ = net_θ(s, t, x)
            denom = (t-s) + ε  if t ≥ s  (forward map)
            denom = (t-s) - ε  if t < s  (backward map, e.g. X_{t,s} in Lagrangian step)
            u_θ  = (x̂_θ - x) / denom   ← pulls x toward x̂_θ
            X_{s,t}(x) = x + (t-s) * u_θ ≈ x̂_θ  when ε ≪ (t-s)

        Args:
            s: Source time (batch_size,)
            t: Target time (batch_size,)
            x: State (batch_size, C, H, W)

        Returns:
            X_{s,t}(x): Transformed state
        """
        m = self._func_model()
        raw = m(x, s, t)
        delta = (t - s).view(-1, 1, 1, 1)

        if self.predict_x0:
            # Sign-safe epsilon: denominator keeps the same sign as delta so it never
            # crosses zero for backward maps (delta < 0) called as flow_map(t, s, x).
            #   forward  (delta > 0): denom = delta + ε  > 0
            #   backward (delta < 0): denom = delta - ε  < 0
            #   boundary (delta = 0): uses +ε; delta * u_theta = 0 anyway → returns x
            if clamp_x:
                # Optional clamping for stability during sampling (can be removed if not needed)
                raw = raw.clamp(-1, 1)
                
            # sign-safe epsilon so denominator never crosses 0 for backward maps
            denom = torch.where(delta >= 0, delta + self.step_epsilon, delta - self.step_epsilon)
            u_theta = (raw - x) / denom         # pull toward x̂_θ: moves x to raw
            return x + delta * u_theta
        else:
            return x + delta * raw              # original v_θ path

    # =========================================================================
    # Time pair / triple sampling with diagonal annealing
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

    def sample_time_triples(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample triples (s, t, u) with 0 ≤ s < t < u ≤ 1 under the same diagonal
        annealing curriculum used for pairs.

        The total spread δ_total = u - s uses the same triangular inverse-CDF as
        sample_time_pairs. The intermediate time t is sampled uniformly between s and u
        with sigma margins so it is strictly interior.

        Note: _step is NOT incremented here — only sample_time_pairs increments it,
        so curriculum advancement is paced by the Lagrangian loss, not the semigroup loss.

        Args:
            batch_size: Number of triples to sample

        Returns:
            (s, t, u): each of shape (batch_size,)
        """
        current_gap = self.get_time_gap(self._step.item())
        sigma = self.sigma_min
        max_gap = 1.0 - 2.0 * sigma
        gap = min(current_gap, max_gap)
        min_delta_total = 2.0 * sigma   # need room for t strictly between s and u

        if gap <= min_delta_total:
            # Degenerate fallback: gap is too small for a proper triple.
            # Re-use pair sampler and set t to the midpoint.
            s, u = self.sample_time_pairs(batch_size)
            t = (s + u) / 2.0
            return s, t, u

        # Sample total spread δ_total = u - s from p(δ) ∝ (gap - δ) on [min_delta_total, gap]
        # (same triangular inverse-CDF as sample_time_pairs)
        a, b = min_delta_total, gap
        u_rand = torch.rand(batch_size, device=self.device)
        delta_total = b - (b - a) * torch.sqrt(1.0 - u_rand)   # ∈ [a, b]

        # Sample s uniformly in the feasible interval
        s = sigma + torch.rand(batch_size, device=self.device) * (max_gap - delta_total)
        u = s + delta_total

        # Sample t uniformly strictly between s and u, keeping sigma margins
        t = s + sigma + torch.rand(batch_size, device=self.device) * (delta_total - 2.0 * sigma)

        return s, t, u

    # =========================================================================
    # Loss computation (Prop. 3.11 + optional semigroup)
    # =========================================================================

    # def compute_loss(self, x_1: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
    #     """
    #     Compute Flow Map Matching loss from Prop. 3.11, plus optional semigroup loss.

    #     Core terms:
    #         1. Derivative matching: |∂_t X_{s,t}(X_{t,s}(I_t)) - I_dot_t|²
    #         2. Reconstruction:      |X_{s,t}(X_{t,s}(I_t)) - I_t|²

    #     Optional semigroup term (when semigroup_weight > 0):
    #         3. L_sg = ||X_{s,u}(I_s) - X_{t,u}(X_{s,t}(I_s))||²
    #            computed on a half-batch to control compute cost.

    #     Note: Uses w_{s,t} = 1 (uniform weighting).

    #     Args:
    #         x_1: Clean data samples (batch_size, C, H, W)

    #     Returns:
    #         loss: Total loss
    #         metrics: Logging dict
    #     """
    #     B = x_1.shape[0]

    #     x0 = torch.randn_like(x_1)
    #     s, t = self.sample_time_pairs(B)

    #     I_t = self.interpolant_I_t(t, x0, x_1)
    #     I_t_dot = self.interpolant_I_t_dot(t, x0, x_1)

    #     total_loss, loss_dict = self.lagrangian_loss(s, t, I_t, I_t_dot)

    #     metrics = {
    #         'loss': total_loss.item(),
    #         's_mean': s.mean().item(),
    #         't_mean': t.mean().item(),
    #         'delta_t_mean': (t - s).mean().item(),
    #         'current_gap': self.get_time_gap(self._step.item()),
    #         **loss_dict
    #     }

    #     # Optional semigroup consistency loss over triples (s < t < u)
    #     if self.semigroup_weight > 0.0:
    #         # Use a half-batch to limit the 3-extra-forward-pass overhead
    #         B_sg = max(1, B // 2)
    #         x0_sg = x0[:B_sg]
    #         x1_sg = x_1[:B_sg]
    #         s3, t3, u3 = self.sample_time_triples(B_sg)
    #         sg_loss, sg_dict = self.semigroup_loss(s3, t3, u3, x0_sg, x1_sg)
    #         total_loss = total_loss + self.semigroup_weight * sg_loss
    #         metrics.update(sg_dict)
    #         metrics['loss'] = total_loss.item()

    #     return total_loss, metrics

    # def semigroup_loss(
    #     self,
    #     s: torch.Tensor,
    #     t: torch.Tensor,
    #     u: torch.Tensor,
    #     x0: torch.Tensor,
    #     x1: torch.Tensor,
    # ) -> Tuple[torch.Tensor, Dict[str, float]]:
    #     """
    #     Semigroup (composition) consistency loss over triples (s, t, u).

    #     Enforces the group law: X_{s,u}(I_s) = X_{t,u}(X_{s,t}(I_s))

    #     Gradient flows symmetrically through both branches — the two-step composition
    #     branch (X_{t,u} ∘ X_{s,t}) and the direct jump branch (X_{s,u}).
    #     This symmetric penalty directly penalises inconsistency in both directions.

    #     If training is noisy, replace `F.mse_loss(X_tu_Xst, X_su)` with
    #     `F.mse_loss(X_tu_Xst, X_su.detach())` to use X_{s,u} as a fixed "teacher" target
    #     (consistency-model style), which stabilises training at the cost of under-penalising
    #     errors in the direct jump.

    #     Args:
    #         s: Source time (batch_size,)
    #         t: Intermediate time (batch_size,), s < t < u
    #         u: Target time (batch_size,)
    #         x0: Noise samples (batch_size, C, H, W)
    #         x1: Data samples (batch_size, C, H, W)

    #     Returns:
    #         loss: Scalar semigroup loss
    #         metrics: Dict with 'semigroup_loss'
    #     """
    #     I_s = self.interpolant_I_t(s, x0, x1)      # reference state at time s

    #     X_st     = self.flow_map(s, t, I_s)         # Φ_{s,t}(I_s)
    #     X_su     = self.flow_map(s, u, I_s)         # Φ_{s,u}(I_s)       — direct jump
    #     X_tu_Xst = self.flow_map(t, u, X_st)        # Φ_{t,u}(Φ_{s,t})  — two-step

    #     # Symmetric penalty: gradient flows through both branches
    #     loss = F.mse_loss(X_tu_Xst, X_su)
    #     return loss, {'semigroup_loss': loss.item()}

    # def lagrangian_loss(
    #     self,
    #     s: torch.Tensor,
    #     t: torch.Tensor,
    #     I_t: torch.Tensor,
    #     I_t_dot: torch.Tensor
    # ) -> Tuple[torch.Tensor, Dict[str, float]]:
    #     """
    #     Compute Lagrangian loss via composition and JVP (Prop. 3.11).

    #     When predict_x0=False: JVP differentiates t ↦ v_θ(s, t, x)  (original)
    #     When predict_x0=True:  JVP differentiates t ↦ u_θ(s, t, x)
    #                            where u_θ = (x - x̂_θ(s,t,x)) / ((t-s) + step_epsilon)
    #                            Forward-mode AD automatically handles the chain rule
    #                            through both the network and the 1/((t-s)+ε) factor.

    #     Steps:
    #     1. Compute X_{t,s}(I_t) - reverse-direction learned map
    #     2. Compute net_θ or u_θ at (s, t, X_{t,s}(I_t)) and its ∂_t via JVP
    #     3. Compute ∂_t X analytically: net + (t-s) * ∂_t net
    #     4. Derivative loss: |∂_t X_{s,t}(X_{t,s}(I_t)) - I_t_dot|²
    #     5. Reconstruction loss: |X_{s,t}(X_{t,s}(I_t)) - I_t|²

    #     Args:
    #         s: Source time (batch_size,)
    #         t: Target time (batch_size,)
    #         I_t: Interpolant at t (batch_size, C, H, W)
    #         I_t_dot: Time derivative (batch_size, C, H, W)

    #     Returns:
    #         total_loss, loss_components
    #     """
    #     # Step 1: Reverse-direction learned map X_{t,s}(I_t)
    #     # Can use AMP here (no JVP)
    #     X_ts_It = self.flow_map(t, s, I_t)

    #     # Detach for Lagrangian frame (evaluate at fixed point)
    #     X_ts_It_fixed = X_ts_It.detach()

    #     # Step 2: Compute net_θ and ∂_t net_θ using JVP
    #     # IMPORTANT: JVP must be in fp32 for stability with torch.func
    #     # Temporarily disable autocast for this section
    #     with torch.amp.autocast(device_type='cuda', enabled=False):
    #         # Ensure inputs are fp32 for JVP
    #         s_fp32 = s.float()
    #         t_fp32 = t.float()
    #         X_ts_It_fp32 = X_ts_It_fixed.float()

    #     m = self._func_model()

    #     # Capture predict_x0 and step_epsilon for use inside vmap/jvp closures
    #     _predict_x0 = self.predict_x0
    #     _step_epsilon = self.step_epsilon

    #     def model_single(t_single: torch.Tensor, s_single: torch.Tensor, x_single: torch.Tensor) -> torch.Tensor:
    #         """
    #         Network output as a function of t_single (for JVP differentiation).

    #         When predict_x0=False: returns v_θ(s, t, x)              shape (C, H, W)
    #         When predict_x0=True:  returns u_θ(s, t, x)              shape (C, H, W)
    #             u_θ = (x̂_θ - x) / ((t-s) + step_epsilon)   pulls toward x̂_θ
    #             x_single is treated as a constant primal (not differentiated).
    #         """
    #         s_batch = s_single.unsqueeze(0)
    #         t_batch = t_single.unsqueeze(0)
    #         x_batch = x_single.unsqueeze(0)
    #         raw = m(x_batch, s_batch, t_batch).squeeze(0)   # x̂_θ or v_θ

    #         if _predict_x0:
    #             denom = (t_single - s_single) + _step_epsilon  # scalar (t-s+ε); t>s in JVP path
    #             return (raw - x_single) / denom                 # u_θ: pull toward x̂_θ
    #         return raw                                           # v_θ(s, t, x)

    #     def compute_net_and_dnet_dt(s_i: torch.Tensor, t_i: torch.Tensor, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #         """
    #         Compute net_θ and ∂_t net_θ for a single sample via JVP.

    #         Returns:
    #             net_θ:    v_θ or u_θ depending on predict_x0
    #             ∂_t net_θ: time derivative of the above
    #         """
    #         def fn(t_val):
    #             return model_single(t_val, s_i, x_i)

    #         primals = (t_i,)
    #         tangents = (torch.ones_like(t_i),)

    #         net_theta, dnet_dt = jvp(fn, primals, tangents)
    #         return net_theta, dnet_dt

    #     was_training = m.training
    #     m.eval()
    #     try:
    #         with torch.amp.autocast(device_type="cuda", enabled=False):
    #             # Vectorize over batch (JVP path)
    #             batched_jvp = vmap(compute_net_and_dnet_dt, in_dims=(0, 0, 0), randomness="different")
    #             net_theta, dnet_dt = batched_jvp(s_fp32, t_fp32, X_ts_It_fp32)

    #             # delta = (t-s), shape (B, 1, 1, 1) for broadcasting
    #             delta = (t_fp32 - s_fp32).view(-1, 1, 1, 1)

    #             # Step 3: Analytical derivative ∂_t X_{s,t}(x) = net_θ + (t-s) * ∂_t net_θ
    #             partial_t_X_st = net_theta + delta * dnet_dt

    #             # Step 4: Derivative matching loss
    #             derivative_loss = F.mse_loss(partial_t_X_st, I_t_dot.float())

    #             # Step 5: Reconstruction loss
    #             # Backprop through X_ts_It (NOT fixed/detached) so gradient flows to Φ_{t,s} too
    #             raw_rec = m(X_ts_It.float(), s_fp32, t_fp32)
    #             if _predict_x0:
    #                 denom_rec = delta + _step_epsilon
    #                 u_theta_rec = (raw_rec - X_ts_It.float()) / denom_rec
    #                 X_st_X_ts_It = X_ts_It.float() + delta * u_theta_rec
    #             else:
    #                 X_st_X_ts_It = X_ts_It.float() + delta * raw_rec
    #             reconstruction_loss = F.mse_loss(X_st_X_ts_It, I_t.float())

    #     finally:
    #         m.train(was_training)

    #     # Total loss (Prop. 3.11)
    #     total_loss = derivative_loss + self.reconstruction_weight * reconstruction_loss

    #     loss_components = {
    #         'derivative_loss': derivative_loss.item(),
    #         'reconstruction_loss': reconstruction_loss.item(),
    #     }

    #     return total_loss, loss_components

    def compute_loss(self, x_1: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Prop 3.11 + stabilized semigroup.
        """
        B = x_1.shape[0]
        device = x_1.device
        dtype = x_1.dtype

        # Noise and time pairs
        x0 = torch.randn_like(x_1)
        s, t = self.sample_time_pairs(B)

        I_t = self.interpolant_I_t(t, x0, x_1)
        I_t_dot = self.interpolant_I_t_dot(t, x0, x_1)

        total_loss, loss_dict = self.lagrangian_loss(s, t, I_t, I_t_dot)

        step_int = int(self._step.item()) if hasattr(self, "_step") else 0

        metrics = {
            "loss": float(total_loss.detach().item()),
            "s_mean": float(s.mean().detach().item()),
            "t_mean": float(t.mean().detach().item()),
            "delta_t_mean": float((t - s).mean().detach().item()),
            "current_gap": float(self.get_time_gap(step_int)) if hasattr(self, "get_time_gap") else float("nan"),
            **loss_dict,
        }

        # -------------------------
        # Stabilized semigroup term
        # -------------------------
        # base_sg_w = float(getattr(self, "semigroup_weight", 0.0))
        # if base_sg_w > 0.0:
        #     # Ramp semigroup ON only after diagonal warmup is done.
        #     warmup = int(getattr(self, "warmup_steps", 0))
        #     sg_ramp_steps = int(getattr(self, "sg_ramp_steps", 50000))  # DEFAULT: 50k ramp

        #     if step_int < warmup:
        #         sg_w = 0.0
        #     else:
        #         # linear ramp from 0 -> base_sg_w over sg_ramp_steps
        #         frac = min(1.0, max(0.0, (step_int - warmup) / max(1, sg_ramp_steps)))
        #         sg_w = base_sg_w * frac

        #     if sg_w > 0.0:
        #         B_sg = max(1, B // 2)
        #         x0_sg = x0[:B_sg]
        #         x1_sg = x_1[:B_sg]

        #         s3, t3, u3 = self.sample_time_triples(B_sg)

        #         sg_loss, sg_dict = self.semigroup_loss(s3, t3, u3, x0_sg, x1_sg)
        #         total_loss = total_loss + (sg_w * sg_loss)

        #         metrics.update(sg_dict)
        #         metrics["semigroup_weight_eff"] = sg_w
        #         metrics["loss"] = float(total_loss.detach().item())
        sg_w_target = float(self.semigroup_weight)
        if sg_w_target > 0.0:
            step_int = int(self._step.item())
            # Start semigroup only AFTER warmup + delay
            sg_start = int(self.warmup_steps) + int(self.sg_delay_steps)
            sg_ramp = max(1, int(self.sg_ramp_steps))

            if step_int < sg_start:
                sg_w_eff = 0.0
            else:
                frac = min(1.0, max(0.0, (step_int - sg_start) / sg_ramp))
                sg_w_eff = sg_w_target * frac

            metrics["semigroup_weight_eff"] = sg_w_eff

            if sg_w_eff > 0.0:
                B_sg = max(1, B // 2)
                x0_sg = x0[:B_sg]
                x1_sg = x_1[:B_sg]
                s3, t3, u3 = self.sample_time_triples(B_sg)
                sg_loss, sg_dict = self.semigroup_loss(s3, t3, u3, x0_sg, x1_sg)

                total_loss = total_loss + sg_w_eff * sg_loss
                metrics.update(sg_dict)
                metrics["loss"] = float(total_loss.detach().item())

        return total_loss, metrics


    # def semigroup_loss(
    #     self,
    #     s: torch.Tensor,
    #     t: torch.Tensor,
    #     u: torch.Tensor,
    #     x0: torch.Tensor,
    #     x1: torch.Tensor,
    # ) -> Tuple[torch.Tensor, Dict[str, float]]:
    #     """
    #     Stabilized semigroup (composition) loss:

    #     - Uses I_s as the anchor (as you had).
    #     - Computes direct jump X_{s,u}(I_s) as a *teacher target* (stop-grad).
    #     - Backprop flows through the two-step composition branch only.
    #     - Optionally skips/downsweights triples with very small segments early (reduces noise).

    #     This is the single most impactful semigroup stabilization for your setup.
    #     """
    #     I_s = self.interpolant_I_t(s, x0, x1)

    #     # Optional: ignore tiny segment constraints (esp. helpful early / with vmap/jvp noise)
    #     sg_min_seg = float(getattr(self, "sg_min_seg", 0.05))  # DEFAULT: 0.05
    #     seg1 = (t - s).abs()
    #     seg2 = (u - t).abs()
    #     mask = (seg1 >= sg_min_seg) & (seg2 >= sg_min_seg)

    #     # If everything masked out, return 0 (no gradient)
    #     if not bool(mask.any()):
    #         zero = I_s.new_zeros(())
    #         return zero, {"semigroup_loss": 0.0, "semigroup_keep_frac": 0.0}

    #     # Apply mask to reduce compute + focus loss on meaningful segments
    #     I_s_m = I_s[mask]
    #     s_m = s[mask]
    #     t_m = t[mask]
    #     u_m = u[mask]

    #     X_st = self.flow_map(s_m, t_m, I_s_m)
    #     X_tu_Xst = self.flow_map(t_m, u_m, X_st)

    #     # Teacher target (stop-grad) for stability
    #     with torch.no_grad():
    #         X_su = self.flow_map(s_m, u_m, I_s_m)

    #     loss = F.mse_loss(X_tu_Xst, X_su)

    #     return loss, {
    #         "semigroup_loss": float(loss.detach().item()),
    #         "semigroup_keep_frac": float(mask.float().mean().detach().item()),
    #         "semigroup_seg1_mean": float(seg1.mean().detach().item()),
    #         "semigroup_seg2_mean": float(seg2.mean().detach().item()),
    #     }
    
    def semigroup_loss(self, s, t, u, x0, x1):
        I_s = self.interpolant_I_t(s, x0, x1)

        seg1 = (t - s).abs()
        seg2 = (u - t).abs()
        mask = (seg1 >= self.sg_min_seg) & (seg2 >= self.sg_min_seg)

        if not bool(mask.any()):
            zero = I_s.new_zeros(())
            return zero, {
                "semigroup_loss": 0.0,
                "semigroup_keep_frac": 0.0,
                "semigroup_seg1_mean": float(seg1.mean().item()),
                "semigroup_seg2_mean": float(seg2.mean().item()),
            }

        I_s = I_s[mask]
        s = s[mask]; t = t[mask]; u = u[mask]

        X_st = self.flow_map(s, t, I_s)
        X_tu_Xst = self.flow_map(t, u, X_st)

        # teacher target: stop-grad direct jump
        with torch.no_grad():
            X_su = self.flow_map(s, u, I_s)

        loss = F.mse_loss(X_tu_Xst, X_su)
        return loss, {
            "semigroup_loss": float(loss.item()),
            "semigroup_keep_frac": float(mask.float().mean().item()),
            "semigroup_seg1_mean": float(seg1.mean().item()),
            "semigroup_seg2_mean": float(seg2.mean().item()),
        }


    def lagrangian_loss(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        I_t: torch.Tensor,
        I_t_dot: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Prop 3.11 Lagrangian loss with device-safe fp32 JVP.
        """
        device_type = I_t.device.type
        B = I_t.shape[0]

        # Reverse learned map X_{t,s}(I_t)
        X_ts_It = self.flow_map(t, s, I_t)
        X_ts_It_fixed = X_ts_It.detach()

        m = self._func_model()
        was_training = m.training
        m.eval()  # avoid dropout noise during jvp

        # JVP must be fp32; disable autocast safely for this device
        with torch.amp.autocast(device_type=device_type, enabled=False):
            s_fp32 = s.float()
            t_fp32 = t.float()
            X_fp32 = X_ts_It_fixed.float()
            I_dot_fp32 = I_t_dot.float()
            I_t_fp32 = I_t.float()

        _predict_x0 = bool(getattr(self, "predict_x0", False))
        _step_epsilon = float(getattr(self, "step_epsilon", 1e-3))

        def model_single(t_single: torch.Tensor, s_single: torch.Tensor, x_single: torch.Tensor) -> torch.Tensor:
            # output: v_theta (predict_x0=False) OR u_theta (predict_x0=True)
            s_batch = s_single.unsqueeze(0)
            t_batch = t_single.unsqueeze(0)
            x_batch = x_single.unsqueeze(0)
            raw = m(x_batch, s_batch, t_batch).squeeze(0)

            if _predict_x0:
                denom = (t_single - s_single) + _step_epsilon
                return (raw - x_single) / denom
            return raw

        def compute_net_and_dnet_dt(s_i: torch.Tensor, t_i: torch.Tensor, x_i: torch.Tensor):
            def fn(tt):
                return model_single(tt, s_i, x_i)

            primals = (t_i,)
            tangents = (torch.ones_like(t_i),)
            net_theta, dnet_dt = jvp(fn, primals, tangents)
            return net_theta, dnet_dt

        try:
            with torch.amp.autocast(device_type=device_type, enabled=False):
                batched_jvp = vmap(compute_net_and_dnet_dt, in_dims=(0, 0, 0), randomness="different")
                net_theta, dnet_dt = batched_jvp(s_fp32, t_fp32, X_fp32)

                delta = (t_fp32 - s_fp32).view(-1, 1, 1, 1)
                partial_t_X = net_theta + delta * dnet_dt

                derivative_loss = F.mse_loss(partial_t_X, I_dot_fp32)

                # Reconstruction branch: IMPORTANT to backprop through X_ts_It (not detached)
                # Use flow_map again for consistency, but keep gradients.
                X_st_X_ts_It = self.flow_map(s, t, X_ts_It)  # uses current (possibly fp16) path; ok
                reconstruction_loss = F.mse_loss(X_st_X_ts_It.float(), I_t_fp32)

        finally:
            m.train(was_training)

        total_loss = derivative_loss + float(getattr(self, "reconstruction_weight", 0.5)) * reconstruction_loss

        return total_loss, {
            "derivative_loss": float(derivative_loss.detach().item()),
            "reconstruction_loss": float(reconstruction_loss.detach().item()),
        }
    
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
            x = self.flow_map(s, t, x, clamp_x=self.clamp_x)  # Optional clamping for stability during sampling

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

        Only includes _step buffer. Config params (predict_x0, semigroup_weight,
        step_epsilon) are scalar attributes saved separately in the checkpoint's
        method_config dict by train.py.
        """
        state = super().state_dict()
        # _step is already included as a buffer via nn.Module.state_dict()
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
            semigroup_weight=fm_config.get("semigroup_weight", 0.0),
            predict_x0=fm_config.get("predict_x0", False),
            step_epsilon=fm_config.get("step_epsilon", 1e-3),
            clamp_x=fm_config.get("clamp_x", False),
            sg_ramp_steps=fm_config.get("sg_ramp_steps", 50000),
            sg_min_seg=fm_config.get("sg_min_seg", 0.05),
            sg_delay_steps=fm_config.get("sg_delay_steps", 20000),
        )
