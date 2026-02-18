"""
Dual-Time UNet Wrapper for Flow Map Matching

This module provides a wrapper around the standard UNet to handle
two time inputs (s, t) required by Flow Map Matching, instead of
a single time input used by standard diffusion models.

The wrapper creates a second time embedding network and combines
the embeddings for source time s and target time t before passing
through the UNet.
"""

import torch
import torch.nn as nn
from .blocks import TimestepEmbedding


class DualTimeUNet(nn.Module):
    """
    Wrapper to adapt UNet for dual-time input required by Flow Map Matching.

    Flow Map Matching learns transformations Φ(s, t, x) between two time points,
    requiring the network to condition on both source time s and target time t.

    This wrapper:
    1. Creates a second time embedding network for source time s
    2. Combines s and t embeddings via addition
    3. Passes the combined embedding through the base UNet

    Args:
        base_unet: The underlying UNet model
        num_timesteps: Total number of discrete timesteps (for scaling continuous times)
    """

    def __init__(self, base_unet: nn.Module, num_timesteps: int = 1000):
        super().__init__()
        self.unet = base_unet
        self.num_timesteps = num_timesteps

        # Get time embedding dimension from the base UNet
        time_embed_dim = base_unet.time_embed_dim

        # Create second time embedding network for source time s
        # This matches the architecture of the original time embedding
        self.time_s_embedding = TimestepEmbedding(
            time_embed_dim=time_embed_dim,
            hidden_dim=time_embed_dim  # Match the original embedding size
        )

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with two time inputs.

        Args:
            x: Input tensor (batch_size, channels, H, W)
            s: Source time (batch_size,) - continuous values in [0, 1]
            t: Target time (batch_size,) - continuous values in [0, 1]

        Returns:
            Network output φ(s, t, x) of shape (batch_size, channels, H, W)
        """
        # Scale continuous times [0, 1] to [0, num_timesteps] for embeddings
        # The sinusoidal embedding works better with larger values
        s_scaled = s * self.num_timesteps
        t_scaled = t * self.num_timesteps

        # Generate individual time embeddings
        s_emb = self.time_s_embedding(s_scaled)
        t_emb = self.unet.time_embedding(t_scaled)

        # Combine embeddings via addition
        # This is a simple and effective approach used in multi-conditioning
        # Alternative: concatenate and project, but addition works well
        combined_time_emb = s_emb + t_emb

        # Pass through UNet with combined time embedding
        # The UNet.forward() has been modified to accept optional time_emb
        return self.unet.forward(x, t_scaled, time_emb=combined_time_emb)

    # Note: No need to override parameters(), state_dict(), or load_state_dict()
    # PyTorch automatically handles submodules (self.unet and self.time_s_embedding)
    # correctly for all these methods. Custom overrides can break EMA, DDP, and
    # checkpoint compatibility.
