"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep
    
    Encoder (Downsampling path)

    Middle
    
    Decoder (Upsampling path)
    
    Output: (batch_size, channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks
    
    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3, 
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # Pro tips: remember to take care of the time embeddings!
        # Time embeddings
        # `time_embed_dim` is the embedding size used throughout ResBlocks.
        self.time_embed_dim = base_channels * 4
        # Ensure TimestepEmbedding produces vectors of size `time_embed_dim`.
        self.time_embedding = TimestepEmbedding(self.time_embed_dim)
        
        # input conv
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels: List[int] = []

        ch = base_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResBlock(
                        in_channels=ch,
                        out_channels=out_ch,
                        time_embed_dim=self.time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = out_ch
                
                blocks.append(AttentionBlock(ch, num_heads=num_heads))  # conditionally used

                self.skip_channels.append(ch)

            self.down_blocks.append(blocks)

            # Downsample between levels (except last)
            if level != len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
                ds *= 2
            else:
                self.downsamples.append(nn.Identity())
        
        self.mid_block1 = ResBlock(
            in_channels=ch,
            out_channels=ch,
            time_embed_dim=self.time_embed_dim,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        self.mid_attn = AttentionBlock(ch, num_heads=num_heads)
        self.mid_block2 = ResBlock(
            in_channels=ch,
            out_channels=ch,
            time_embed_dim=self.time_embed_dim,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # We will pop skip connections in reverse order during forward
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult

            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = self.skip_channels.pop()
                blocks.append(
                    ResBlock(
                        in_channels=ch + skip_ch,
                        out_channels=out_ch,
                        time_embed_dim=self.time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                ch = out_ch

                blocks.append(AttentionBlock(ch, num_heads=num_heads))  # conditionally used

            self.up_blocks.append(blocks)

            # Upsample between levels (except last in reverse -> which is first level)
            if level != 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        # Output head (name matches `forward()` usage)
        self.output_norm = GroupNorm32(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
    
    def _maybe_attend(self, h: torch.Tensor, attn_block: nn.Module) -> torch.Tensor:
        # Apply attention iff current spatial resolution is in attention_resolutions
        # (assumes square, but checks H anyway)
        _, _, H, W = h.shape
        if H in self.attention_resolutions and W in self.attention_resolutions:
            return attn_block(h)
        return h 
        
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
               This is typically the noisy image x_t
            t: Timestep tensor of shape (batch_size,)
            time_emb: Optional precomputed time embedding of shape (batch_size, time_embed_dim)
                     If provided, this will be used instead of computing embedding from t.
                     This allows dual-time conditioning for Flow Map Matching.

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # print("Input shape to U-Net: ", x.shape)
        # Use precomputed time embedding if provided, otherwise compute from t
        t_emd = time_emb if time_emb is not None else self.time_embedding(t)
        # print("Time embedding shape: ", t_emd.shape)
        
        h = self.input_conv(x)
        # print("After input conv: ", h.shape)
        
        skips = []
        
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, t_emd)
                    # print(f"Down ResBlock (level {level}): ", h.shape)
                    # Save a skip per ResBlock to match how up_blocks consumes skips
                    skips.append(h)
                elif isinstance(block, AttentionBlock):
                    h = self._maybe_attend(h, block)
                    # print(f"Down AttentionBlock (level {level}): ", h.shape)
            # Downsample once per level (after all blocks)
            h = self.downsamples[level](h)
            # print(f"After Downsample (level {level}): ", h.shape)
        
        h = self.mid_block1(h, t_emd)
        # print("Mid ResBlock 1: ", h.shape)
        h = self.mid_attn(h)
        # print("Mid AttentionBlock: ", h.shape)
        h = self.mid_block2(h, t_emd)
        # print("Mid ResBlock 2: ", h.shape)

        for level, blocks in enumerate(self.up_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = block(h, t_emd)
                    # print(f"Up ResBlock (level {level}): ", h.shape)
                elif isinstance(block, AttentionBlock):
                    h = self._maybe_attend(h, block)
                    # print(f"Up AttentionBlock (level {level}): ", h.shape)
            h = self.upsamples[level](h)
            # print(f"After Upsample (level {level}): ", h.shape)
        
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        # print("Output shape from U-Net: ", h.shape)
        return h


def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters
    
    Returns:
        Instantiated UNet model
    """
    model_config = config['model']
    data_config = config['data']
    
    return UNet(
        in_channels=data_config['channels'],
        out_channels=data_config['channels'],
        base_channels=model_config['base_channels'],
        channel_mult=tuple(model_config['channel_mult']),
        num_res_blocks=model_config['num_res_blocks'],
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        use_scale_shift_norm=model_config['use_scale_shift_norm'],
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
