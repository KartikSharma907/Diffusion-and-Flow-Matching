"""
Models module for cmu-10799-diffusion.

This module contains the neural network architectures used for
diffusion models and flow matching.
"""

from .unet import UNet, create_model_from_config
from .dual_time_unet import DualTimeUNet
from .blocks import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)

__all__ = [
    # Main model
    'UNet',
    'DualTimeUNet',
    'create_model_from_config',
    # Building blocks
    'SinusoidalPositionalEmbedding',
    'TimestepEmbedding',
    'ResBlock',
    'AttentionBlock',
    'Downsample',
    'Upsample',
    'GroupNorm32',
]
