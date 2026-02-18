"""
Sampling Script for DDPM (Denoising Diffusion Probabilistic Models)

Generate samples from a trained model. By default, saves individual images to avoid
memory issues with large sample counts. Use --grid to generate a single grid image.

Usage:
    # Sample from DDPM (saves individual images to ./samples/)
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64

    # With custom number of sampling steps
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_steps 500

    # Generate a grid image instead of individual images
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64 --grid

    # Save individual images to custom directory
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --output_dir my_samples

What you need to implement:
- Incorporate your sampling scheme to this pipeline
- Save generated samples as images for logging
"""

import os
import sys
import argparse
from datetime import datetime

import yaml
import torch
from tqdm import tqdm

from src.models import create_model_from_config, DualTimeUNet
from src.data import save_image, unnormalize
from src.methods import DDPM, FlowMatching, FlowMapMatching
from src.utils import EMA

def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def strip_module_prefix_recursive(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            nk = k[7:] if k.startswith("module.") else k
            out[nk] = strip_module_prefix_recursive(v)
        return out
    return x
def unwrap_state_dict(maybe_sd):
    """
    Handle common nesting patterns:
      - {"state_dict": {...}}
      - {"model": {...}} where model is already a state_dict
      - {"model": {"state_dict": {...}}}
      - {"ema": {...}} etc.
    """
    if isinstance(maybe_sd, dict):
        # common wrappers
        if "state_dict" in maybe_sd and isinstance(maybe_sd["state_dict"], dict):
            return maybe_sd["state_dict"]
        if "model" in maybe_sd and isinstance(maybe_sd["model"], dict) and any(
            k in maybe_sd["model"] for k in ["state_dict", "model"]
        ):
            return unwrap_state_dict(maybe_sd["model"])
    return maybe_sd

def load_checkpoint(checkpoint_path: str, device: torch.device, method_name: str = None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # ---- Build model first (must match what was trained) ----
    base_model = create_model_from_config(config).to(device)
    if method_name == "flow_map_matching":
        model = DualTimeUNet(
            base_model,
            num_timesteps=config.get("flow_map_matching", {}).get("num_timesteps", 1000),
        ).to(device)
    else:
        model = base_model

    # ---- Extract + strip model state dict ----
    raw_model_sd = None
    if "model" in checkpoint:
        raw_model_sd = checkpoint["model"]
    elif "state_dict" in checkpoint:
        raw_model_sd = checkpoint["state_dict"]
    else:
        # sometimes checkpoint itself is the sd
        raw_model_sd = checkpoint

    raw_model_sd = unwrap_state_dict(raw_model_sd)
    model_sd = strip_module_prefix_recursive(raw_model_sd)

    # Load with diagnostics
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    print(f"[load] model missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("  first missing:", missing[:10])
    if len(unexpected) > 0:
        print("  first unexpected:", unexpected[:10])

    # If there are still unexpected keys with module., strip again (rare double nesting)
    if any(k.startswith("module.") for k in unexpected):
        model_sd = strip_module_prefix(model_sd)
        missing, unexpected = model.load_state_dict(model_sd, strict=False)
        print(f"[load retry] model missing={len(missing)} unexpected={len(unexpected)}")

    # ---- EMA ----
    ema = EMA(model, decay=config["training"]["ema_decay"])
    if "ema" in checkpoint:
        raw_ema_sd = unwrap_state_dict(checkpoint["ema"])
        ema_sd = strip_module_prefix_recursive(raw_ema_sd)
        # IMPORTANT: strip module prefix here too
        ema.load_state_dict(ema_sd)
    else:
        print("[load] no EMA in checkpoint")

    # ---- Method ----
    if method_name in ("ddpm", "ddim"):
        method = DDPM.from_config(model, config, device)
    elif method_name == "flow_matching":
        method = FlowMatching.from_config(model, config, device)
    elif method_name == "flow_map_matching":
        method = FlowMapMatching.from_config(model, config, device)
    else:
        method = None

    if method is not None and "method" in checkpoint:
        # method state dict may also have module. prefix depending on save logic
        method_sd = strip_module_prefix_recursive(unwrap_state_dict(checkpoint["method"]))
        method.load_state_dict(method_sd)

    return model, config, ema, method

# def load_checkpoint(checkpoint_path: str, device: torch.device, method_name: str = None):
#     """
#     Load checkpoint and return model, config, and EMA.

#     For flow_map_matching, properly wraps in DualTimeUNet before loading EMA.

#     Args:
#         checkpoint_path: Path to checkpoint file
#         device: Device to load to
#         method_name: Method name ('flow_map_matching' requires special handling)

#     Returns:
#         model: Loaded model (wrapped in DualTimeUNet for flow_map_matching)
#         config: Config dict
#         ema: EMA instance
#         method: Method instance (for loading method state)
#     """
#     checkpoint = torch.load(checkpoint_path, map_location=device)

#     sd = checkpoint["model"] if "model" in checkpoint else checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
#     sd = strip_module_prefix(sd)
    
#     config = checkpoint['config']

#     # Create base model
#     base_model = create_model_from_config(config).to(device)

#     # For flow_map_matching: wrap in DualTimeUNet BEFORE loading state
#     # This ensures EMA covers all parameters including time_s_embedding
#     if method_name == 'flow_map_matching':
#         model = DualTimeUNet(base_model, num_timesteps=config.get('flow_map_matching', {}).get('num_timesteps', 1000)).to(device)
#     else:
#         model = base_model

#     # Load model state
#     model.load_state_dict(sd)

#     # Create EMA from the (possibly wrapped) model and load
#     ema = EMA(model, decay=config['training']['ema_decay'])
#     if 'ema' in checkpoint:
#         ema.load_state_dict(checkpoint['ema'])

#     # Create method instance
#     if method_name == 'ddpm' or method_name == 'ddim':
#         method = DDPM.from_config(model, config, device)
#     elif method_name == 'flow_matching':
#         method = FlowMatching.from_config(model, config, device)
#     elif method_name == 'flow_map_matching':
#         method = FlowMapMatching.from_config(model, config, device)
#     else:
#         method = None

#     # Load method state if available
#     if method is not None and 'method' in checkpoint:
#         method.load_state_dict(checkpoint['method'])

#     return model, config, ema, method


def save_samples(
    samples: torch.Tensor,
    save_path: str,
    num_samples: int = None,
    nrow: int = 8,
    **kwargs,
) -> None:
    """
    Args:
        samples: Generated samples tensor with shape (num_samples, C, H, W).
        save_path: File path to save the image grid.
        num_samples: Number of samples, used to calculate grid layout.
    """

    # samples: (B, C, H, W) or (C, H, W)
    if samples.dim() == 3:
        samples = samples.unsqueeze(0)

    # Optionally trim to requested number of samples
    if num_samples is not None:
        samples = samples[:num_samples]

    # # If samples are in [-1, 1], convert to [0, 1] for saving
    # # Heuristic: check min/max
    # if samples.min() >= -1.0 and samples.max() <= 1.0:
    #     images = unnormalize(samples)
    # else:
    #     images = samples
    
    # Model/data is in [-1, 1]. Always convert to [0, 1] for saving.
    images = unnormalize(samples).clamp(0.0, 1.0)

    # Move to CPU and ensure float
    images = images.detach().cpu().float()

    # If saving a single image, set nrow=1
    if images.shape[0] == 1:
        save_image(images, save_path, nrow=1, **kwargs)
    else:
        save_image(images, save_path, nrow=nrow, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--method', type=str, required=True,
                       choices=['ddpm', 'ddim', 'flow_matching', 'flow_map_matching'],
                       help='Method used for training: ddpm, ddim, flow_matching, or flow_map_matching')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='samples',
                       help='Directory to save individual images (default: samples)')
    parser.add_argument('--grid', action='store_true',
                       help='Save as grid image instead of individual images')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for grid (only used with --grid, default: samples_<timestamp>.png)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Sampling arguments
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of sampling steps (default: from config)')
    
    # Other options
    parser.add_argument('--no_ema', action='store_true',
                       help='Use training weights instead of EMA weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--ddim_num_steps', type=int, default=100,
                       help='Number of steps for DDIM sampling (only used if --method is ddim, default: 100)')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                    help='DDIM eta (0=deterministic')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Load checkpoint (method_name passed for proper DualTimeUNet wrapping)
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, ema, method = load_checkpoint(args.checkpoint, device, args.method)

    if method is None:
        raise ValueError(f"Unknown method: {args.method}. Supported: 'ddpm', 'ddim', 'flow_matching', 'flow_map_matching'.")

    # Apply EMA weights
    if not args.no_ema:
        print("Using EMA weights")
        ema.apply_shadow()
    else:
        print("Using training weights (no EMA)")
    
    method.eval_mode()
    
    # Image shape
    data_config = config['data']
    image_shape = (data_config['channels'], data_config['image_size'], data_config['image_size'])
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")

    all_samples = []
    remaining = args.num_samples
    sample_idx = 0

    # Create output directory if saving individual images
    if not args.grid:
        os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating samples")
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)

            if args.method == 'ddim':
                num_steps = args.ddim_num_steps
            else:
                num_steps = args.num_steps or config['sampling']['num_steps']

            samples = method.sample(
                batch_size=batch_size,
                image_shape=image_shape,
                num_steps=num_steps,
                ddim_enabled=(args.method == 'ddim'),
                eta=(args.ddim_eta if args.method == 'ddim' else 0.0),
            )

            # Save individual images immediately or collect for grid
            if args.grid:
                all_samples.append(samples)
            else:
                for i in range(samples.shape[0]):
                    img_path = os.path.join(args.output_dir, f"{sample_idx:06d}.png")
                    save_samples(samples[i:i+1], img_path, 1)
                    sample_idx += 1

            remaining -= batch_size
            pbar.update(batch_size)

        pbar.close()

    # Save samples
    if args.grid:
        # Concatenate all samples for grid
        all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]

        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"samples_{timestamp}.png"

        save_samples(all_samples, args.output, nrow=8)
        print(f"Saved grid to {args.output}")
    else:
        print(f"Saved {args.num_samples} individual images to {args.output_dir}")

    # Restore EMA if applied
    if not args.no_ema:
        ema.restore()


if __name__ == '__main__':
    main()
