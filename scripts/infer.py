#!/usr/bin/env python3
"""
Inference script for StyA2KNet with optional multi-style fusion.

Example:
    python scripts/infer.py \
        --checkpoint checkpoints/stya2k_e040.pt \
        --content path/to/content.jpg \
        --style path/to/style.jpg \
        --output stylized.png

    # Fusionar varios estilos
    python scripts/infer.py \
        --checkpoint checkpoints/stya2k_e040.pt \
        --content content.jpg \
        --style style_a.jpg --style style_b.jpg \
        --style-weights 0.7 0.3 \
        --output stylized_mix.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.inference.internet_inference import (
    build_inference_transform,
    fuse_styles,
    prepare_tensor_from_source,
)
from src.model.styA2kNet import StyA2KNet
from src.model.vgg_extractor import get_vgg_encoder
from src.training.train_model import denorm_imagenet


def load_model(checkpoint: Path | None, device: torch.device) -> StyA2KNet:
    encoder = get_vgg_encoder(device=device)
    model = StyA2KNet(encoder=encoder, device=str(device)).to(device)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stylize an image pair (or multi-style fusion) with StyA2KNet.")
    parser.add_argument("--checkpoint", type=Path, required=False, help="Path to a checkpoint with trained weights.")
    parser.add_argument("--content", type=str, required=True, help="Path or URL to the content image.")
    parser.add_argument(
        "--style",
        type=str,
        action="append",
        required=True,
        help="Path or URL to a style reference image. Repeat this flag to fuse multiple styles.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to save the stylized output.")
    parser.add_argument("--size", type=int, default=256, help="Target resolution for preprocessing (default: 256).")
    parser.add_argument(
        "--style-weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional weights (same length as number of styles) to control fusion.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Interpolation factor between fused style and content features (1.0 = full style).",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (default: autodetect between CUDA and CPU)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tfm = build_inference_transform(args.size)

    model = load_model(args.checkpoint, device)

    content = prepare_tensor_from_source(args.content, tfm, device)
    style_tensors = [prepare_tensor_from_source(src, tfm, device) for src in args.style]
    style = fuse_styles(style_tensors, weights=args.style_weights)

    with torch.no_grad():
        output = model(content, style, alpha=args.alpha)
    output = denorm_imagenet(torch.sigmoid(output)).clamp(0.0, 1.0).cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(output, str(args.output))
    print(f"Stylized image saved to {args.output}")


if __name__ == "__main__":
    main()
