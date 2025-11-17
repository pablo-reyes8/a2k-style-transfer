#!/usr/bin/env python3
"""
Simple inference script for StyA2KNet.

Example:
    python scripts/infer.py \
        --checkpoint checkpoints/stya2k_e040.pt \
        --content path/to/content.jpg \
        --style path/to/style.jpg \
        --output stylized.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.model.styA2kNet import StyA2KNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_preprocess(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),])


def load_image(path: Path, preprocess: transforms.Compose, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    return tensor.to(device)


def denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return x * std + mean


def load_model(checkpoint: Path | None, device: torch.device) -> StyA2KNet:
    model = StyA2KNet(device=str(device)).to(device)
    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stylize an image pair with StyA2KNet.")
    parser.add_argument("--checkpoint", type=Path, required=False, help="Path to a checkpoint with trained weights.")
    parser.add_argument("--content", type=Path, required=True, help="Path to the content image.")
    parser.add_argument("--style", type=Path, required=True, help="Path to the style reference image.")
    parser.add_argument("--output", type=Path, required=True, help="Path to save the stylized output.")
    parser.add_argument("--size", type=int, default=252, help="Target resolution for preprocessing (default: 252).")
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (default: autodetect between CUDA and CPU).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    preprocess = build_preprocess(args.size)
    content = load_image(args.content, preprocess, device)
    style = load_image(args.style, preprocess, device)

    model = load_model(args.checkpoint, device)

    with torch.no_grad():
        output = model(content, style)
    output = denorm_imagenet(output).clamp(0.0, 1.0).cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(output, str(args.output))
    print(f"Stylized image saved to {args.output}")


if __name__ == "__main__":
    main()
