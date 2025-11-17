#!/usr/bin/env python3
"""
Example training script for StyA2KNet.

This mirrors the notebook workflow but can be executed headlessly:
    python scripts/train.py --config configs/stya2k_base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch.optim import Adam
from torchvision import transforms
from datasets import load_dataset  # type: ignore
import yaml

from src.data.load_data import (
    cast_to_image,
    detect_image_col,
    filter_valid_images,
    HFDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    make_loader,
    make_train_iterator,
    set_seed,)

from src.model.loss import PerceptualLoss, build_vgg_loss_extractor
from src.model.styA2kNet import StyA2KNet
from src.training.chekpoint import save_checkpoint
from src.training.train_model import train_stya2k


def _ensure_tuple(value: Iterable[float] | None, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    seq = tuple(float(v) for v in value)
    if len(seq) != 2:
        raise ValueError(f"Expected 2 values for scale, got {seq}")
    return seq  # type: ignore[return-value]


def build_transform(size: int, scale: Iterable[float], flip_prob: float) -> transforms.Compose:
    scale_tuple = _ensure_tuple(scale, (0.5, 1.0))
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=scale_tuple),
            transforms.RandomHorizontalFlip(p=float(flip_prob)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),])


def _trim_dataset(ds, max_images: int | None):
    if not max_images:
        return ds
    max_images = min(max_images, ds.num_rows)
    return ds.select(range(max_images))


def build_dataset(cfg: Dict[str, Any], transform: transforms.Compose) -> HFDataset:
    if "hub_id" not in cfg:
        raise ValueError("Dataset config must include 'hub_id'")

    ds = load_dataset(cfg["hub_id"], cfg.get("config"), split=cfg.get("split", "train"))
    image_col = detect_image_col(ds)
    ds = cast_to_image(filter_valid_images(ds, image_col), image_col)
    ds = _trim_dataset(ds, cfg.get("max_images"))
    return HFDataset(ds, img_key="image", transform=transform)


def build_loader(cfg: Dict[str, Any], transform: transforms.Compose, data_cfg: Dict[str, Any]):
    dataset = build_dataset(cfg, transform)
    return make_loader(
        dataset,
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
        pin=data_cfg.get("pin_memory", True),
        drop_last=data_cfg.get("drop_last", True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StyA2KNet from the command line.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stya2k_base.yaml"),
        help="YAML config describing experiment/data/training hyper-parameters.",)
    
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="Optional path to store the final checkpoint after training.",)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    with args.config.open("r", encoding="utf-8") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    experiment_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})
    optimizer_cfg = cfg.get("optimizer", {})
    training_cfg = cfg.get("training", {})

    set_seed(experiment_cfg.get("seed", 7))

    if "content" not in data_cfg or "style" not in data_cfg:
        raise ValueError("Data config must include 'content' and 'style' sections.")

    target_size = data_cfg.get("final_size", 252)
    content_cfg = data_cfg.get("content", {})
    style_cfg = data_cfg.get("style", {})

    content_tf = build_transform(
        size=target_size,
        scale=content_cfg.get("random_resized_crop_scale", (0.5, 1.0)),
        flip_prob=content_cfg.get("horizontal_flip", 0.5),)
    
    style_tf = build_transform(
        size=target_size,
        scale=style_cfg.get("random_resized_crop_scale", (0.7, 1.0)),
        flip_prob=style_cfg.get("horizontal_flip", 0.2),)


    logging.info("Loading datasets from Hugging Face Hub...")
    content_loader = build_loader(content_cfg, content_tf, data_cfg)
    style_loader = build_loader(style_cfg, style_tf, data_cfg)

    device = cfg.get("model", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model = StyA2KNet(device=device).to(device)
    loss_extractor = build_vgg_loss_extractor(device=device)
    criterion = PerceptualLoss(loss_extractor)

    optimizer = Adam(
        model.parameters(),
        lr=optimizer_cfg.get("lr", 1e-4),
        betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
        weight_decay=optimizer_cfg.get("weight_decay", 0.0),)

    training_state = train_stya2k(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=training_cfg.get("epochs", 40),
        amp_enabled=training_cfg.get("amp_enabled", True),
        amp_dtype=training_cfg.get("amp_dtype", "bf16"),
        grad_clip=training_cfg.get("grad_clip"),
        log_every=training_cfg.get("log_every", 50),
        run_name=experiment_cfg.get("name", "StyA2KNet"),
        sample_every=training_cfg.get("sample_every", 1),
        sample_dir=experiment_cfg.get("sample_dir", "samples_stya2k"),
        content_loader=content_loader,
        style_loader=style_loader)

    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            path=str(args.checkpoint_out),
            model=model,
            optimizer=optimizer,
            epoch=training_state.get("last_epoch", training_cfg.get("epochs", 40)),
            global_step=training_state.get("global_step", 0),
            extra={"config": str(args.config)},)
        
        logging.info("Checkpoint stored at %s", args.checkpoint_out)


if __name__ == "__main__":
    main()
