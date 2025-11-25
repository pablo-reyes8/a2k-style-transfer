import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Ensure the project root is on the import path when tests run from /testing
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.attention_fusion import StyA2KAttentionFusion  # noqa: E402
from src.model.decoder_net import StyA2KDecoderMultiLevel  # noqa: E402
from src.model.styA2kNet import StyA2KNet  # noqa: E402


class DummyEncoder(nn.Module):
    """Lightweight replacement for VGG during tests (no download needed)."""

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        device = x.device
        dtype = x.dtype
        return {
            "relu4_1": torch.ones(b, 512, 32, 32, device=device, dtype=dtype),
            "relu3_1": torch.ones(b, 256, 64, 64, device=device, dtype=dtype),}


def test_attention_fusion_preserves_spatial_shape():
    fusion = StyA2KAttentionFusion(in_channels=512)
    content = torch.randn(2, 512, 32, 32)
    style = torch.randn(2, 512, 32, 32)

    fused = fusion(content, style)

    assert fused.shape == content.shape
    assert torch.isfinite(fused).all()


def test_decoder_outputs_rgb_image_with_target_size():
    decoder = StyA2KDecoderMultiLevel()
    fused4 = torch.randn(3, 512, 32, 32)
    fused3 = torch.randn(3, 256, 64, 64)

    output = decoder(fused4, fused3)

    assert output.shape == (3, 3, 256, 256)
    assert torch.isfinite(output).all()


def test_stya2knet_forward_runs_end_to_end():
    model = StyA2KNet(encoder=DummyEncoder(), device="cpu")
    content = torch.randn(2, 3, 256, 256)
    style = torch.randn(2, 3, 256, 256)

    stylized = model(content, style)

    assert stylized.shape == content.shape
    assert torch.isfinite(stylized).all()
