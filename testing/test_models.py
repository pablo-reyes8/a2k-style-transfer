import sys
from pathlib import Path

import pytest
import torch

# Ensure the project root is on the import path when tests run from /testing
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.attention_fusion import StyA2KAttentionFusion
from src.model.decoder_net import StyA2KDecoder
from src.model.styA2kNet import StyA2KNet


@pytest.fixture
def dummy_loss_extractor(monkeypatch):
    """Patch StyA2KNet to avoid downloading VGG19 weights during tests."""

    class DummyExtractor(torch.nn.Module):
        def forward(self, x: torch.Tensor):
            b = x.shape[0]
            # Deterministic pseudo features with the expected VGG shape
            features = torch.ones(b, 512, 31, 31, device=x.device, dtype=x.dtype)
            return {"relu4_1": features}

    def _builder(*args, **kwargs):
        return DummyExtractor()

    monkeypatch.setattr(
        "src.model.styA2kNet.build_vgg_loss_extractor",
        _builder,
        raising=True,
    )


def test_attention_fusion_preserves_spatial_shape():
    fusion = StyA2KAttentionFusion(in_channels=512, key_dim=64, pool_hw=4, residual=True)
    content = torch.randn(2, 512, 31, 31)
    style = torch.randn(2, 512, 31, 31)

    fused = fusion(content, style)

    assert fused.shape == content.shape
    assert torch.isfinite(fused).all()


def test_attention_fusion_without_residual_has_no_passthrough():
    fusion = StyA2KAttentionFusion(in_channels=256, key_dim=32, pool_hw=2, residual=False)
    content = torch.zeros(1, 256, 15, 15)
    style = torch.zeros(1, 256, 15, 15)

    fused = fusion(content, style)

    assert torch.allclose(fused, torch.zeros_like(fused))


def test_decoder_outputs_rgb_image_with_target_size():
    decoder = StyA2KDecoder(out_size=(252, 252))
    features = torch.randn(3, 512, 31, 31)

    output = decoder(features)

    assert output.shape == (3, 3, 252, 252)
    assert torch.isfinite(output).all()


def test_stya2knet_forward_runs_end_to_end(dummy_loss_extractor):
    model = StyA2KNet(device="cpu")
    content = torch.randn(2, 3, 252, 252)
    style = torch.randn(2, 3, 252, 252)

    stylized = model(content, style)

    assert stylized.shape == content.shape
    assert torch.isfinite(stylized).all()
