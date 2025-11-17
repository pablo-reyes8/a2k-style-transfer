import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_data import DualBatchIterator, filter_valid_images  # noqa: E402
from src.model.loss import PerceptualLoss  # noqa: E402


class DummyHFDataset:
    """Minimal subset of the Hugging Face dataset API used in tests."""

    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, fn, num_proc=1):
        filtered = [row for row in self._rows if fn(row)]
        return DummyHFDataset(filtered)

    def __getitem__(self, idx):
        return self._rows[idx]

    @property
    def num_rows(self):
        return len(self._rows)


def test_filter_valid_images_removes_invalid_samples():
    valid_pil = Image.new("RGB", (4, 4), color=128)
    dataset = DummyHFDataset(
        [
            {"image": valid_pil},
            {"image": None},
            {"image": {"path": ""}},
            {"image": {"bytes": b"abc"}},
        ]
    )
    filtered = filter_valid_images(dataset, "image")
    assert filtered.num_rows == 2


class DummyLoader:
    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        for batch in self._batches:
            yield batch

    def __len__(self):
        return len(self._batches)


def test_dual_batch_iterator_aligns_batch_sizes():
    torch.manual_seed(0)
    content_batches = [
        torch.randn(2, 3, 4, 4),
        torch.randn(2, 3, 4, 4),
        torch.randn(1, 3, 4, 4),
    ]
    style_batches = [torch.randn(1, 3, 4, 4)]

    iterator = DualBatchIterator(DummyLoader(content_batches), DummyLoader(style_batches))
    yielded = list(iterator)

    assert len(yielded) == len(content_batches)
    for x_c, x_s in yielded:
        assert x_c.shape[0] == x_s.shape[0]


class IdentityExtractor(torch.nn.Module):
    def forward(self, x):
        return {"relu4_1": x}


@pytest.fixture
def perceptual_loss():
    extractor = IdentityExtractor()
    return PerceptualLoss(
        loss_extractor=extractor,
        content_layers=["relu4_1"],
        style_layers=["relu4_1"],
        content_weights={"relu4_1": 1.0},
        style_weights={"relu4_1": 1.0},
        clamp_pred=False,
    )


def test_perceptual_loss_zero_when_inputs_match(perceptual_loss):
    x = torch.ones(1, 3, 4, 4)
    loss, parts = perceptual_loss(x, x, x)

    assert pytest.approx(0.0, abs=1e-6) == loss.item()
    assert pytest.approx(0.0, abs=1e-6) == parts["content"]
    assert pytest.approx(0.0, abs=1e-6) == parts["style"]


def test_perceptual_loss_detects_content_and_style_shift(perceptual_loss):
    pred = torch.zeros(1, 3, 4, 4)
    content = torch.ones(1, 3, 4, 4)
    style = torch.ones(1, 3, 4, 4) * 2.0

    _, parts = perceptual_loss(pred, content, style)

    assert parts["content"] > 0.0
    assert parts["style"] > 0.0
