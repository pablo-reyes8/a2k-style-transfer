
import torch
import torch.nn as nn
from typing import Dict, Tuple, Iterable, List
from src.model.vgg_extractor import *


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    feat: [B,C,H,W] -> Gram por batch, normalizado por (H*W) (no por C*H*W)
    FP32 para estabilidad en FP16/BF16.
    """
    B, C, H, W = feat.shape
    F = feat.float().view(B, C, H * W)              # FP32
    G = torch.bmm(F, F.transpose(1, 2))             # [B,C,C] FP32
    G = G / (H * W)
    return G


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """
    TV loss simple: suaviza diferencias entre píxeles vecinos.
    """
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).mean()
    return dh + dw


class PerceptualLoss(nn.Module):
    """
    Perceptual + Style Loss estable:
      - content: L2 entre activaciones (FP32 en pred)
      - style:  L2 entre Gram matrices (FP32)
      - tv:     Total Variation sobre la predicción
    """
    def __init__(self,
                 loss_extractor: VGGFeatureExtractor,
                 content_layers: List[str] = None,
                 style_layers: List[str] = None,
                 content_weights: Dict[str,float] = None,
                 style_weights: Dict[str,float] = None,
                 clamp_pred: bool = True,
                 w_tv: float = 1e-6):
        super().__init__()
        self.extractor = loss_extractor.eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        self.content_layers = content_layers or DEFAULT_CONTENT_LAYERS
        self.style_layers   = style_layers   or DEFAULT_STYLE_LAYERS
        self.content_w = content_weights or {l: 1.0 for l in self.content_layers}
        self.style_w   = style_weights   or {l: 1.0 for l in self.style_layers}
        self.l2 = nn.MSELoss(reduction="mean")
        self.clamp_pred = clamp_pred
        self.w_tv = float(w_tv)

        # buffers de normalización ImageNet para VGG
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("vgg_mean", mean)
        self.register_buffer("vgg_std", std)

    def forward(self, pred: torch.Tensor,
                content: torch.Tensor,
                style: torch.Tensor):

        with torch.no_grad():
            f_c_true = self.extractor(content)
            f_s_true = self.extractor(style)

        # Predicción: clamp en [0,1] para imagen y normalización tipo VGG
        if self.clamp_pred:
            pred_img = pred.clamp(0.0, 1.0)
        else:
            pred_img = pred

        # normalizar como VGG: (x - mean)/std
        mean = self.vgg_mean.to(pred_img.device, dtype=pred_img.dtype)
        std  = self.vgg_std.to(pred_img.device, dtype=pred_img.dtype)
        pred_vgg = (pred_img - mean) / std

        with torch.cuda.amp.autocast(enabled=False):
            f_pred = self.extractor(pred_vgg.float())

        # Content loss
        lc = 0.0
        for l in self.content_layers:
            lc = lc + self.content_w[l] * self.l2(f_pred[l], f_c_true[l])

        #  Style loss (Gram matrices)
        ls = 0.0
        for l in self.style_layers:
            Gp = gram_matrix(f_pred[l])
            Gs = gram_matrix(f_s_true[l])
            ls = ls + self.style_w[l] * self.l2(Gp, Gs)

        #  Total Variation sobre la imagen predicha (en espacio [0,1])
        ltv = self.w_tv * total_variation_loss(pred_img)

        total = lc + ls + ltv
        return total, {
            "content": float(lc.detach()),
            "style":   float(ls.detach()),
            "tv":      float(ltv.detach())}
