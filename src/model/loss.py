
import torch
import torch.nn as nn
from typing import Dict, Tuple, Iterable, List
from src.model.vgg_extractor import *

DEFAULT_CONTENT_LAYERS = ["relu4_1"]
DEFAULT_STYLE_LAYERS   = ["relu1_1","relu2_1","relu3_1","relu4_1","relu5_1"]


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


class PerceptualLoss(nn.Module):
    """
    Perceptual + Style Loss estable:
      - content: L2 entre activaciones (FP32 en pred)
      - style:  L2 entre Gram matrices (FP32)
    """
    def __init__(self,
                 loss_extractor: VGGFeatureExtractor,
                 content_layers: List[str] = None,
                 style_layers: List[str] = None,
                 content_weights: Dict[str,float] = None,
                 style_weights: Dict[str,float] = None,
                 clamp_pred: bool = True):
        super().__init__()
        self.extractor = loss_extractor.eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        self.content_layers = content_layers or DEFAULT_CONTENT_LAYERS
        self.style_layers   = style_layers  or DEFAULT_STYLE_LAYERS
        
        self.content_w = content_weights or {l: 1.0 for l in self.content_layers}
        self.style_w   = style_weights   or {l: 1.0 for l in self.style_layers}
        self.l2 = nn.MSELoss(reduction="mean")
        self.clamp_pred = clamp_pred

    def forward(self, pred: torch.Tensor, content: torch.Tensor, style: torch.Tensor):
        # Targets sin gradiente
        with torch.no_grad():
            f_c_true = self.extractor(content)  
            f_s_true = self.extractor(style)

        # Pred
        if self.clamp_pred:
            pred = pred.clamp(0.0, 1.0)

        # El extractor de pérdidas para 'pred' va en FP32
        with torch.cuda.amp.autocast(enabled=False):
            f_pred = self.extractor(pred.float())

        # Content
        lc = 0.0
        for l in self.content_layers:
            lc = lc + self.content_w[l] * self.l2(f_pred[l], f_c_true[l])

        # Style 
        ls = 0.0
        for l in self.style_layers:
            Gp = gram_matrix(f_pred[l])
            Gs = gram_matrix(f_s_true[l])
            ls = ls + self.style_w[l] * self.l2(Gp, Gs)

        total = lc + ls
        return total, {"content": float(lc.detach()), "style": float(ls.detach())}