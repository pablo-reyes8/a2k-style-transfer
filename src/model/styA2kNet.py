import torch.nn.functional as F
import torch
import torch.nn as nn

from src.model.vgg_extractor import * 
from src.model.attention_fusion import *
from src.model.decoder_net import *

class StyA2KNet(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        # Encoder compartido (
        self.vgg_enc = build_vgg_loss_extractor(
            device=device,
            content_layers=["relu4_1"],
            style_layers=["relu4_1"],   )

        # Attention fusion a nivel relu4_1
        self.fusion = StyA2KAttentionFusion(
            in_channels=512,
            key_dim=128,
            pool_hw=8,
            residual=True,)

        # Decoder
        self.decoder = StyA2KDecoder(out_size=(252, 252))

    def encode_relu4_1(self, x: torch.Tensor):
        feats = self.vgg_enc(x)
        return feats["relu4_1"]


    def forward(self, x_content: torch.Tensor, x_style: torch.Tensor):
        """
        x_content, x_style: [B,3,252,252] normalizados ImageNet
        return: y estilizada [B,3,252,252]
        """
        F_c = self.encode_relu4_1(x_content)  # [B,512,31,31]
        F_s = self.encode_relu4_1(x_style)    # [B,512,31,31]

        F_cs = self.fusion(F_c, F_s)          # [B,512,31,31]
        y = self.decoder(F_cs)                # [B,3,252,252]
        return y
