import torch
import torch.nn as nn

from src.model.attention_fusion import StyA2KAttentionFusion
from src.model.decoder_net import StyA2KDecoderMultiLevel
from src.model.vgg_extractor import VGGEncoder, get_vgg_encoder


class StyA2KNet(nn.Module):
    def __init__(self, encoder: VGGEncoder | None = None, device: str = "cuda"):
        """
        Args:
            encoder: Instancia de VGGEncoder (debe extraer relu3_1 y relu4_1).
            device: Target device ('cuda' o 'cpu').
        """
        super().__init__()
        self.device = device
        self.vgg_encoder = encoder or get_vgg_encoder(device)

        # Congelar VGG (Seguridad)
        for param in self.vgg_encoder.parameters():
            param.requires_grad = False

        #  DOS MÓDULOS DE ATENCIÓN 
        self.fusion4 = StyA2KAttentionFusion(in_channels=512)

        # Uno para el color y textura fina 
        self.fusion3 = StyA2KAttentionFusion(in_channels=256)

        # Decoder Multi-Nivel
        self.decoder = StyA2KDecoderMultiLevel()

        self.to(device)

    def encode_multi(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extrae ambos niveles a la vez."""
        features = self.vgg_encoder(x)
        return features["relu4_1"], features["relu3_1"]

    def forward(self, x_content: torch.Tensor, x_style: torch.Tensor, alpha: float = 1.0):
        # Extracción de Features (Multi-Nivel)
        with torch.no_grad():
            c4, c3 = self.encode_multi(x_content)
            s4, s3 = self.encode_multi(x_style)

        # Fusión Nivel 4 (Estructura global)
        fused_4 = self.fusion4(c4, s4)

        # Fusión Nivel 3 (Color y Detalle)
        fused_3 = self.fusion3(c3, s3)

        # Control de Alpha (Interpolación)
        if alpha < 1.0:
            fused_4 = alpha * fused_4 + (1 - alpha) * c4
            fused_3 = alpha * fused_3 + (1 - alpha) * c3

        output_image = self.decoder(fused_4, fused_3)
        return output_image

