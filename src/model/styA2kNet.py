import torch.nn.functional as F
import torch
import torch.nn as nn

from src.model.vgg_extractor import * 
from src.model.attention_fusion import *
from src.model.decoder_net import *

class StyA2KNet(nn.Module):
    def __init__(self, encoder, device="cuda"):
        """
        ARGS:
            encoder: Instancia de VGGEncoder (debe extraer relu3_1 y relu4_1).
        """
        super().__init__()
        self.device = device
        self.vgg_encoder = encoder
        
        # Congelar VGG (Seguridad)
        for param in self.vgg_encoder.parameters():
            param.requires_grad = False
        
        #  DOS MÓDULOS DE ATENCIÓN 
        self.fusion4 = StyA2KAttentionFusion(in_channels=512)
        
        # Uno para el color y textura fina 
        self.fusion3 = StyA2KAttentionFusion(in_channels=256)
        
        #Decoder Multi-Nivel
        self.decoder = StyA2KDecoderMultiLevel()
        
        self.to(device)

    def encode_multi(self, x):
        """Helper para extraer ambos niveles a la vez"""
        features = self.vgg_encoder(x)
        return features["relu4_1"], features["relu3_1"]

    
    def forward(self, x_content, x_style, alpha=1.0):
        
        # Extracción de Features (Multi-Nivel)
        with torch.no_grad():
            # Obtenemos features profundas (c4) y medias (c3)
            c4, c3 = self.encode_multi(x_content)
            s4, s3 = self.encode_multi(x_style)
        
        # Fusión Nivel 4 (Estructura global)
        F_stylized_4 = self.fusion4(c4, s4)
        
        # Fusión Nivel 3 (Color y Detalle)
        F_stylized_3 = self.fusion3(c3, s3)
        
        # Control de Alpha (Interpolación)
        if alpha < 1.0:
            F_stylized_4 = alpha * F_stylized_4 + (1 - alpha) * c4
            F_stylized_3 = alpha * F_stylized_3 + (1 - alpha) * c3
            
        output_image = self.decoder(F_stylized_4, F_stylized_3)
        
        return output_image

