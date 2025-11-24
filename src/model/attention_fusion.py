import torch
import torch.nn as nn
import math 

class StyA2KAttentionFusion(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.in_channels = in_channels
        
        # BLOQUE DE ATENCIÓN 
        self.f_query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.f_key   = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.f_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        
        # FUSION LAYER 
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels, affine=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1))

    def mean_variance_norm(self, feat):
        size = feat.size()
        mean, std = self.calc_mean_std(feat)
        normalized_feat = (feat - mean.expand(size)) / (std.expand(size) + 1e-5)
        return normalized_feat

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, content_feat, style_feat):
        B, C, H, W = content_feat.size()
        
        # Normalización (Para calcular similitud semántica, no magnitud)
        norm_content = self.mean_variance_norm(content_feat)
        norm_style   = self.mean_variance_norm(style_feat)
        
        # Proyecciones
        query = self.f_query(norm_content).view(B, C, -1).permute(0, 2, 1) # (B, HW, C)
        key   = self.f_key(norm_style).view(B, C, -1)                      # (B, C, HW)
        
        # Escalamos por la raíz cuadrada de C para evitar saturar el Softmax
        energy = torch.bmm(query, key) / math.sqrt(C) 
        attention = self.softmax(energy) 
        
        # Values (Usamos features originales para mantener textura real)
        value = self.f_value(style_feat).view(B, C, -1) 
        
        # Aplicar atención
        attended_style = torch.bmm(value, attention.permute(0, 2, 1)).view(B, C, H, W)
        
        # Fusión
        combined = torch.cat([content_feat, attended_style], dim=1)
        out = self.fusion_conv(combined)
        
        return out + content_feat
