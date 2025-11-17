import math
import torch
import torch.nn as nn

class StyA2KAttentionFusion(nn.Module):
    """
    Attention Fusion simplificado tipo all-to-key:
      - Content: F_c [B, C, Hc, Wc]
      - Style:   F_s [B, C, Hs, Ws]
    Usa:
      - Pool sobre estilo -> keys/values en pocas posiciones (key positions).
      - Atención entre queries (content) y keys (style).
      - Fusión residual: out = F_c + proj(attended_style).

    No es la implementación exacta del paper, pero respeta la idea:
      cada posición de content atiende a 'keys' de estilo.
    """
    def __init__(
        self,
        in_channels: int = 512,
        key_dim: int = 128,      
        pool_hw: int = 8,          
        residual: bool = True):
      
        super().__init__()
        self.residual = residual

        # Proyecciones para Q, K, V
        self.query = nn.Conv2d(in_channels, key_dim, kernel_size=1)
        self.key = nn.Conv2d(in_channels, key_dim, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Proyección de salida
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Pool para obtener 'keys' compactos de estilo
        self.style_pool = nn.AdaptiveAvgPool2d((pool_hw, pool_hw))

    def forward(self, F_c: torch.Tensor, F_s: torch.Tensor):
        """
        F_c: [B, C, Hc, Wc]  (content feature)
        F_s: [B, C, Hs, Ws]  (style feature)
        return: F_fused [B, C, Hc, Wc]
        """
        B, C, Hc, Wc = F_c.shape

        # Reducimos estilo para obtener key positions (más estable + más eficiente)
        F_s_pooled = self.style_pool(F_s)          
        _, _, Hk, Wk = F_s_pooled.shape
        Ns = Hk * Wk


        Q = self.query(F_c)                  
        K = self.key(F_s_pooled)                
        V = self.value(F_s_pooled)      

        # Flatten espacial
        Q = Q.view(B, -1, Hc * Wc).transpose(1, 2)
        K = K.view(B, -1, Ns)                        
        V = V.view(B, C, Ns)                      
        Nc = Hc * Wc
        d_k = K.shape[1]

        attn = torch.bmm(Q, K) / math.sqrt(d_k)
        attn = torch.softmax(attn, dim=-1)

        # Mezcla estilo según atención, Multplicar por V
        F_att = torch.bmm(V, attn.transpose(1, 2))  
        F_att = F_att.view(B, C, Hc, Wc)            

        F_att = self.out_proj(F_att)         

        if self.residual:
            return F_c + F_att
        else:
            return F_att