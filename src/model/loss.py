
import torch
import torch.nn as nn
from typing import Dict, Tuple, Iterable, List
from src.model.vgg_extractor import *


def gram_matrix_optimized(input_tensor):
    """
    Calcula Gram Matrix normalizada SOLO por tamaño espacial.
    Esto evita que los valores se vuelvan infinitesimales.
    """
    B, C, H, W = input_tensor.size()
    features = input_tensor.view(B, C, H * W)
    G = torch.bmm(features, features.transpose(1, 2)) 
    
    return G.div(H * W)


class StyleTransferLoss(nn.Module):
    def __init__(self, encoder, content_weight=1.0, style_weight=5.0, tv_weight=1e-5, style_layer_weights=None, moment_weight=1.0):
        """
        ARGS:
            moment_weight (float): Peso extra para forzar coincidencia de Media/Std (Color).
                                   Recomendado: 1.0 a 5.0 si quieres mucho color.
        """
        super().__init__()
        self.encoder = encoder
        
        # Pesos globales
        self.cw = content_weight
        self.sw = style_weight
        self.tvw = tv_weight
        self.mw  = moment_weight # Peso de momentos (Color)
        
        if style_layer_weights is None:
            self.style_layer_weights = {
                'relu1_1': 1.0, 
                'relu2_1': 1.0, 
                'relu3_1': 1.0, 
                'relu4_1': 1.0}
        else:
            self.style_layer_weights = style_layer_weights
        
        self.mse = nn.MSELoss()
        
        # Constantes ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_prediction(self, img_0_1):
        return (img_0_1 - self.mean) / self.std

    def calculate_tv(self, img):
        return (torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + 
                torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))) / img.numel()

    def calc_mean_std(self, feat, eps=1e-5):
        """Calcula media y desviación estándar espacial."""
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    
    def forward(self, pred_logits, target_content_norm, target_style_norm):
        
        pred_img_0_1 = torch.sigmoid(pred_logits)
        pred_norm = self.normalize_prediction(pred_img_0_1)
        
        # Concatenar para una sola pasada
        full_batch = torch.cat([pred_norm, target_content_norm, target_style_norm], dim=0)
        feats = self.encoder(full_batch)
        
        # Separar
        b = pred_logits.size(0)
        f_pred = {k: v[:b] for k, v in feats.items()}
        f_cont = {k: v[b:2*b] for k, v in feats.items()}
        f_styl = {k: v[2*b:] for k, v in feats.items()}
        
        #  CONTENT LOSS 
        loss_c = self.mse(f_pred['relu4_1'], f_cont['relu4_1'].detach())
        
        # STYLE LOSS (Gram + Momentos) 
        loss_s = 0.0
        
        for layer, w in self.style_layer_weights.items():
            if layer in f_pred:
                # GRAM LOSS (Textura)
                gp = gram_matrix_optimized(f_pred[layer])
                gs = gram_matrix_optimized(f_styl[layer])
                loss_gram = self.mse(gp, gs.detach())
                
                # MOMENT LOSS (Color y Atmósfera) 
                # Calculamos media y std de la predicción y el target
                m_pred, s_pred = self.calc_mean_std(f_pred[layer])
                m_styl, s_styl = self.calc_mean_std(f_styl[layer])
                
                loss_moments = self.mse(m_pred, m_styl.detach()) + self.mse(s_pred, s_styl.detach())
                
                loss_s += w * (loss_gram + (self.mw * loss_moments))
            
        #  TV LOSS 
        loss_tv = self.calculate_tv(pred_img_0_1)
        
        total_loss = (self.cw * loss_c) + (self.sw * loss_s) + (self.tvw * loss_tv)
        
        return total_loss, {
            "total": total_loss.item(),
            "content": loss_c.item(),
            "style": loss_s.item(),
            "tv": loss_tv.item()}
