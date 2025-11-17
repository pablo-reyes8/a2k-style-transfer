import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple, Iterable, List


VGG19_IDX2NAME = {
    0:"conv1_1", 1:"relu1_1", 2:"conv1_2", 3:"relu1_2", 4:"pool1",
    5:"conv2_1", 6:"relu2_1", 7:"conv2_2", 8:"relu2_2", 9:"pool2",
    10:"conv3_1",11:"relu3_1",12:"conv3_2",13:"relu3_2",
    14:"conv3_3",15:"relu3_3",16:"conv3_4",17:"relu3_4",18:"pool3",
    19:"conv4_1",20:"relu4_1",21:"conv4_2",22:"relu4_2",
    23:"conv4_3",24:"relu4_3",25:"conv4_4",26:"relu4_4",27:"pool4",
    28:"conv5_1",29:"relu5_1",30:"conv5_2",31:"relu5_2",
    32:"conv5_3",33:"relu5_3",34:"conv5_4",35:"relu5_4",36:"pool5"}

VGG19_NAME2IDX = {v:k for k,v in VGG19_IDX2NAME.items()}

# Por convención de estilo/perceptual loss usamos activaciones ReLU:
DEFAULT_CONTENT_LAYERS = ["relu4_1"] # El contenido esta bien representado aca 
DEFAULT_STYLE_LAYERS   = ["relu1_1","relu2_1","relu3_1","relu4_1","relu5_1"] # El estilo es mas multifacetico y se captura en varias capas 


def load_vgg19_features(device="cuda"):
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

    for p in vgg.parameters():
        p.requires_grad = False # Congelar pesos IMPORTANTE

    vgg.eval().to(device)
    return vgg

class VGGFeatureExtractor(nn.Module):
    """
    Recorre vgg.features y devuelve un dict con activaciones
    en las capas solicitadas (por nombre tipo 'relu4_1', etc.).
    Incluye Normalization al inicio para VGG-preprocessing.
    """
    def __init__(self,
                 vgg_features: nn.Sequential,
                 target_layer_names: Iterable[str],
                 device="cuda"):
        super().__init__()
        self.device = device
        self.vgg = vgg_features 
        self.target_idx = sorted([VGG19_NAME2IDX[name] for name in target_layer_names])
        self.target_set = set(self.target_idx)
        self.return_names = [VGG19_IDX2NAME[i] for i in self.target_idx]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.target_set:
                feats[VGG19_IDX2NAME[i]] = x
                if len(feats) == len(self.target_set):
                    # podrías hacer break si quieres, pero no es obligatorio
                    pass
        return feats

# Constructores específicos (content / style / losses)
def build_vgg_content_extractor(device="cuda",
                                content_layers: List[str] = None) -> VGGFeatureExtractor:
    if content_layers is None:
        content_layers = DEFAULT_CONTENT_LAYERS
    vgg = load_vgg19_features(device)
    return VGGFeatureExtractor(vgg, content_layers, device=device)

def build_vgg_style_extractor(device="cuda",
                              style_layers: List[str] = None) -> VGGFeatureExtractor:
    if style_layers is None:
        style_layers = DEFAULT_STYLE_LAYERS
    vgg = load_vgg19_features(device)
    return VGGFeatureExtractor(vgg, style_layers, device=device)

def build_vgg_loss_extractor(device="cuda",
                             content_layers: List[str] = None,
                             style_layers: List[str] = None) -> VGGFeatureExtractor:
    """
    Extractor 'final' para pérdidas: set de capas = content ∪ style.
    """
    if content_layers is None: 
      content_layers = DEFAULT_CONTENT_LAYERS
      
    if style_layers is None:   
      style_layers = DEFAULT_STYLE_LAYERS

    all_layers = sorted(set(content_layers) | set(style_layers),
                        key=lambda n: VGG19_NAME2IDX[n])
    
    vgg = load_vgg19_features(device)
    return VGGFeatureExtractor(vgg, all_layers, device=device)