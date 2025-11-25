import torch
import torch.nn as nn
import torchvision.models as models

# Mapeo estándar VGG19 (Features block)
VGG19_IDX2NAME = {
    0: "conv1_1", 1: "relu1_1", 2: "conv1_2", 3: "relu1_2", 4: "pool1",
    5: "conv2_1", 6: "relu2_1", 7: "conv2_2", 8: "relu2_2", 9: "pool2",
    10: "conv3_1", 11: "relu3_1", 12: "conv3_2", 13: "relu3_2",
    14: "conv3_3", 15: "relu3_3", 16: "conv3_4", 17: "relu3_4", 18: "pool3",
    19: "conv4_1", 20: "relu4_1", 21: "conv4_2", 22: "relu4_2",
    23: "conv4_3", 24: "relu4_3", 25: "conv4_4", 26: "relu4_4", 27: "pool4",
    28: "conv5_1", 29: "relu5_1", 30: "conv5_2", 31: "relu5_2",
    32: "conv5_3", 33: "relu5_3", 34: "conv5_4", 35: "relu5_4", 36: "pool5"}

VGG19_NAME2IDX = {v: k for k, v in VGG19_IDX2NAME.items()}

class VGGEncoder(nn.Module):
    def __init__(self, layers_to_extract=None, device="cuda", pretrained: bool = True):
        super().__init__()

        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg19(weights=weights).features
        
        for name, layer in vgg.named_children():
            if isinstance(layer, nn.ReLU):
                layer.inplace = False
        
        self.vgg_layers = vgg
        self.device = device
        
        if layers_to_extract is None:
            self.target_layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
        else:
            self.target_layers = layers_to_extract

        self.target_indices = {VGG19_NAME2IDX[n]: n for n in self.target_layers}
        self.max_idx = max(self.target_indices.keys())
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.to(device)
        self.eval()

    def forward(self, x):
        """
        Entrada: x (Tensor normalizado con ImageNet stats)
        Salida: Diccionario { 'relu1_1': tensor, ... }
        """
        outputs = {}
        
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            
            if i in self.target_indices:
                layer_name = self.target_indices[i]
                outputs[layer_name] = x

            if i >= self.max_idx:
                break
                
        return outputs

def get_vgg_encoder(device: str, pretrained: bool = True):
    targets = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
    return VGGEncoder(layers_to_extract=targets, device=device, pretrained=pretrained)
    

