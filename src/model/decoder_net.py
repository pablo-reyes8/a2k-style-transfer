import torch.nn.functional as F
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(p),
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=0),
            nn.InstanceNorm2d(out_ch, affine=True),]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class StyA2KDecoder(nn.Module):
    """
    Decoder tipo AdaIN:
        in:  [B, 512, ~31, ~31]  (features desde relu4_1 + attention fusion)
        out: [B, 3, 252, 252]    (imagen estilizada)

    Arquitectura (aprox AdaIN):
      Upsample
        512 -> 256
      Upsample
        256 -> 256 -> 256 -> 128
      Upsample
        128 -> 128 -> 64 -> 64 -> 3
    """
    
    def __init__(self, out_size=(252, 252)):
        super().__init__()
        self.out_size = out_size

        # Bloque alto nivel (512 -> 256)
        self.block1 = ConvBlock(512, 256)

        # Bloque medio (256 -> 256 -> 256 -> 128)
        self.block2_1 = ConvBlock(256, 256)
        self.block2_2 = ConvBlock(256, 256)
        self.block2_3 = ConvBlock(256, 128)

        # Bloque bajo (128 -> 128 -> 64 -> 64 -> 3)
        self.block3_1 = ConvBlock(128, 128)
        self.block3_2 = ConvBlock(128, 64)
        self.block3_3 = ConvBlock(64, 64)
        
        # última conv 
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0))

    def forward(self, x):
        """
        x: [B, 512, H, W] ~ [B, 512, 31, 31]
        """
        # 1Upsample + block1
        x = F.interpolate(x, scale_factor=2.0, mode="nearest") 
        x = self.block1(x)  # 512->256

        # Upsample + middle blocks
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")  
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)  # 256->128

        # Upsample + low-level blocks
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")  
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.out_conv(x)  # 64->3

        # Ajuste fino a 252x252 (d
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)

        return x