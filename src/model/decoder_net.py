import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d(p)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=0)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.act(x)
        return x


class StyA2KDecoderMultiLevel(nn.Module):
    """
    Multi-scale decoder that upsamples fused relu4_1/3_1 features back to RGB space.

    Expected input shapes:
      - x_fused_4: [B, 512, 32, 32]  (structural features)
      - x_fused_3: [B, 256, 64, 64]  (color/texture features)
    """

    def __init__(self):
        super().__init__()

        # --- Bloque 4 (512 -> 256) ---
        self.block4_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 32 -> 64
            ConvBlock(512, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),)

        # Mezcla con nivel 3
        self.mix_3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),)

        # Bloque 3 (256 -> 128)
        self.block3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 64 -> 128
            ConvBlock(256, 128),
            ConvBlock(128, 128),)

        # Bloque 2 (128 -> 64)
        self.block2_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 128 -> 256
            ConvBlock(128, 64),
            ConvBlock(64, 64),)

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, padding=0),)

    def forward(self, x_fused_4: torch.Tensor, x_fused_3: torch.Tensor) -> torch.Tensor:
        h = self.block4_up(x_fused_4)  # [B, 256, 64, 64]

        h_cat = torch.cat([h, x_fused_3], dim=1)  # [B, 512, 64, 64]
        h = self.mix_3(h_cat)  # [B, 256, 64, 64]

        h = self.block3_up(h)  # [B, 128, 128, 128]
        h = self.block2_up(h)  # [B, 64, 256, 256]

        return self.out_conv(h)  # [B, 3, 256, 256]

