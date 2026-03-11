import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 3D U-Net Core Components
# =========================

class DoubleConv3D(nn.Module):
    """(Conv3D → BN → ReLU) ×2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Down3D(nn.Module):
    """Downsampling block: maxpool H/W then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up3D(nn.Module):
    """Upsampling block: ConvTranspose H/W then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=(1,2,2), stride=(1,2,2))
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)
        x1 = F.pad(
            x1,
            [diffW // 2, diffW - diffW // 2,
             diffH // 2, diffH - diffH // 2,
             0, 0]  # no pad T
        )
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Up_nocat3D(nn.Module):
    """Upsample H/W then conv, no skip concat"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(1,2,2), stride=(1,2,2))
        self.conv = DoubleConv3D(out_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x