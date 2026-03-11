import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# DoubleConv3D
# =========================
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# Down block
# =========================
class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 只在 H,W 上池化，保持时间维不变
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

# =========================
# Up block
# =========================
class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1,2,2), mode="trilinear", align_corners=False)
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffT = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [0,diffW,0,diffH,0,diffT])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# =========================
# Temporal Residual Block
# =========================
class TemporalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, (3,1,1), padding=(1,0,0))
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, (3,1,1), padding=(1,0,0))
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + identity
        return F.relu(x)

# =========================
# Spatial Attention
# =========================
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2,1,7,padding=3)

    def forward(self, x):
        avg = torch.mean(x,dim=1,keepdim=True)
        mx,_ = torch.max(x,dim=1,keepdim=True)
        att = torch.cat([avg,mx],dim=1)
        att = torch.sigmoid(self.conv(att))
        return x*att

# =========================
# Temporal Attention
# =========================
class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//2,1,1)
        )

    def forward(self, x):
        score = self.net(x)
        weight = torch.softmax(score, dim=2)
        x = x * weight
        x = torch.sum(x, dim=2)
        return x

# =========================
# Radar3DUNet v4 Safe
# =========================
class Radar3DUNet(nn.Module):
    def __init__(self, in_channels=1, base_ch=32):
        super().__init__()
        self.inc = DoubleConv3D(in_channels, base_ch)

        self.down1 = Down3D(base_ch, base_ch*2)
        self.down2 = Down3D(base_ch*2, base_ch*4)
        self.down3 = Down3D(base_ch*4, base_ch*8)

        self.up1 = Up3D(base_ch*12, base_ch*4)
        self.up2 = Up3D(base_ch*6, base_ch*2)
        self.up3 = Up3D(base_ch*3, base_ch)

        self.temporal1 = TemporalBlock(base_ch)
        self.temporal2 = TemporalBlock(base_ch)

        self.spatial_att = SpatialAttention()
        self.temporal_att = TemporalAttention(base_ch)

        # occupancy head
        self.occ_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 1)
        )

        # height head
        self.hgt_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch,1,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: B,1,T,H,W
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4,x3)
        x = self.up2(x,x2)
        x = self.up3(x,x1)

        x = self.temporal1(x)
        x = self.temporal2(x)
        x = self.spatial_att(x)
        x = self.temporal_att(x)  # now x: B,C,H,W

        occ = self.occ_head(x)
        hgt = self.hgt_head(x)

        return occ,hgt

# =========================
# Test
# =========================
if __name__=="__main__":
    model = Radar3DUNet()
    x = torch.randn(2,1,5,128,128)
    occ,hgt = model(x)
    print(occ.shape)
    print(hgt.shape)