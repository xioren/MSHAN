import torch
import torch.nn as nn


class MSFEM(nn.Module):
    """Multi-Scale Feature Extraction Module"""
    def __init__(self, in_channels, out_channels):
        super(MSFEM, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        feat3 = self.relu(self.conv3(x))
        feat5 = self.relu(self.conv5(x))
        feat7 = self.relu(self.conv7(x))
        feat_dilated = self.relu(self.dilated_conv(x))
        concat = torch.cat([feat3, feat5, feat7, feat_dilated], dim=1)
        out = self.fusion(concat)
        return out


class LPU(nn.Module):
    """Local Perception Unit"""
    def __init__(self, dim):
        super(LPU, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class CFFN(nn.Module):
    """Convolutional Feed-Forward Network"""
    def __init__(self, dim, ffn_expansion_factor=4):
        super(CFFN, self).__init__()
        hidden_dim = dim * ffn_expansion_factor
        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class HATB(nn.Module):
    """Hybrid Attention Transformer Block"""
    def __init__(self, dim, num_heads):
        super(HATB, self).__init__()
        self.lpu = LPU(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.cffn = CFFN(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Local Perception
        x = self.lpu(x)

        # Self-Attention
        x_residual = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x_residual + attn_output

        # Feed-Forward Network
        x_residual = x
        x = self.norm2(x)
        x = self.cffn(x)
        x = x_residual + x

        return x


class PUM(nn.Module):
    """Pixel Upsampling Module"""
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(PUM, self).__init__()
        self.upscale_factor = upscale_factor
        self.layers = []
        for _ in range(int(upscale_factor).bit_length() - 1):
            self.layers += [
                nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                ChannelAttention(in_channels),
                SpatialAttention(),
            ]
        self.layers = nn.Sequential(*self.layers)
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.final_conv(x)
        return x


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y


class MSHAN(nn.Module):
    """Multi-Scale Hybrid Attention Network"""
    def __init__(self, in_channels=3, out_channels=3, dim=64, num_blocks=6, num_heads=8, upscale_factor=2):
        super(MSHAN, self).__init__()
        self.msfem = MSFEM(in_channels, dim)
        self.hatb_blocks = nn.ModuleList([
            HATB(dim=dim, num_heads=num_heads) for _ in range(num_blocks)
        ])
        self.pum = PUM(dim, out_channels, upscale_factor)
        self.global_residual = nn.Upsample(scale_factor=upscale_factor, mode="bilinear", align_corners=False)
        self.act = nn.Tanh()
    
    def forward(self, x):
        # Extract multi-scale features
        feat = self.msfem(x)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # Shape: (B, N, C), where N = H * W

        # Hybrid Attention Transformer Blocks
        for block in self.hatb_blocks:
            feat = block(feat)

        # Reshape back to image
        feat = feat.transpose(1, 2).view(B, C, H, W)

        # Progressive Upsampling
        out = self.pum(feat)

        # Global Residual Learning
        residual = self.global_residual(x)
        out = out + residual

        return self.act(out)
