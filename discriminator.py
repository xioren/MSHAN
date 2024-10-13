import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNormResidualBlock(nn.Module):
    """Spectral Normalized Residual Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(SpectralNormResidualBlock, self).__init__()
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)
        return out


class SelfAttention(nn.Module):
    """Self-Attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.ch_in = in_dim

        self.query_conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        )
        self.key_conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        )
        self.value_conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height)  # B, C', N
        proj_query = proj_query.permute(0, 2, 1)  # B, N, C'
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B, C', N
        energy = torch.bmm(proj_query, proj_key)  # B, N, N
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B, C, N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B, C, N
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ChannelAttentionModule(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    """Spatial Attention Module"""
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # B, 1, H, W
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # B, 1, H, W
        x_cat = torch.cat([avg_out, max_out], dim=1)  # B, 2, H, W
        attention = self.sigmoid(self.conv(x_cat))
        out = x * attention
        return out


class MSHAD(nn.Module):
    """Multi-Scale Hybrid Attention Discriminator"""
    def __init__(self, in_channels=3, base_channels=64):
        super(MSHAD, self).__init__()
        self.scale1 = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormResidualBlock(base_channels, base_channels),
            CBAM(base_channels),
            SpectralNormResidualBlock(base_channels, base_channels * 2, stride=2),
            CBAM(base_channels * 2),
            SpectralNormResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            SelfAttention(base_channels * 4),
            SpectralNormResidualBlock(base_channels * 4, base_channels * 8, stride=2),
            CBAM(base_channels * 8),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(base_channels * 8, 1))
        )

        # For multi-scale input processing
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormResidualBlock(base_channels, base_channels),
            CBAM(base_channels),
            SpectralNormResidualBlock(base_channels, base_channels * 2, stride=2),
            CBAM(base_channels * 2),
            SpectralNormResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            SelfAttention(base_channels * 4),
            SpectralNormResidualBlock(base_channels * 4, base_channels * 8, stride=2),
            CBAM(base_channels * 8),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(base_channels * 8, 1))
        )

    def forward(self, x):
        out_scale1 = self.scale1(x)
        out_scale2 = self.scale2(x)
        out = out_scale1 + out_scale2
        return out
