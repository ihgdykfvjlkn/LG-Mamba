
#!/usr/bin/env python3
"""
Modified demomodel.py (NR-SCIQA Pipeline with High-Pass FFT Enhancement)
Refinement Version: distorted-image-only prediction with reference-assisted training.
Targeting improvements for MAR, JP2K, and LSC distortion types.
"""

import sys

vim_path = "/home/ssss/Vim/vim"
sys.path.insert(0, vim_path)

from models_mamba import VisionMamba

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.fft

eps = 1e-8


# =========================================================
# 1. Distortion Attention
# =========================================================
class DistortionAttention(nn.Module):
    def __init__(self, channels):
        super(DistortionAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.attention(x)
        out = x + x * weight
        return out


# =========================================================
# 2. High-Pass FFT Branch (Replaces LocalContrast for JP2K/MAR/LSC)
# =========================================================
class HighPassFFTBranch(nn.Module):
    """
    频域高通滤波增强模块（替代 LocalContrastBranch）。
    通过 2D 傅里叶变换将图像转换到频域，遮蔽中心低频区域，
    仅保留边缘、纹理与高频噪声/伪影（针对 MAR/JP2K/LSC 的模糊与块效应）。
    使用可学习门控参数 gamma 进行温和融合。
    """

    def __init__(self, in_channels=3, cutoff_ratio=0.15):
        """
        :param in_channels: 输入特征图通道数 (RGB 为 3)
        :param cutoff_ratio: 低频遮蔽半径比例 (0.10~0.20 之间，遮住中心低频)
        """
        super().__init__()
        self.cutoff_ratio = cutoff_ratio

        # 高频特征后处理卷积层
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

        # 可学习门控参数，初始化为 0，防止破坏预训练权重的稳定性
        self.gamma = nn.Parameter(torch.zeros(1))

    def create_highpass_mask(self, h, w, device):
        # 创建中心点为 0（低频），四周为 1（高频）的 Mask
        crop_h = max(1, int(h * self.cutoff_ratio))
        crop_w = max(1, int(w * self.cutoff_ratio))

        mask = torch.ones((1, 1, h, w), device=device, dtype=torch.float32)
        center_h, center_w = h // 2, w // 2

        # 将中心低频部分置零
        mask[:, :, center_h - crop_h: center_h + crop_h, center_w - crop_w: center_w + crop_w] = 0.0
        return mask

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. 正向 2D Real FFT 变换
        fft_x = torch.fft.fft2(x, norm='ortho')
        fft_shift = torch.fft.fftshift(fft_x)  # 将低频成分平移至中心

        # 2. 施加高通 Mask 滤除低频
        mask = self.create_highpass_mask(h, w, x.device)
        high_freq_shift = fft_shift * mask

        # 3. 逆傅里叶变换还原至空间域
        high_freq_x = torch.fft.ifftshift(high_freq_shift)
        x_high = torch.fft.ifft2(high_freq_x, norm='ortho').real  # 提取实部

        # 4. 卷积特征映射
        x_high_feat = self.proj(x_high)

        # 5. 可学习残差融合：x + gamma * high_freq_feat
        return x + self.gamma * x_high_feat


# =========================================================
# 3. Edge Enhancement (SCI Dedicated)
# =========================================================
class SCIEdge(nn.Module):
    """
    针对屏幕图像(SCI)设计的多分支高频/边缘感知增强模块
    分支包含：
      1. 多方向一阶 Sobel 边缘 (0°, 45°, 90°, 135°)
      2. 二阶拉普拉斯高频感知
      3. 高斯差分(DoG)多尺度频率分支
      4. 轻量级通道注意力融合机制
    """

    def __init__(self, in_channels=3, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.in_channels = in_channels

        # 1. Edge Branch: 多方向 Sobel 算子
        k_sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        k_sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        k_sobel_45 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32)
        k_sobel_135 = torch.tensor([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=torch.float32)

        self.register_buffer("kx", k_sobel_x.view(1, 1, 3, 3))
        self.register_buffer("ky", k_sobel_y.view(1, 1, 3, 3))
        self.register_buffer("k45", k_sobel_45.view(1, 1, 3, 3))
        self.register_buffer("k135", k_sobel_135.view(1, 1, 3, 3))

        # 2. Laplacian Branch: 二阶算子
        k_laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("klap", k_laplacian.view(1, 1, 3, 3))

        # 3. Frequency Branch: 高斯核 (DoG)
        k_gauss_3 = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
        k_gauss_5 = torch.tensor([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32) / 256.0

        self.register_buffer("kgauss3", k_gauss_3.view(1, 1, 3, 3))
        self.register_buffer("kgauss5", k_gauss_5.view(1, 1, 5, 5))

        # 4. Concatenation & Attention Fusion
        concat_channels = in_channels * 4

        self.proj = nn.Sequential(
            nn.Conv2d(concat_channels, in_channels * 2, kernel_size=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, (in_channels * 2) // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d((in_channels * 2) // 4, in_channels * 2, kernel_size=1),
            nn.Sigmoid()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # Branch 1: Sobel
        gx = F.conv2d(x, self.kx.repeat(c, 1, 1, 1), padding=1, groups=c)
        gy = F.conv2d(x, self.ky.repeat(c, 1, 1, 1), padding=1, groups=c)
        g45 = F.conv2d(x, self.k45.repeat(c, 1, 1, 1), padding=1, groups=c)
        g135 = F.conv2d(x, self.k135.repeat(c, 1, 1, 1), padding=1, groups=c)
        edge_feat = torch.sqrt(gx ** 2 + gy ** 2 + g45 ** 2 + g135 ** 2 + self.eps)

        # Branch 2: Laplacian
        lap_feat = torch.abs(F.conv2d(x, self.klap.repeat(c, 1, 1, 1), padding=1, groups=c))

        # Branch 3: DoG
        low_freq3 = F.conv2d(x, self.kgauss3.repeat(c, 1, 1, 1), padding=1, groups=c)
        low_freq5 = F.conv2d(x, self.kgauss5.repeat(c, 1, 1, 1), padding=2, groups=c)
        freq_feat = torch.abs(x - low_freq3) + torch.abs(low_freq3 - low_freq5)

        # Branch 4: Fusion
        feat_concat = torch.cat([x, edge_feat, lap_feat, freq_feat], dim=1)
        feat_proj = self.proj(feat_concat)
        weights = self.attn(feat_proj)
        feat_focused = feat_proj * weights
        enhanced = self.out_conv(feat_focused)

        return x + enhanced


# =========================================================
# 4. CNN Backbone (MobileNetV2 Shallow Features)
# =========================================================
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        self.mobilenet_features = m.features[:7]
        self.proj = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.mobilenet_features(x)
        x = F.adaptive_avg_pool2d(x, (16, 16))
        x = self.proj(x)
        return x

    def set_stage2_unfreeze(self):
        for param in self.mobilenet_features[:5].parameters():
            param.requires_grad = False
        for param in self.mobilenet_features[5:].parameters():
            param.requires_grad = True


# =========================================================
# 5. Vision Mamba Branch & Local-Global Interaction
# =========================================================
class VisionMambaBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.mamba = VisionMamba(
            img_size=32,
            patch_size=4,
            stride=4,
            embed_dim=192,
            depth=6,
            num_classes=0,
            if_cls_token=False,
            final_pool_type='all'
        )

        self.proj = nn.Linear(192, 128)

    def forward(self, x):
        feat = self.mamba(x)
        if feat.dim() == 3:
            feat = feat.mean(dim=1)

        assert feat.shape[-1] == 192, f"Mamba output error: {feat.shape}"

        out = self.proj(feat)
        return out


class LocalGlobalInteraction(nn.Module):
    """Fuse local and global features extracted from the same image."""

    def __init__(self, channels=128, hidden_channels=256, dropout=0.1):
        super().__init__()
        self.local_norm = nn.LayerNorm(channels)
        self.global_norm = nn.LayerNorm(channels)
        interaction_channels = channels * 4

        self.interaction = nn.Sequential(
            nn.Linear(interaction_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels)
        )
        self.gate = nn.Sequential(
            nn.Linear(interaction_channels, channels),
            nn.Sigmoid()
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float32)
        )

    def forward(self, local, global_feature):
        local_norm = self.local_norm(local)
        global_norm = self.global_norm(global_feature)
        interaction_input = torch.cat(
            [
                local_norm,
                global_norm,
                torch.abs(local_norm - global_norm),
                local_norm * global_norm,
            ],
            dim=1,
        )
        update = self.interaction(interaction_input)
        gate = self.gate(interaction_input)
        return local + self.residual_scale * gate * update


# =========================================================
# 6. No-reference Quality Feature Projection
# =========================================================
class NRFeatureProjection(nn.Module):
    """Build the quality feature from the distorted image only."""

    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )

    def forward(self, dist):
        return F.normalize(self.fusion(dist), dim=1)


# =========================================================
# 7. Base IQA Network
# =========================================================
class BaseIQANet(nn.Module):
    def __init__(
            self,
            istrain=True,
            n_class=46,
            use_mamba=False
    ):
        super().__init__()
        self.istrain = istrain
        self.use_mamba = use_mamba
        self.n_class = n_class

        # --- 输入并行双分支增强模块 ---
        self.sci_edge = SCIEdge()
        # 将原有的 LocalContrastBranch 替换为频域高通增强模块
        self.fft_highpass = HighPassFFTBranch(in_channels=3, cutoff_ratio=0.15)

        # 融合层：Concat (3通道 Edge + 3通道 FFT 高频 = 6通道) -> 还原为 3 通道
        self.edge_fft_fusion = nn.Sequential(
            nn.Conv2d(3 + 3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.GELU()
        )

        # CNN Backbone
        self.cnn = CNNBackbone()

        # optional Mamba
        if self.use_mamba:
            self.mamba = VisionMambaBranch()
            self.local_global_interaction = LocalGlobalInteraction()
        else:
            self.fusion = NRFeatureProjection()

        # Patch Score 预测器
        self.patch_score = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # Distortion Attention
        self.dist_attention = DistortionAttention(channels=128)

        # Regression Head
        self.regression = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        # Learnable Score Scale
        self.scw = nn.Parameter(torch.tensor([1.0]))

        # Distortion Classification
        self.distype_cls = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_class)
        )

    def encode(self, x):
        # 1. 空间域 SCI 边缘特征提取
        edge_feat = self.sci_edge(x)

        # 2. 频域高通滤波特征提取（保留全局/局部高频，去除中心低频）
        fft_feat = self.fft_highpass(x)

        # 3. 拼接并降维融合
        enhanced = torch.cat([edge_feat, fft_feat], dim=1)
        enhanced = self.edge_fft_fusion(enhanced)

        # 4. 残差叠加，喂给 MobileNetV2 Backbone
        f = self.cnn(x + enhanced)

        local = F.adaptive_avg_pool2d(f, 1).flatten(1)

        if self.use_mamba:
            global_feature = self.mamba(x)
            fused = self.local_global_interaction(local, global_feature)
            return fused, global_feature

        return self.fusion(local), None

    def forward(self, x1, x2=None, x3=None):
        B, P = x1.shape[:2]

        # The inference score path only encodes distorted-image patches.
        x1 = x1.view(-1, *x1.shape[-3:])
        l1, g1 = self.encode(x1)

        # Reference and pseudo-reference images are training-only auxiliaries.
        if self.istrain:
            if x2 is None or x3 is None:
                raise ValueError(
                    'Training mode requires distorted, reference, and '
                    'pseudo-reference inputs.'
                )
            x2 = x2.view(-1, *x2.shape[-3:])
            x3 = x3.view(-1, *x3.shape[-3:])
            l2, g2 = self.encode(x2)
            l3, g3 = self.encode(x3)

        # -----------------------------
        # No-reference prediction path. l1 already contains local-global
        # interaction from the distorted image only. l2/l3 remain auxiliary
        # features for the training losses below.
        # -----------------------------
        feat = l1

        # patch aggregation 维度：[B * P, 128] -> [B, P, 128]
        feat = feat.view(B, P, -1)

        # Soft Distortion Attention
        score_patch = self.patch_score(feat).squeeze(-1)  # [B, P]
        weight = torch.softmax(score_patch, dim=1)  # [B, P]

        # 加权平均求和
        feat = torch.sum(feat * weight.unsqueeze(-1), dim=1)  # [B, 128]

        # distortion attention
        feat = self.dist_attention(feat)
        # classification branch
        cls_out = self.distype_cls(feat)
        # Regression branch
        raw_score = self.regression(feat)

        # 恢复 0-100 的输出标尺
        score = torch.sigmoid(raw_score) * 100.0
        score = score.squeeze(dim=-1)
        score = score * torch.clamp(self.scw, 0.5, 2.0)

        if self.istrain:
            diff_ref = torch.abs(l1 - l2)
            diff_pref = torch.abs(l1 - l3)

            return (
                score,
                l1,
                l2,
                l3,
                l2,
                l1,
                diff_ref,
                diff_pref,
                cls_out
            )
        else:
            return score


# =========================================================
# External Interfaces
# =========================================================
class IQANet(BaseIQANet):
    def __init__(self, istrain=True, n_class=46):
        super().__init__(
            istrain=istrain,
            n_class=n_class,
            use_mamba=False
        )


class DFSSMambaNet(BaseIQANet):
    def __init__(self, istrain=True, n_class=46, mamba_img_size=32):
        super().__init__(
            istrain=istrain,
            n_class=n_class,
            use_mamba=True
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DFSSMambaNet(istrain=True).to(device)
    model.eval()

    dummy = torch.randn(2, 32, 3, 32, 32).to(device)

    with torch.no_grad():
        out = model(dummy, dummy, dummy)

    print("Train Output Elements Count:", len(out))
    print("Score Output Shape:", out[0].shape)
    print("Diff Ref Feature Shape:", out[4].shape)
    print("Model forward test passed successfully!")
