import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONV + BN + RELU
# ============================================================

class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# MODIFIED RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvBNReLU(
            in_channels,
            out_channels
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + identity

        return self.relu(out)


# ============================================================
# MEMORY-SAFE MULTI-HEAD ATTENTION
#
# IMPORTANT:
# Attention is applied only at the bottleneck.
# For 1024x1024 input:
#
# 1024 -> 512 -> 256 -> 128 -> 64 -> 32
#
# Attention therefore operates on 32x32 = 1024 tokens,
# instead of 512x512 = 262144 tokens.
# ============================================================

class MultiHeadAttentionBlock(nn.Module):

    def __init__(
        self,
        channels,
        num_heads=8
    ):
        super().__init__()

        self.norm = nn.LayerNorm(channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        b, c, h, w = x.shape

        # B,C,H,W -> B,H*W,C
        tokens = x.flatten(2).transpose(1, 2)

        normalized = self.norm(tokens)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False
        )

        attention_output = self.projection(
            attention_output
        )

        tokens = tokens + attention_output

        # B,H*W,C -> B,C,H,W
        output = tokens.transpose(
            1, 2
        ).reshape(
            b,
            c,
            h,
            w
        )

        return output


# ============================================================
# CSSE
# ============================================================

class CSSEBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        hidden = max(channels // 4, 1)

        # Channel branch
        self.channel_branch = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

        # Spatial branch
        self.spatial_branch = nn.Sequential(

            nn.Conv2d(
                channels,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        channel_attention = self.channel_branch(x)

        spatial_attention = self.spatial_branch(x)

        channel_output = x * channel_attention

        spatial_output = x * spatial_attention

        return channel_output + spatial_output


# ============================================================
# CBAM - CHANNEL ATTENTION
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16
    ):

        super().__init__()

        hidden = max(
            channels // reduction,
            1
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(

            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_attention = self.mlp(
            self.avg_pool(x)
        )

        max_attention = self.mlp(
            self.max_pool(x)
        )

        attention = self.sigmoid(
            avg_attention + max_attention
        )

        return x * attention


# ============================================================
# CBAM - SPATIAL ATTENTION
# ============================================================

class SpatialAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_map = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_map, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        combined = torch.cat(
            [avg_map, max_map],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(combined)
        )

        return x * attention


# ============================================================
# CBAM
# ============================================================

class CBAM(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel_attention = ChannelAttention(
            channels
        )

        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x


# ============================================================
# DECODER BLOCK
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.residual = ResidualBlock(
            in_channels + skip_channels,
            out_channels
        )

        self.csse = CSSEBlock(
            out_channels
        )

        self.cbam = CBAM(
            out_channels
        )

    def forward(self, x, skip):

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.residual(x)

        x = self.csse(x)

        x = self.cbam(x)

        return x


# ============================================================
# LITE-RSNET
# ============================================================

class LiteRSNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_classes=1
    ):

        super().__init__()

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        self.enc1 = ResidualBlock(
            in_channels,
            32
        )

        self.enc2 = ResidualBlock(
            32,
            64
        )

        self.enc3 = ResidualBlock(
            64,
            128
        )

        self.enc4 = ResidualBlock(
            128,
            256
        )

        self.bottleneck = ResidualBlock(
            256,
            512
        )

        # ----------------------------------------------------
        # POOLING
        # ----------------------------------------------------

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ----------------------------------------------------
        # ATTENTION ONLY AT BOTTLENECK
        # ----------------------------------------------------

        self.bottleneck_attention = (
            MultiHeadAttentionBlock(
                channels=512,
                num_heads=8
            )
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.dec4 = DecoderBlock(
            512,
            256,
            256
        )

        self.dec3 = DecoderBlock(
            256,
            128,
            128
        )

        self.dec2 = DecoderBlock(
            128,
            64,
            64
        )

        self.dec1 = DecoderBlock(
            64,
            32,
            32
        )

        # ----------------------------------------------------
        # FINAL RAIL SEGMENTATION HEAD
        #
        # 1 channel:
        # 0 = background
        # 1 = rail
        # ----------------------------------------------------

        self.final_conv = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(x)

        p1 = self.pool(e1)

        e2 = self.enc2(p1)

        p2 = self.pool(e2)

        e3 = self.enc3(p2)

        p3 = self.pool(e3)

        e4 = self.enc4(p3)

        p4 = self.pool(e4)

        # ====================================================
        # BOTTLENECK
        #
        # 1024x1024 input -> 64x64 here
        # ====================================================

        b = self.bottleneck(p4)

        # Further reduce before attention
        b_small = F.max_pool2d(
            b,
            kernel_size=2,
            stride=2
        )

        # 64x64 -> 32x32
        b_small = self.bottleneck_attention(
            b_small
        )

        # Restore to 64x64
        b = F.interpolate(
            b_small,
            size=b.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Residual fusion
        b = b + self.bottleneck(p4)

        # ====================================================
        # DECODER
        # ====================================================

        d4 = self.dec4(
            b,
            e4
        )

        d3 = self.dec3(
            d4,
            e3
        )

        d2 = self.dec2(
            d3,
            e2
        )

        d1 = self.dec1(
            d2,
            e1
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        output = self.final_conv(d1)

        output = F.interpolate(
            output,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return output


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MEMORY-SAFE LITE-RSNET TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU Memory:",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = LiteRSNet(
        in_channels=3,
        num_classes=1
    ).to(device)

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: {total_params:,}"
    )

    print(
        f"Trainable parameters: {trainable_params:,}"
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    x = torch.randn(
        1,
        3,
        1024,
        1024,
        device=device
    )

    print(
        "Input shape:",
        tuple(x.shape)
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    torch.cuda.empty_cache()

    with torch.no_grad():

        output = model(x)

    print(
        "Output shape:",
        tuple(output.shape)
    )

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        print(
            f"GPU allocated: {allocated:.2f} GB"
        )

        print(
            f"GPU reserved: {reserved:.2f} GB"
        )

    print("=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
