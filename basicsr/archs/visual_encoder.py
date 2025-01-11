import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.vqgan_arch import normalize, swish

from basicsr.utils.registry import ARCH_REGISTRY

class Upsample_visual_encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        up_size = {256: 512, 512: 768, 768: 256}
        x = F.interpolate(x, size=up_size[x.shape[-1]], mode="nearest")
        x = self.conv(x)

        return x

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class AttnBlock1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = torch.nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = torch.nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = torch.nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, s = q.shape  # s: feature size
        # q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        # k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        # v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        # h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_

@ARCH_REGISTRY.register()
class VisualEncoder(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, out_channels=77):
        super().__init__()
        self.nf = nf
        self.ch_mult = ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size
        self.in_channels = emb_dim
        self.out_channels = out_channels  # the size of text embedding for LCM is (77, 768)
        block_in_ch = self.nf * self.ch_mult[-1]

        blocks = []
        # initial conv
        blocks.append(nn.Conv1d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))  # (B, N, feature size): (B, 197, 768) -> (B, 512, 768)
        blocks.append(Upsample_visual_encoder(block_in_ch))  # (B, N, feature size): (B, 512, 768) -> (B, 512, 256)

        # non-local attention block
        blocks.append(ResBlock1D(block_in_ch, block_in_ch))  # (B, 512, 256) -> (B, 512, 256)
        blocks.append(AttnBlock1D(block_in_ch))              # (B, 512, 256) -> (B, 512, 256)
        blocks.append(ResBlock1D(block_in_ch, block_in_ch))  # (B, 512, 256) -> (B, 512, 256)

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = int(self.nf * self.ch_mult[i])

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock1D(block_in_ch, block_out_ch))  # (B, 512, 256) -> (B, 512, 256), (B, 512, 256) -> (B, 512, 256)
                block_in_ch = block_out_ch

                if block_out_ch in [512,]:
                    blocks.append(AttnBlock1D(block_in_ch))  # (B, 512, 256) -> (B, 512, 256)

            if i != 0:
                blocks.append(Upsample_visual_encoder(block_in_ch))

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv1d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x