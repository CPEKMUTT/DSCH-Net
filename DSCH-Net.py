import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.layers import trunc_normal_
import math

class PDEBlock(nn.Module):
    
    #PDE-inspired Diffusion Block for Haze Removal
    #Applies a physics-motivated step based on the diffusion equation:
    #    x_{t+1} = x_t + Δt · div(grad(x_t))
    #Which models the "smoothing" or "diffusing" of haze using Laplacian.
    def __init__(self, channels, step_size=0.1, use_gate=True):
        super(PDEBlock, self).__init__()
        self.channels = channels
        self.step_size = step_size  # Δt
        self.use_gate = use_gate

        # Optional: Learnable gate for modulating the diffusion strength per pixel
        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
                nn.BatchNorm2d(channels),
                nn.Sigmoid()
            )

    def forward(self, x):
        
        #x: (B, C, H, W) feature map
        #Returns: (B, C, H, W) updated map with diffused features
        # Compute gradients (finite differences)
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]  # (B, C, H, W-1)
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]  # (B, C, H-1, W)
        grad_x = F.pad(grad_x, (0, 1, 0, 0))  # pad W dim
        grad_y = F.pad(grad_y, (0, 0, 0, 1))  # pad H dim
        div_x = grad_x[:, :, :, :-1] - grad_x[:, :, :, 1:]
        div_y = grad_y[:, :, :-1, :] - grad_y[:, :, 1:, :]
        div_x = F.pad(div_x, (1, 0, 0, 0))  # pad left
        div_y = F.pad(div_y, (0, 0, 1, 0))  # pad top
        laplacian = div_x + div_y  # shape: (B, C, H, W)

        # Optionally apply gate
        if self.use_gate:
            gate_map = self.gate(x)  # shape: (B, C, H, W), values in [0, 1]
            update = gate_map * laplacian
        else:
            update = laplacian
        # PDE update: x_{t+1} = x_t + Δt · laplacian
        x_next = x + self.step_size * update
        return x_next

# --------- small utilities ---------
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps    = eps
    def forward(self, x):
        # x: (N, C, H, W)
        mean = x.mean(dim=(2,3), keepdim=True)
        var  = x.var(dim=(2,3), unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias

def nhwc(x):  # (N, C, H, W) -> (N, H, W, C)
    return x.permute(0, 2, 3, 1).contiguous()
def nchw(x):  # (N, H, W, C) -> (N, C, H, W)
    return x.permute(0, 3, 1, 2).contiguous()

# --------- optional Mamba import ---------
_has_mamba = False
try:
    # pip install mamba-ssm  (and flash-attn==2.x if you want speedups)
    from mamba_ssm import Mamba
    _has_mamba = True
except Exception:
    _has_mamba = False


# --------- Fallback 1D "scan" (not a real SSM, but keeps API) ---------
class FauxScan1D(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=3):
        super().__init__()
        pad = (d_conv - 1) // 2       # <-- was d_conv//2
        self.dw = nn.Conv1d(dim, dim, kernel_size=d_conv, padding=pad, groups=dim)
        self.gate = nn.Conv1d(dim, dim, kernel_size=1)
    def forward(self, x_seq):  # (N, L, C)
        x = x_seq.transpose(1, 2)  # (N, C, L)
        y = self.dw(x)
        g = torch.sigmoid(self.gate(x))
        y = y * g
        return y.transpose(1, 2)   # (N, L, C)

# --------- One directional scan (→) ---------
class MambaScan1D(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if _has_mamba:
            self.core = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            if d_conv % 2 == 0:        # <-- enforce odd kernel in fallback
                d_conv += 1
            self.core = FauxScan1D(dim, d_state=d_state, d_conv=d_conv)


    def forward(self, x_seq):  # (N, L, C)
        return self.core(x_seq)

# --------- Multi-directional 2D block ---------
class MD_SSB(nn.Module):
    """
    Drop-in replacement for local/global attention.
    Input/Output: (N, C, H, W)
    """
    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=4,
        expand=2,
        ff_mult=2,
        directions=('lr', 'rl', 'tb', 'bt'),  # → ← ↓ ↑
        use_sum_fusion=True,
        dw_kernel=3,
        drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = LayerNorm2d(dim)
        self.dw = nn.Conv2d(dim, dim, kernel_size=dw_kernel, padding=dw_kernel//2, groups=dim)

        # directional scanners share parameters or not; here: separate scanners
        self.directions = directions
        self.scanners = nn.ModuleDict({  
            'lr': MambaScan1D(dim, d_state, d_conv, expand),
            'rl': MambaScan1D(dim, d_state, d_conv, expand),
            'tb': MambaScan1D(dim, d_state, d_conv, expand),
            'bt': MambaScan1D(dim, d_state, d_conv, expand),
        })

        self.use_sum_fusion = use_sum_fusion
        fuse_in = dim if use_sum_fusion else dim * len(directions)
        self.fuse = nn.Conv2d(fuse_in, dim, kernel_size=1)

        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        # FFGN (GEGLU) for channel mixing
        self.norm2 = LayerNorm2d(dim)
        hidden = int(ff_mult * dim)
        self.ffgn = nn.Sequential(
            nn.Conv2d(dim, hidden * 2, kernel_size=1),
            nn.GLU(dim=1),                # GEGLU-style gating (split on channel)
            nn.Conv2d(hidden, dim, kernel_size=1),
        )

    def _scan_lr(self, x):  # (N,H,W,C) -> (N,H,W,C) using → direction
        N, H, W, C = x.shape
        seq = x.reshape(N*H, W, C)          # rows as sequences
        out = self.scanners['lr'](seq)
        return out.reshape(N, H, W, C)

    def _scan_rl(self, x):
        N, H, W, C = x.shape
        seq = torch.flip(x, dims=[2]).reshape(N*H, W, C)
        out = self.scanners['rl'](seq)
        out = out.reshape(N, H, W, C)
        return torch.flip(out, dims=[2])

    def _scan_tb(self, x):
        N, H, W, C = x.shape
        seq = x.permute(0,2,1,3).reshape(N*W, H, C)   # columns as sequences (top->bottom)
        out = self.scanners['tb'](seq).reshape(N, W, H, C).permute(0,2,1,3)
        return out

    def _scan_bt(self, x):
        N, H, W, C = x.shape
        seq = torch.flip(x, dims=[1]).permute(0,2,1,3).reshape(N*W, H, C)
        out = self.scanners['bt'](seq).reshape(N, W, H, C).permute(0,2,1,3)
        out = torch.flip(out, dims=[1])
        return out

    def forward(self, x):  # (N,C,H,W)
        # Local preconditioning
        y = self.norm1(x)
        y = self.dw(y)
        # Channels-last for faster sequence ops
        y = nhwc(y)  # (N,H,W,C)
        outs = []
        if 'lr' in self.directions: outs.append(self._scan_lr(y))
        if 'rl' in self.directions: outs.append(self._scan_rl(y))
        if 'tb' in self.directions: outs.append(self._scan_tb(y))
        if 'bt' in self.directions: outs.append(self._scan_bt(y))

        if self.use_sum_fusion:
            y = torch.stack(outs, dim=0).sum(dim=0)      # sum-fuse
        else:
            y = torch.cat(outs, dim=-1)                  # concat-fuse

        y = nchw(y)                                      # (N,C,H,W)
        y = self.fuse(y)
        y = self.drop(y)
        x = x + y                                        # residual 1
        # FFGN
        z = self.norm2(x)
        z = self.ffgn(z)
        z = self.drop(z)
        return x + z                                     # residual 2

class RMDB(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 4, 8)):
        super(RMDB, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)

        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False)
            for d in dilations
        ])

        self.fuse_conv1 = nn.Conv2d(out_channels * len(dilations), out_channels, kernel_size=1, bias=False)
        self.fuse_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.conv1x1(x)
        residual = self.relu(residual)

        feats = [conv(residual) for conv in self.dilated_convs]
        out = torch.cat(feats, dim=1)
        out = self.fuse_conv1(out)

        out = out + residual
        out = self.fuse_conv2(out)
        out = out + residual
        return out

class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.Mish(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1/4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class DSC(nn.Module):
    def __init__(self, dim, network_depth):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.dim = dim
        # shallow feature extraction layer
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1) # main
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect') # main

        self.PDE = PDEBlock(channels=dim, step_size=0.1, use_gate=True)

        self.md_ssb = MD_SSB(
            dim=dim, d_state=16, d_conv=4, expand=2,
            directions=('lr', 'rl', 'tb', 'bt'),
            #directions=('lr','rl','tb','bt'),  # 4-way scan
            use_sum_fusion=True, ff_mult=2, drop=0.0
        )

        self.rmdb = RMDB(dim, dim)
        
        self.mlp  = Mlp(network_depth, dim, hidden_features=int(dim*4.), out_features=dim)
        self.mlp2 = Mlp(network_depth, dim, hidden_features=int(dim*4.), out_features=dim)
        

    def forward(self, x):
        # PDE Block
        identity = x
        x = self.norm1(x)
        x = self.PDE(x)
        x = self.mlp(x)
        x = identity + x

        # MD_SSB and RMD Block
        identity = x
        x = self.norm2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        #attn = torch.cat([self.md_ssb(x)], dim=1)
        attn = torch.cat([self.md_ssb(x), self.rmdb(x)], dim=1)
        x = self.mlp2(attn)
        x = identity + x
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, network_depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [DSC(dim=dim, network_depth=network_depth) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

# Selective Fusion Gate Mechanism
class SFGM(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SFGM, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')
    def forward(self, x):
        x = self.proj(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        if kernel_size is None:
            kernel_size = 1
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )
    def forward(self, x):
        x = self.proj(x)
        return x


class DSCH_Net_model(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, embed_dims=[24, 48, 96, 48, 24], depths=[1, 1, 2, 1, 1]):
        super(DSCH_Net_model, self).__init__()
        self.patch_size = 4

        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0], network_depth=sum(depths))

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)
        
        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)
        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1], network_depth=sum(depths))

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)
        
        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)
        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2], network_depth=sum(depths))

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        assert embed_dims[1] == embed_dims[3]
        self.sfgm1 = SFGM(embed_dims[3]) 
        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3], network_depth=sum(depths))

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        assert embed_dims[0] == embed_dims[4]
        self.sfgm2 = SFGM(embed_dims[4])  
        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4], network_depth=sum(depths))
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=1)
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
	
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x
        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x
        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)
        x = self.sfgm1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)
        x = self.sfgm2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x


    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x

def DSCH_Net_t():
    return DSCH_Net_model(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[1, 1, 2, 1, 1])