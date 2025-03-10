import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.autograd import Variable
import numpy as np

# class LayerNorm2d(nn.Module):
#     """Layer Normalization for 2D input.

#     Args:
#         num_channels (int): Number of channels.
#         eps (float): Epsilon. Default: 1e-6.
#     """

#     def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(num_channels))
#         self.bias = nn.Parameter(torch.zeros(num_channels))
#         self.eps = eps

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward function.

#         Args:
#             x (torch.Tensor): Input feature map in shape (N, C, H, W)

#         Returns:
#             torch.Tensor: Output feature map in shape (N, C, H, W)
#         """
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         x = self.weight[:, None, None] * x + self.bias[:, None, None]
#         return x
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

def get_activation(type):
    if type == 'leaky relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif type == 'relu':
        return nn.ReLU(inplace=True)
    elif type == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError

class ConvWithActivation(nn.Module):
    def __init__(self, conv_type, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation='relu', using_sn=True):
        super(ConvWithActivation, self).__init__()
        if conv_type == 'conv':
            conv_func = nn.Conv2d 
        elif conv_type == 'deconv':
            conv_func = nn.ConvTranspose2d
        self.conv2d = conv_func(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        self.using_sn = using_sn
        if using_sn == True:
            self.conv2d = nn.utils.spectral_norm(self.conv2d)
        else:
            # self.norm = nn.InstanceNorm2d(out_channels)
            self.norm = LayerNorm2d(out_channels)
            
        self.activation = get_activation(activation)

        for m in self.modules():
            if isinstance(m, conv_func):
                nn.init.kaiming_normal_(m.weight)
        
    def forward(self, x):
        x = self.conv2d(x)
        if not self.using_sn:
            x = self.norm(x)
        x = self.activation(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # import pdb;pdb.set_trace()
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class PixelDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 **kwargs):
        super(PixelDecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.sr_norm = nn.LayerNorm(self.in_channels)
        
        self.ups= nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.in_channels, self.in_channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                    ConvWithActivation('conv', self.in_channels//2, self.in_channels, 3, 1, 1, using_sn=False),
                ),
                nn.Sequential(
                    nn.Conv2d(self.in_channels, self.in_channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                    ConvWithActivation('conv', self.in_channels // 2, self.in_channels, 3, 1, 1, using_sn=False),
                ),
        ])
        
        self.con_out1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.con_out2 = nn.Conv2d(64, 3, 3, 1, 1)
        
        self.proj = nn.Conv2d(self.in_channels, 64, 3, 1, 1)
        
        self.dropout=nn.Dropout2d(p=0.7)

    def forward(self,
                feat,
                text_feat,
                middle):
        
        
        b, c = feat.shape[0], feat.shape[1]
        
        src = feat #.permute(0, 2, 1).view(b, c, 8, 32) 
        
        # import pdb;pdb.set_trace()
        x = self.ups[0](src)
        x = self.ups[1](x)
        
        query = self.sr_norm(text_feat)
        out = torch.einsum('n c h w, n b c -> n b h w', x, query)
        previous = F.interpolate(middle[0].permute(0, 2, 1).view(b, c, 8, 32), scale_factor=4.0)
        
        # # import pdb;pdb.set_trace()
        out = self.dropout(self.con_out1(out + self.proj(previous)))
        out = self.con_out2(out)
        
        # out = self.con_out1(out + self.proj(previous))
        
        return (torch.tanh(out) + 1)/2
   

    def inference(self,
                feat,
                text_feat, 
                middle) -> torch.Tensor:
        """Sample function. Used for inference.

        Args:
            feat (torch.Tensor): Backbone output of shape (N, T, C).
            out_enc (torch.Tensor): Encoder output of shape (N, T, C). Unused.
                Defaults to None.

        Returns:
            torch.Tensor: Output of shape (N, max_len).
        """
        return self.forward(feat=feat, text_feat=text_feat, middle=middle)

def build_pixel_deocder(args):
    return PixelDecoder(768, 3)
