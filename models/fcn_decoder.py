import torch
import torch.nn as nn

class FCNDecoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=3):
        super(FCNDecoder, self).__init__()

        res_modules = []
        for _ in range(n_res):
            res_modules += [ResBlock(dim)]
        self.res_module = nn.Sequential(*res_modules)

        up_modules = []
        for i in range(ups):
            if i == ups - 2: stride_dim = 4
            elif i == ups - 1: stride_dim = 1
            else: stride_dim = 2
            up_modules += [
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    Conv2dBlock(dim * 2, dim // stride_dim, 3, 1, 1)
                )
            ]
            dim //= stride_dim
        self.up_modules = nn.ModuleList(up_modules)

        self.out_conv = Conv2dBlock(dim, out_dim, 3, 1, 1, norm_type='none', activation='tanh')

        self.ups = ups
    
    def forward(self, x, skip_connected_features):
        y = self.res_module(x)
        for i in range(self.ups):
            skip = skip_connected_features[self.ups-i-1]
            y = torch.cat([skip, y], dim=1)
            y = self.up_modules[i](y)
        y = self.out_conv(y)
        return y

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            Conv2dBlock(dim, dim, 3, 1, 1),
            Conv2dBlock(dim, dim, 3, 1, 1, activation='none')
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        out += x
        out = self.relu(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, norm_type='instance', activation='relu'):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_dim)
        elif norm_type == 'none':
            self.norm = None
        else:
            print(f'Unsupported norm type {norm_type}')
            raise NotImplementedError
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            print(f'Unsupported activation type {activation}')
            raise NotImplementedError
        
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x