import os
import copy
import torch
import torchvision
import numpy as np
import torch.nn as nn

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
            self.norm = nn.InstanceNorm2d(out_channels)
            
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

class VGG16(nn.Module):
    def __init__(self, pretrained_path):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16()
        vgg16.load_state_dict(torch.load(pretrained_path))

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        for i in range(3):
            for param in getattr(self, f'enc_{i+1:d}').parameters():
                param.requires_grad = False 

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, f'enc_{i+1:d}')
            results.append(func(results[-1]))
        return results[1:]

def build_lateral_connection(input_dim):
    return nn.Sequential(
        nn.Conv2d(input_dim, input_dim, 1, 1, 0),
        nn.Conv2d(input_dim, input_dim*2, 3, 1, 1),
        nn.Conv2d(input_dim*2, input_dim*2, 3, 1, 1),
        nn.Conv2d(input_dim*2, input_dim, 1, 1, 0)
    )

def get_activation(type):
    if type == 'leaky relu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif type == 'relu':
        return nn.ReLU(inplace=True)
    elif type == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError

def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

def build_feature_extractor(args):
    pretrained_path = os.path.join(args.code_dir, 'pretrained/vgg16-397923af.pth')
    return VGG16(pretrained_path)

