import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn.parameter import Parameter

from models.sean_networks.sync_batchnorm import SynchronizedBatchNorm2d


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class SEAN(nn.Module):
    def __init__(self, config_text='seansyncbatch3x3', label_nc=10, norm_nc=32, len_latent=512, inject_st=True):
        super().__init__()
        assert config_text.startswith('sean')
        parsed = re.search('sean(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.len_latent = len_latent
        self.inject_st = inject_st

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        pw = ks // 2
        if self.inject_st:
            self.A_i_j = nn.Conv2d(label_nc, label_nc, kernel_size=1, padding=0)
            self.mlp_gamma_s = nn.Conv2d(self.len_latent, norm_nc, kernel_size=ks, padding=pw)
            self.mlp_beta_s = nn.Conv2d(self.len_latent, norm_nc, kernel_size=ks, padding=pw)

            self.alpha_beta = Parameter(torch.rand(1), requires_grad=True)
            self.alpha_gamma = Parameter(torch.rand(1), requires_grad=True)
        nhidden = 128
        self.mlp_mask = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma_o = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta_o = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, st, segmap, x):

        assert self.len_latent == st.size(2) and st.size(1) == segmap.size(1)

        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_mask(segmap)

        beta_o = self.mlp_beta_o(actv)
        gamma_o = self.mlp_gamma_o(actv)
        if self.inject_st:
            st = self.A_i_j(st.unsqueeze(3))
            st = st.expand(st.size(0), st.size(1), st.size(2), segmap.size(3)).permute(0, 3, 2, 1)
            style_map = st.matmul(segmap.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            beta_s = self.mlp_beta_s(style_map)
            gamma_s = self.mlp_gamma_s(style_map)

            gamma = self.alpha_gamma * gamma_s + (1. - self.alpha_gamma) * gamma_o
            beta = self.alpha_beta * beta_s + (1. - self.alpha_beta) * beta_o
            out = normalized * (1 + gamma) + beta
        else:
            out = normalized * (1 + gamma_o) + beta_o
        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


if __name__ == '__main__':
    x = torch.randn((4, 5, 32, 32))
    mask = torch.randn((4, 10, 256, 256))
    st = torch.randn((4, 10, 512))
    sean = SEAN(config_text='seanbatch3x3', label_nc=10, norm_nc=5, len_latent=512)
    out = sean(st, mask, x)
    print(1)
