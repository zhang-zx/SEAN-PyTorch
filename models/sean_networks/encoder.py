import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sean_networks.base_network import BaseNetwork
from models.sean_networks.normalization import get_nonspade_norm_layer


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.nef  # ndf=32
        # ndf = 32
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        # norm_layer = get_nonspade_norm_layer(opt, 'spectralinstance')
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=1, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.ConvTranspose2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))
        self.tanh = nn.Tanh()
        self.pool = RegionWiseAvgPooling()

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x, mask):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.tanh(x)
        out = self.pool(feature_map=x, mask=mask)

        return out


class RegionWiseAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature_map, mask):
        if mask.size(2) != feature_map.size(2) or mask.size(3) != feature_map.size(3):
            mask = F.interpolate(mask, size=(feature_map.size(2), feature_map.size(3)), mode='bilinear',
                                 align_corners=True)
            mask = (mask >= 0.5).type_as(mask)
        out = list()
        for i in range(mask.size(1)):
            region_mask = torch.cat([mask[:, i, :, :].unsqueeze(1)] * feature_map.size(1), dim=1)
            out.append(self.avg_pool(region_mask * feature_map).squeeze(2).squeeze(2).unsqueeze(1))
        return torch.cat(out, dim=1)


if __name__ == '__main__':
    opt = None
    encoder = ConvEncoder(opt)
    x = torch.randn((3, 3, 256, 256))
    mask = torch.zeros((3, 10, 256, 256))
    mask[:, :, 128:, :] = 1
    out = encoder(x, mask)
    print(1)
