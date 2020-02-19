import torch.nn as nn
import torch.nn.functional as F

from models.sean_networks.architecture import SEANResnetBlock as SEANResnetBlock  # fin, fout, opt, inject_st
from models.sean_networks.base_network import BaseNetwork


class SEANGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralseansyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SEANResnetBlock(16 * nf, 16 * nf, opt, inject_st=True)  # x, seg, st

        self.G_middle_0 = SEANResnetBlock(16 * nf, 16 * nf, opt, inject_st=True)
        self.G_middle_1 = SEANResnetBlock(16 * nf, 16 * nf, opt, inject_st=True)

        self.up_0 = SEANResnetBlock(16 * nf, 8 * nf, opt, inject_st=True)
        self.up_1 = SEANResnetBlock(8 * nf, 4 * nf, opt, inject_st=True)
        self.up_2 = SEANResnetBlock(4 * nf, 2 * nf, opt, inject_st=True)
        self.up_3 = SEANResnetBlock(2 * nf, 1 * nf, opt, inject_st=True)

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        num_up_layers = 6

        sw = opt.crop_size // (2 ** num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, st):
        seg = input
        # import ipdb
        # ipdb.set_trace()
        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)

        x = self.head_0(x, seg, st)

        x = self.up(x)
        x = self.G_middle_0(x, seg, st)

        x = self.up(x)

        x = self.G_middle_1(x, seg, st)

        x = self.up(x)
        x = self.up_0(x, seg, st)
        x = self.up(x)
        x = self.up_1(x, seg, st)
        x = self.up(x)
        x = self.up_2(x, seg, st)
        x = self.up(x)
        x = self.up_3(x, seg, st)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
