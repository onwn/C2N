import torch
import torch.distributions as torch_distb
import torch.nn as nn
import torch.nn.functional as F

d = 1e-6


class ResBlock(nn.Module):
    def __init__(self, n_ch_in, ksize=3, bias=True):
        self.n_ch = n_ch_in
        self.bias = bias

        super().__init__()

        layer = []
        layer.append(nn.Conv2d(self.n_ch, self.n_ch, ksize,
                               padding=(ksize // 2), bias=self.bias,
                               padding_mode='reflect'))
        layer.append(nn.PReLU())
        layer.append(nn.Conv2d(self.n_ch, self.n_ch, ksize,
                               padding=(ksize // 2), bias=self.bias,
                               padding_mode='reflect'))

        self.body = nn.Sequential(*layer)

    def forward(self, x):
        return x + self.body(x)


class C2N_D(nn.Module):
    def __init__(self, n_ch_in):
        self.n_ch_unit = 64

        super().__init__()

        self.n_block = 6

        self.n_ch_in = n_ch_in
        self.head = nn.Sequential(
            nn.Conv2d(self.n_ch_in, self.n_ch_unit, 3,
                      padding=1, bias=True,
                      padding_mode='reflect'),
            nn.PReLU()
        )

        layers = [ResBlock(self.n_ch_unit) for _ in range(self.n_block)]
        self.body = nn.Sequential(*layers)

        self.tail = nn.Conv2d(self.n_ch_unit, 1, 3,
                              padding=1, bias=True,
                              padding_mode='reflect')

    def forward(self, b_img_Gout):
        (N, C, H, W) = b_img_Gout.size()

        y = self.head(b_img_Gout)
        y = self.body(y)
        y = self.tail(y)

        return y


class C2N_G(nn.Module):
    def __init__(self, n_ch_in=3, n_ch_out=3, n_r=32):
        self.n_ch_unit = 64         # number of base channel
        self.n_ext = 5              # number of residual blocks in feature extractor
        self.n_block_indep = 3      # number of residual blocks in independent module
        self.n_block_dep = 2        # number of residual blocks in dependent module

        self.n_ch_in = n_ch_in      # number of input channels
        self.n_ch_out = n_ch_out    # number of output channels
        self.n_r = n_r        # length of r vector

        super().__init__()

        # feature extractor
        self.ext_head = nn.Sequential(
            nn.Conv2d(n_ch_in, self.n_ch_unit, 3,
                      padding=1, bias=True,
                      padding_mode='reflect'),
            nn.PReLU(),
            nn.Conv2d(self.n_ch_unit, self.n_ch_unit * 2, 3,
                      padding=1, bias=True,
                      padding_mode='reflect')
        )
        self.ext_merge = nn.Sequential(
            nn.Conv2d((self.n_ch_unit * 2) + self.n_r, 2 * self.n_ch_unit, 3,
                      padding=1, bias=True,
                      padding_mode='reflect'),
            nn.PReLU()
        )
        self.ext = nn.Sequential(
            *[ResBlock(2 * self.n_ch_unit) for _ in range(self.n_ext)]
        )

        # pipe-indep
        self.indep_merge = nn.Conv2d(self.n_ch_unit, self.n_ch_unit, 1,
                                     padding=0, bias=True,
                                     padding_mode='reflect')
        self.pipe_indep_1 = nn.Sequential(
            *[ResBlock(self.n_ch_unit, ksize=1, bias=False)
                for _ in range(self.n_block_indep)]
        )
        self.pipe_indep_3 = nn.Sequential(
            *[ResBlock(self.n_ch_unit, ksize=3, bias=False)
                for _ in range(self.n_block_indep)]
        )

        # pipe-dep
        self.dep_merge = nn.Conv2d(self.n_ch_unit, self.n_ch_unit, 1,
                                   padding=0, bias=True,
                                   padding_mode='reflect')
        self.pipe_dep_1 = nn.Sequential(
            *[ResBlock(self.n_ch_unit, ksize=1, bias=False)
                for _ in range(self.n_block_dep)]
        )
        self.pipe_dep_3 = nn.Sequential(
            *[ResBlock(self.n_ch_unit, ksize=3, bias=False)
                for _ in range(self.n_block_dep)]
        )

        # T tail
        self.T_tail = nn.Conv2d(self.n_ch_unit, self.n_ch_out, 1,
                                padding=0, bias=True,
                                padding_mode='reflect')

    def forward(self, x, r_vector=None):
        (N, C, H, W) = x.size()

        # r map
        if r_vector is None:
            r_vector = torch.randn(N, self.n_r)
        r_map = r_vector.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        r_map = r_map.float().detach()
        r_map = r_map.to(x.device)

        # feat extractor
        feat_CL = self.ext_head(x)
        list_cat = [feat_CL, r_map]
        feat_CL = self.ext_merge(torch.cat(list_cat, 1))
        feat_CL = self.ext(feat_CL)

        # make initial dep noise feature
        normal_scale = F.relu(feat_CL[:, self.n_ch_unit:, :, :]) + d
        get_feat_dep = torch_distb.Normal(loc=feat_CL[:, :self.n_ch_unit, :, :],
                                          scale=normal_scale)
        feat_noise_dep = get_feat_dep.rsample().to(x.device)

        # make initial indep noise feature
        feat_noise_indep = torch.rand_like(feat_noise_dep, requires_grad=True)
        feat_noise_indep = feat_noise_indep.to(x.device)

        # =====

        # pipe-indep
        list_cat = [feat_noise_indep]
        feat_noise_indep = self.indep_merge(torch.cat(list_cat, 1))
        feat_noise_indep = self.pipe_indep_1(feat_noise_indep) + \
            self.pipe_indep_3(feat_noise_indep)

        # pipe-dep
        list_cat = [feat_noise_dep]
        feat_noise_dep = self.dep_merge(torch.cat(list_cat, 1))
        feat_noise_dep = self.pipe_dep_1(feat_noise_dep) + \
            self.pipe_dep_3(feat_noise_dep)

        feat_noise = feat_noise_indep + feat_noise_dep
        noise = self.T_tail(feat_noise)

        return x + noise


if __name__ == '__main__':
    x = torch.randn(2, 3, 64, 64)

    c2nd = C2N_D(3)
    print(c2nd(x).shape)
    c2ng = C2N_G(3, 3, 32)
    print(c2ng(x).shape)
