import torch
import torch.nn as nn

# ==================================================


class DnCNN(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, n_block):
        self.n_block = n_block  # = 17 for non-blind, 20 for blind
        self.n_ch_unit = 64
        self.conv_ksize = 3
        self.batch_norm = True

        super().__init__()

        layers = []
        layers.append(nn.Conv2d(n_ch_in, self.n_ch_unit, self.conv_ksize,
                                padding=(self.conv_ksize // 2), groups=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(self.n_block - 2):
            layers.append(nn.Conv2d(self.n_ch_unit, self.n_ch_unit, self.conv_ksize,
                                    padding=(self.conv_ksize // 2),
                                    groups=1, bias=False))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(self.n_ch_unit))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(self.n_ch_unit, n_ch_out, self.conv_ksize,
                                padding=(self.conv_ksize // 2), groups=1, bias=True))

        self.body = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x):
        return x - self.body(x)

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

# ==================================================


class DnCNN_S(DnCNN):
    def __init__(self, n_ch_in=1, n_ch_out=1):
        super(DnCNN_S, self).__init__(n_ch_in, n_ch_out, n_block=17)


class DnCNN_B(DnCNN):
    def __init__(self, n_ch_in=1, n_ch_out=1):
        super(DnCNN_B, self).__init__(n_ch_in, n_ch_out, n_block=20)


class CDnCNN_S(DnCNN):
    def __init__(self, n_ch_in=3, n_ch_out=3):
        super(CDnCNN_S, self).__init__(n_ch_in, n_ch_out, n_block=17)


class CDnCNN_B(DnCNN):
    def __init__(self, n_ch_in=3, n_ch_out=3):
        super(CDnCNN_B, self).__init__(n_ch_in, n_ch_out, n_block=20)

# ==================================================


if __name__ == '__main__':
    x = torch.randn(2, 3, 64, 64)

    cdncnnb = CDnCNN_B(3, 3)
    print(cdncnnb(x).shape)
