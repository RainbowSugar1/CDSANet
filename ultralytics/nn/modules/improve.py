import torch
from torch import nn
from torch.nn import init

from ultralytics.nn.modules.block import C2f




class EFAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()

        # x_c
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

        # x_s
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # x_c
        y1 = self.gap(x)
        y1 = y1.squeeze(-1).permute(0, 2, 1)
        y1 = self.conv(y1)
        y1 = self.sigmoid(y1)
        y1 = y1.permute(0, 2, 1).unsqueeze(-1)
        x_c = x * y1.expand_as(x)

        # x_s
        q = self.Conv1x1(x)
        q = self.norm(q)
        x_s = x * q
        return x_c + x_s


class C2f_EFAttention(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(EFAttention(self.c) for _ in range(n))

### by CSDN    AI  Little monster    https://blog.csdn.net/m0_63774211?type=lately

import torch
import torch.nn as nn


from ultralytics.nn.modules.conv import Conv, autopad





class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class SPPELAN1(nn.Module):
    # spp-elan
    def __init__(self, c1, c2, c3):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))
