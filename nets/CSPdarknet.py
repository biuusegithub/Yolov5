import torch
import torch.nn as nn


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p


class Focus(nn.Module):
    # 在一张图片中每隔一个像素拿到一个值，这个时候获得了四个独立的特征层，然后将四个独立的特征层进行堆叠
    # 此时宽高信息就集中到了通道信息，输入通道扩充了四倍
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act_fn="silu") :
        super().__init__()
        self.conv = Conv_block(in_channels*4, out_channels, kernel_size, stride=1, groups=groups)

    def forward(self, x):
        x_1 = x[..., ::2, ::2]
        x_2 = x[..., 1::2, ::2]
        x_3 = x[..., ::2, 1::2]
        x_4 = x[..., 1::2, 1::2]

        return self.conv(torch.cat((x_1, x_2, x_3, x_4), dim=1))


class Conv_block(nn.Module):
    # 卷积+标准化+激活 => 作用是特征整合
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, p=None, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, p), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        # 不做 batch_norms
        return self.act(self.conv(x))


# 可选择的残差连接
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, g=1, scale=0.5):  
        super().__init__()
        num_hiddens = int(out_channels * scale)  
        self.cv1 = Conv_block(in_channels, num_hiddens, 1, 1)
        self.cv2 = Conv_block(num_hiddens, out_channels, 3, 1, groups=g)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        if self.add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, g=1, scale=0.5):  
        super().__init__()
        num_hiddens = int(out_channels * scale)  
        self.cv1 = Conv_block(in_channels, num_hiddens, 1, 1)
        self.cv2 = Conv_block(in_channels, num_hiddens, 1, 1)
        self.cv3 = Conv_block(2 * num_hiddens, out_channels, 1) 
        self.m = nn.Sequential(*[Bottleneck(num_hiddens, num_hiddens, shortcut, g, scale=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.m(self.cv1(x))
        x2 = self.cv2(x)
        y = torch.cat((x1, x2), dim=1)
        return self.cv3(y)


class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  
        self.cv1 = Conv_block(c1, c_, 1, 1)
        self.cv2 = Conv_block(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class CSPDarknet(nn.Module):
    def __init__(self, base_channels, base_depth):
        super().__init__()

        # Focus
        self.stem = Focus(3, base_channels, kernel_size=3)

        # dark2
        self.dark2 = nn.Sequential(
            Conv_block(base_channels, base_channels*2, 3, 2),
            CSPLayer(base_channels*2, base_channels*2, n=base_depth),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv_block(base_channels*2, base_channels*4, 3, 2),
            CSPLayer(base_channels*4, base_channels*4, n=base_depth*3),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv_block(base_channels*4, base_channels*8, 3, 2),
            CSPLayer(base_channels*8, base_channels*8, n=base_depth*3),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv_block(base_channels*8, base_channels*16, 3, 2),
            SPP(base_channels*16, base_channels*16),
            CSPLayer(base_channels*16, base_channels*16, n=base_depth, shortcut=False),
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.dark2(x)

        x = self.dark3(x)
        feat1 = x

        x = self.dark4(x)
        feat2 = x

        x = self.dark5(x)
        feat3 = x

        return feat1, feat2, feat3


if __name__ == '__main__':

    x = torch.rand((1, 3, 640, 640))
    net = CSPDarknet(64, 3)
    y1, y2, y3 = net(x)

    print(y1.shape)
    print(y2.shape)
    print(y3.shape)