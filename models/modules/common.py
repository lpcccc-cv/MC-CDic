import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # if init == 'xavier':
        #     mutil.initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        # else:
        #     mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        # mutil.initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Conv_up(nn.Module):
    def __init__(self, c_in, mid_c, up_factor):
        super(Conv_up, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=mid_c, kernel_size=3, padding=3 // 2), nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        conv = default_conv
        ## x3 00
        ## x2 11
        if up_factor == 2:
            modules_tail = [
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=up_factor, padding=1, output_padding=1),
                nn.Upsample(scale_factor=2),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 3:
            modules_tail = [
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=up_factor, padding=0, output_padding=0),
                nn.Upsample(scale_factor=3),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 4:
            modules_tail = [
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Upsample(scale_factor=4),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):

        out = self.body(input)
        out = self.tail(out)
        return out


class Conv_down(nn.Module):
    def __init__(self, c_in,mid_c, up_factor):
        super(Conv_down, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=mid_c, kernel_size=3, padding=3 // 2), nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        conv = default_conv
        if up_factor == 4:
            modules_tail = [
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
                nn.MaxPool2d(4),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 3:
            modules_tail = [
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=up_factor),
                nn.MaxPool2d(3),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]

        elif up_factor == 2:
            modules_tail = [
                # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=up_factor),
                nn.MaxPool2d(2),
                conv(mid_c, c_in,3),
                conv(c_in, c_in, 3)]
                
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):

        out = self.body(input)
        out = self.tail(out)
        return out

class HinResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(HinResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(feature // 2, affine=True)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))

        out_1, out_2 = torch.chunk(residual, 2, dim=1)
        residual = torch.cat([self.norm(out_1), out_2], dim=1)

        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        elif net_structure == 'Resnet':
            return ResBlock(channel_in, channel_out)
        elif net_structure == 'HinResnet':
            return HinResBlock(channel_in, channel_out)
        else:
            return None
    return constructor

class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num1, channel_num2, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_num1  # 1
        self.split_len2 = channel_num2  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        #in_channels = 3
        # self.invconv = InvertibleConv1x1(channel_num, LU_decomposed=True)
        # self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    # def forward(self, x, rev=False):
    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            # invert1x1conv
            # x, logdet = self.flow_permutation(x, logdet=0, rev=False)

            # split to 1 channel and 2 channel.
            # x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

            y1 = x1 + self.F(x2)  # 1 channel
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
            
        else:
            # split.
            # x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

            # x = torch.cat((y1, y2), 1)
            # print("rev_inn")
            # inv permutation
            # out = x
        out = torch.cat((y1, y2), 1)
        return out