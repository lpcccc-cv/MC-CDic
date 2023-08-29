import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ResidualBlock_noBN_noAct(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_noAct, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResBlock_fft_bench(nn.Module):
    def __init__(self, n_feat=64, norm='backward'): # 'ortho'
        super(ResBlock_fft_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=True),
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=True),
            BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = n_feat
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    print(x.size()[-2:])
    print(flow.size()[1:3])
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


class GridNet(nn.Module):
    def __init__(self):
        super(GridNet, self).__init__()
        self.lateral = nn.ModuleList([
            nn.ModuleList([
                LateralBlock(24),
                LateralBlock(24),
                LateralBlock(24),
                LateralBlock(24),
                LateralBlock(24),
                LateralBlock(24),
            ]),
            nn.ModuleList([
                LateralBlock(48),
                LateralBlock(48),
                LateralBlock(48),
                LateralBlock(48),
                LateralBlock(48),
            ]),
            nn.ModuleList([
                LateralBlock(96),
                LateralBlock(96),
                LateralBlock(96),
                LateralBlock(96),
                LateralBlock(96),
            ])
        ])
        self.down = nn.ModuleList([
            nn.ModuleList([
                DownBlock(24, 48),
                DownBlock(24, 48),
                DownBlock(24, 48),
            ]),
            nn.ModuleList([
                DownBlock(48, 96),
                DownBlock(48, 96),
                DownBlock(48, 96),
            ])
        ])
        self.up = nn.ModuleList([
            nn.ModuleList([
                UpBlock(48, 24),
                UpBlock(48, 24),
                UpBlock(48, 24),
            ]),
            nn.ModuleList([
                UpBlock(96, 48),
                UpBlock(96, 48),
                UpBlock(96, 48),
            ])
        ])
        self.compress = nn.ModuleList([
            nn.Conv2d(24*3, 24, 3, 1, 1),
            nn.Conv2d(48*3, 48, 3, 1, 1),
            nn.Conv2d(96*3, 96, 3, 1, 1)
        ])
        self.to_RGB = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(24, 1, 3, 1, 1)
        )

    def forward(self, pyramid):
        x_0_0, x_1_0, x_2_0 = pyramid

        # compress dim 32 x 2 -> 32
        x_0_0 = self.compress[0](x_0_0)
        x_1_0 = self.compress[1](x_1_0)
        x_2_0 = self.compress[2](x_2_0)

        # first half: down & lateral
        x_0_1 = self.lateral[0][0](x_0_0)
        x_0_2 = self.lateral[0][1](x_0_1)
        x_0_3 = self.lateral[0][2](x_0_2)

        x_1_0 = x_1_0 + self.down[0][0](x_0_0)
        x_2_0 = x_2_0 + self.down[1][0](x_1_0)

        x_1_1 = self.lateral[1][0](x_1_0)
        x_2_1 = self.lateral[2][0](x_2_0)

        x_1_1 = x_1_1 + self.down[0][1](x_0_1)
        x_2_1 = x_2_1 + self.down[1][1](x_1_1)

        x_1_2 = self.lateral[1][1](x_1_1)
        x_2_2 = self.lateral[2][1](x_2_1)

        x_1_2 = x_1_2 + self.down[0][2](x_0_2)
        x_2_2 = x_2_2 + self.down[1][2](x_1_2)

        x_1_3 = self.lateral[1][2](x_1_2)
        x_2_3 = self.lateral[2][2](x_2_2)

        # second half: up & lateral
        x_2_4 = self.lateral[2][3](x_2_3)
        x_2_5 = self.lateral[2][4](x_2_4)

        x_1_3 = x_1_3 + self.up[1][0](x_2_3)
        x_0_3 = x_0_3 + self.up[0][0](x_1_3)

        x_1_4 = self.lateral[1][3](x_1_3)
        x_0_4 = self.lateral[0][3](x_0_3)

        x_1_4 = x_1_4 + self.up[1][1](x_2_4)
        x_0_4 = x_0_4 + self.up[0][1](x_1_4)

        x_1_5 = self.lateral[1][4](x_1_4)
        x_0_5 = self.lateral[0][4](x_0_4)

        x_1_5 = x_1_5 + self.up[1][2](x_2_5)
        x_0_5 = x_0_5 + self.up[0][2](x_1_5)

        # final synthesis
        output = self.lateral[0][5](x_0_5)
        output = self.to_RGB(output)

        return output



class LateralBlock(nn.Module):
    def __init__(self, dim):
        super(LateralBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )

    def forward(self, x):
        res = x
        x = self.layers(x)
        return x + res


class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DownBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_dim, out_dim, 3, 2, 1),
            nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UpBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        return self.layers(x)

