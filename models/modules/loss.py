from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fnn
from torch.autograd import Variable
import math

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y, mask):
        diff = torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff*mask))
        return loss

class ncc_loss(nn.Module):
    def __init__(self):
        super(ncc_loss, self).__init__()

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = fnn.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = fnn.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = fnn.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = fnn.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = fnn.conv2d(IJ, filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        return I_var, J_var, cross

    def forward(self, I, J, win=None):
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        if win is None:
            win = [9] * ndims
        else:
            win = win * ndims
        conv_fn = getattr(fnn, 'conv%dd' % ndims)
        I2 = I * I
        J2 = J * J
        IJ = I * J
        sum_filt = torch.ones([1, 1, *win]).to("cuda")
        pad_no = math.floor(win[0] / 2)
        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)
        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)
        cc = cross * cross / (I_var * J_var + 1e-5)

        return torch.mean(cc)


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w), 
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr

class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        
    def forward(self, input, target):
        # input shape :[B, N, C, H, W]
        if len(input.shape) == 5:
            B,N,C,H,W = input.size()
            input = input.view(-1, C, H , W)
            target = target.view(-1, C, H, W)
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma, 
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input  = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(fnn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))



# if __name__ == "__main__":  
