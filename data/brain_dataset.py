import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os
import random


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors.
    Args:
        x (torch.Tensor): A PyTorch tensor.
        shift (int): Amount to roll.
        dim (int): Which dimension to roll.
    Returns:
        torch.Tensor: Rolled version of x.
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=(-2,-1)):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    Args:
        x (torch.Tensor): A PyTorch tensor.
        dim (int): Which dimension to fftshift.
    Returns:
        torch.Tensor: fftshifted version of x.
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]

    return roll(x, shift, dim)

def real_to_complex(img):
    if len(img.shape)==3:
        data = img.unsqueeze(0)
    else:
        data = img
    y = torch.fft.fft2(data)
    y = fftshift(y, dim=(-2,-1))  ## (1,1,h,w)
    y_complex = torch.cat([y.real, y.imag], 1)  ## (1,2,h,w)
    if len(img.shape)==3:
        y_complex = y_complex[0]
    return y_complex

def complex_to_real(data):
    if len(data.shape)==3:
        data1 = data.unsqueeze(0)
    else:
        data1 = data
    h, w = data.shape[-2], data.shape[-1]
    y_real, y_imag = torch.chunk(data1, 2, dim=1)
    y = torch.complex(y_real, y_imag)
    y = fftshift(y, dim=(-2,-1))  ## (1,1,h,w)
    y = torch.fft.irfft2(y,s=(h,w))
    # y = torch.fft.ifft2(y,s=(h,w)).abs()
    if len(data.shape)==3:
        y = y[0]
    return y

def crop_k_data(data, scale):
    _,h,w = data.shape
    lr_h = h//scale
    lr_w = w//scale
    top_left_h = h//2-lr_h//2
    top_left_w = w//2-lr_w//2
    croped_data = data[:, top_left_h:(top_left_h+lr_h), top_left_w:(top_left_w+lr_w)]
    return croped_data

class brain_train(data.Dataset):
    def __init__(self, opt, train):
        super(brain_train, self).__init__()
        path = opt['dataroot_GT']
        GT_list = sorted(os.listdir(path))
        self.GT_paths = [os.path.join(path, i) for i in GT_list]
        self.train = train
        self.crop_size = opt['crop_size']
        self.task = opt['task']
        self.scale = int(opt['scale'])
        self.hr_in = opt['hr_in']

    def __len__(self):
        if self.train:
            return len(self.GT_paths)
        else:
            return len(self.GT_paths)
            # return 200

    def __getitem__(self, idx):

        GT_img_path = self.GT_paths[idx]
        ref_GT_img_path = self.GT_paths[idx].replace('T1', 'T2')
        if self.task == 'rec':
            if self.scale == 4:
                mask_path = '/home/lpc/dataset/IXI/MC_MRI/mask_x4.png'
            elif self.scale == 8:
                mask_path = '/home/lpc/dataset/IXI/MC_MRI/mask_x8.png'
            else:
                print('Wrong scale for reconstruction!')
        elif self.task == 'sr':
            if self.scale == 2:
                mask_path = '/home/lpc/dataset/BrainTS/mask_sr_x2.png'
            elif self.scale == 4:
                mask_path = '/home/lpc/dataset/BrainTS/mask_sr_x4.png'
            else:
                print('Wrong scale for SR!')

        # read image file
        im1_GT = cv2.imread(GT_img_path, cv2.IMREAD_UNCHANGED)
        im2_GT = cv2.imread(ref_GT_img_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        im1_GT = torch.tensor(im1_GT).unsqueeze(0).float()/255.
        im2_GT = torch.tensor(im2_GT).unsqueeze(0).float()/255. 
        mask = torch.tensor(mask).unsqueeze(0).float().repeat(2,1,1)/255

        # FFT
        im1_GT_k = real_to_complex(im1_GT)
        im2_GT_k = real_to_complex(im2_GT)
        # apply mask
        # zero-padding
        if self.hr_in:
            im1_LQ_k = im1_GT_k * mask
            im2_LQ_k = im2_GT_k * mask
        # center_crop
        else:
            im1_LQ_k = crop_k_data(im1_GT_k, self.scale)
            im2_LQ_k = crop_k_data(im2_GT_k, self.scale)
        # IFFT
        im1_LQ = complex_to_real(im1_LQ_k)
        im2_LQ = complex_to_real(im2_LQ_k)

        # crop
        if self.train:
            _, H, W = im1_GT.shape 
            if self.hr_in:
                rnd_h = random.randint(0, max(0, H - self.crop_size))
                rnd_w = random.randint(0, max(0, W - self.crop_size))
                im1_GT = im1_GT[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
                im2_GT = im2_GT[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
                im1_LQ = im1_LQ[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
                im2_LQ = im2_LQ[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
            else:
                rnd_h = random.randint(0, max(0, H//self.scale - self.crop_size))
                rnd_w = random.randint(0, max(0, W//self.scale - self.crop_size))
                rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale), int(rnd_w * self.scale)
                im1_LQ = im1_LQ[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
                im2_LQ = im2_LQ[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
                im1_GT = im1_GT[:, rnd_h_HR:rnd_h_HR + self.crop_size*self.scale, rnd_w_HR:rnd_w_HR + self.crop_size*self.scale]
                im2_GT = im2_GT[:, rnd_h_HR:rnd_h_HR + self.crop_size*self.scale, rnd_w_HR:rnd_w_HR + self.crop_size*self.scale]
                

        return {'im1_LQ':im1_LQ, 'im1_GT':im1_GT, 'im2_LQ':im2_LQ, 'im2_GT':im2_GT, 'mask':mask}
        
