import torch
import numpy as np
import contextlib
import cv2

@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)
def mask_func_random_unique(shape, acc = 8, seed=42):
    """
    Args:
    shape:[320, 320, 2]

    Return:
    [1, 320, 1]非0即1的tensor
    """
    if len(shape) < 3:
        raise ValueError("Shape should have 3 or more dimensions")
    
    rng = np.random
    with temp_seed(rng, seed):
        num_cols = shape[-2]
        if acc == 2:
            center_fraction, acceleration = 0.1, 2#中心采样比例，加速比
        elif acc == 4:
            center_fraction, acceleration = 0.08, 4#中心采样比例，加速比
        elif acc == 6:
            center_fraction, acceleration = 0.06, 6#中心采样比例，加速比
        elif acc == 8:
            center_fraction, acceleration = 0.04, 8#中心采样比例，加速比
        else:
            assert('accelerate rate is not implmented')

        # create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        #
        #(采样条数-中心采样条数) / 所有未采样条数 计算每一行的采样概率
        #
        prob = (num_cols / acceleration - num_low_freqs) / (
            num_cols - num_low_freqs)
        mask = rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad: pad + num_low_freqs] = True

        # reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32)) # mask.shape=[col, 1]
        mask = mask.repeat(shape[0], 1, 1)   
    return mask[:, :, 0]

###### generate mask for reconstruction ####
# control shape and acc to generate different masks
mask = mask_func_random_unique(shape=[240, 240, 2],acc=8)
cv2.imwrite('./mask_x8_brain.png', mask.numpy()*255)

###### generate mask for SR ######
def gen_mask_for_SR(size, scale):
    h,w = size
    mask = torch.zeros(size)
    lr_h = h//scale
    lr_w = w//scale
    top_left_h = h//2-lr_h//2
    top_left_w = w//2-lr_w//2
    mask[top_left_h:(top_left_h+lr_h), top_left_w:(top_left_w+lr_w)] = torch.ones(lr_h,lr_w)
    return mask

##### directly crop K-space data to get the LR image ######
# data: K-space data
# scale: down sampling scale
def crop_k_data(data, scale):
    _,h,w = data.shape
    lr_h = h//scale
    lr_w = w//scale
    top_left_h = h//2-lr_h//2
    top_left_w = w//2-lr_w//2
    croped_data = data[:, top_left_h:(top_left_h+lr_h), top_left_w:(top_left_w+lr_w)]
    return croped_data/scale**2
