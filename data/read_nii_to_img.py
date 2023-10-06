import os
import numpy as np
import cv2
import SimpleITK as sitk

# normalize to 0-255
def norm(data):
    data = data.astype(np.float32)
    # data = np.clip(data, a_min=-200, a_max=400)
    max = np.max(data)
    min = np.min(data)
    data = (data-min)/(max-min)
    return data*255.

# nii folder
file_path = '/home/lpc/dataset/IXI/IXI-PD'
# save image path
save_file_path = '/home/lpc/dataset/IXI/MCSR/test/PD'

step = 1   # sampling interval
skip = 10  # skip low-quality images at the begining and the end.
volume_num = 500  # set the first 500 volumes as the training data
totle_number=0  # total image number

file_list = sorted(os.listdir(file_path))

for number, name in enumerate(file_list):
    if 'IXI014' in name:
        continue  # skip it
    if number<volume_num:
        continue
    print(name)
    filename_1 = file_path+'/'+name
    img_1 = sitk.ReadImage(filename_1, sitk.sitkInt16)
    space = img_1.GetSpacing()
    img_1 = sitk.GetArrayFromImage(img_1)
    width, height, queue = img_1.shape
    # print(name, img.shape)
    # print(name, space)

    # normalize to 0-255
    data_1 = norm(img_1)
    for i in range(skip, width-skip, step):
        totle_number = totle_number+1
        img_arr1 = data_1[i, :, :]
        img_arr1 = np.expand_dims(img_arr1, axis=2)

        cv2.imwrite(save_file_path+'/{:08d}.png'.format(totle_number), img_arr1)
        print('Done!'+save_file_path+'/{:08d}.png'.format(totle_number))

