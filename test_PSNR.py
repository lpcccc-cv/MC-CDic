import torch
from utils import util
from tqdm import tqdm
import cv2, os
import models.modules.MCCDic as MCCDic

import time

def main():
    mode = 'IXI'
    dataset_opt = {}
    dataset_opt['task'] = 'rec'
    dataset_opt['scale'] = 4
    dataset_opt['hr_in'] = True
    dataset_opt['crop_size'] = 0
    dataset_opt['test_size'] = 240
    data1 = True
    joint_rec = False
    save_result = True
    #### create train and val dataloader
    if mode == 'IXI':
        from data.IXI_dataset import IXI_train as D
        dataset_opt['dataroot_GT'] = '/home/lpc/dataset/IXI/MC_MRI/test/T2'
    elif mode == 'brain':
        from data.brain_dataset import brain_train as D
        dataset_opt['dataroot_GT'] = '/home/lpc/dataset/BrainTS/MCSR/T1_test'
    
    val_set = D(dataset_opt, train=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1,pin_memory=True)
    print('Number of val images: {:d}'.format(len(val_set)))

    # creat model

    model_path = '~~~~~~~~~~'
    model = MCCDic.MCCDic().cuda()

    model_params = util.get_model_total_params(model)
    print('Model_params: ', model_params)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    with torch.no_grad():
        #### validation
        if joint_rec:
            avg_psnr_im1 = 0.0
            avg_psnr_im2 = 0.0
            avg_ssim_im1 = 0.0
            avg_ssim_im2 = 0.0
            avg_rmse_im1 = 0.0
            avg_rmse_im2 = 0.0
            idx = 0
            for i, val_data in enumerate(tqdm(val_loader)): 
                im1_lr = val_data['im1_LQ'].cuda()
                im1_gt = val_data['im1_GT'].cuda()
                im2_lr = val_data['im2_LQ'].cuda()
                im2_gt = val_data['im2_GT'].cuda()
                mask = val_data['mask'].cuda()

                sr_img_1, sr_img_2= model(im1_lr, im2_lr, mask)
                
                sr_img_1 = sr_img_1[0,0].cpu().detach().numpy()*255.
                im1_gt = im1_gt[0,0].cpu().detach().numpy()*255.
                sr_img_2 = sr_img_2[0,0].cpu().detach().numpy()*255.
                im2_gt = im2_gt[0,0].cpu().detach().numpy()*255.
                # calculate PSNR
                cur_psnr_im1 = util.calculate_psnr(sr_img_1, im1_gt)
                avg_psnr_im1 += cur_psnr_im1
                cur_psnr_im2 = util.calculate_psnr(sr_img_2, im2_gt)
                avg_psnr_im2 += cur_psnr_im2
                # ssim
                cur_ssim_im1 = util.calculate_ssim(sr_img_1, im1_gt)
                avg_ssim_im1 += cur_ssim_im1
                cur_ssim_im2 = util.calculate_ssim(sr_img_2, im2_gt)
                avg_ssim_im2 += cur_ssim_im2
                # rmse
                cur_rmse_im1 = util.calculate_rmse(sr_img_1, im1_gt)
                avg_rmse_im1 += cur_rmse_im1
                cur_rmse_im2 = util.calculate_rmse(sr_img_2, im2_gt)
                avg_rmse_im2 += cur_rmse_im2
                
                idx += 1

            avg_psnr_im1 = avg_psnr_im1 / idx
            avg_psnr_im2 = avg_psnr_im2 / idx
            avg_ssim_im1 = avg_ssim_im1 / idx
            avg_ssim_im2 = avg_ssim_im2 / idx
            avg_rmse_im1 = avg_rmse_im1 / idx
            avg_rmse_im2 = avg_rmse_im2 / idx
            # log
            print("# image1 Validation # PSNR: {:.6f}".format(avg_psnr_im1))
            print("# image2 Validation # PSNR: {:.6f}".format(avg_psnr_im2))
            print("# image1 Validation # SSIM: {:.6f}".format(avg_ssim_im1))
            print("# image2 Validation # SSIM: {:.6f}".format(avg_ssim_im2))
            print("# image1 Validation # RMSE: {:.6f}".format(avg_rmse_im1))
            print("# image2 Validation # RMSE: {:.6f}".format(avg_rmse_im2))
        else:
            avg_psnr_im1 = 0.0
            avg_ssim_im1 = 0.0
            avg_rmse_im1 = 0.0
            idx = 0
            time_begin = time.time()
            for i,val_data in enumerate(tqdm(val_loader)):
                im1_lr = val_data['im1_LQ'].cuda()
                im1_gt = val_data['im1_GT'].cuda()
                im2_lr = val_data['im2_LQ'].cuda()
                im2_gt = val_data['im2_GT'].cuda()
                mask = val_data['mask'].cuda()

                if data1: 
                    sr_img_1 = model(im1_lr, im2_gt, mask)
                    sr_img_1 = sr_img_1[0,0].cpu().detach().numpy()*255.
                    im1_gt = im1_gt[0,0].cpu().detach().numpy()*255.
               
                    # calculate PSNR
                    cur_psnr_im1 = util.calculate_psnr(sr_img_1, im1_gt)
                    avg_psnr_im1 += cur_psnr_im1
                    cur_ssim_im1 = util.calculate_ssim(sr_img_1, im1_gt)
                    avg_ssim_im1 += cur_ssim_im1
                    cur_rmse_im1 = util.calculate_rmse(sr_img_1, im1_gt)
                    avg_rmse_im1 += cur_rmse_im1
                else:
                    sr_img_1= model(im2_lr, im1_gt, mask)
                    sr_img_1 = sr_img_1[0,0].cpu().detach().numpy()*255.
                    im1_gt = im2_gt[0,0].cpu().detach().numpy()*255.
                    # calculate PSNR
                    cur_psnr_im1 = util.calculate_psnr(sr_img_1, im1_gt)
                    avg_psnr_im1 += cur_psnr_im1
                    cur_ssim_im1 = util.calculate_ssim(sr_img_1, im1_gt)
                    avg_ssim_im1 += cur_ssim_im1
                    cur_rmse_im1 = util.calculate_rmse(sr_img_1, im1_gt)
                    avg_rmse_im1 += cur_rmse_im1

                # 保存图像
                save_path_1 = '/home/lpc/program/MC_MRI/test_result/IXI/rec/'+'IXI_guided_T2_MDUNet_no_align'
                if not os.path.exists(save_path_1):
                    os.makedirs(save_path_1)
                if save_result:
                    cv2.imwrite(os.path.join(save_path_1, '{:08d}.png'.format(i+1)), sr_img_1)


                idx += 1 
            time_end = time.time()   
            avg_psnr_im1 = avg_psnr_im1 / idx
            avg_ssim_im1 = avg_ssim_im1 / idx
            avg_rmse_im1 = avg_rmse_im1 / idx
            # log
            print("# image1 Validation # PSNR: {:.6f}".format(avg_psnr_im1))
            print("# image1 Validation # SSIM: {:.6f}".format(avg_ssim_im1))
            print("# image1 Validation # RMSE: {:.6f}".format(avg_rmse_im1))
            print('Total time:', time_end-time_begin)

### CUDA_VISIBLE_DEVICES=2 python test_PSNR.py
if __name__ == '__main__':
    main()
