# Deep Unfolding Convolutional Dictionary Model for Multi-Contrast MRI Super-resolution and Reconstruction (IJCAI2023)
The official PyTorch implementation of MC-CDic (IJCAI 2023).

Authors: Pengcheng Lei, Faming Fang, Guixu Zhang and Ming Xu.

## Highlights
Magnetic resonance imaging (MRI) tasks often involve multiple contrasts. Recently, numerous deep learning-based multi-contrast MRI super-resolution (SR) and reconstruction methods have been proposed to explore the complementary information from the multi-contrast images. However, these methods either construct parameter-sharing networks or manually design fusion rules, failing to accurately model the correlations between multi-contrast images and lacking certain interpretations. In this paper, we propose a multi-contrast convolutional dictionary (MC-CDic) model under the guidance of the optimization algorithm with a well-designed data fidelity term. Specifically, we bulid an observation model for the multi-contrast MR images to explicitly model the multi-contrast images as common features and unique features. In this way, only the useful information in the reference image can be transferred to the target image, while the inconsistent information will be ignored. We employ the proximal gradient algorithm to optimize the model and unroll the iterative steps into a deep CDic model. Especially, the proximal operators are replaced by learnable ResNet. In addition, multi-scale dictionaries are introduced to further improve the model performance. We test our MC-CDic model on multi-contrast MRI SR and reconstruction tasks. Experimental results demonstrate the superior performance of the proposed MC-CDic model against existing SOTA methods.

## Environment
pytorch version >= 1.8

### 1. Parparing the datasets:
The two publicly available multi-modal MR image datasets IXI and BraTS2018 can be downloaded at:
 [[IXI dataset]](https://brain-development.org/ixi-dataset/) and  [[BrainTS dataset]](http://www.braintumorsegmentation.org/).    
(1) The original data are _**.nii**_ data. Split your data set into training sets, validation sets, and test sets;  
(2) Read _**.nii**_ data and save these slices as **_.png_** images into two different folders as:
```bash
[T1 folder:]
000001.png,  000002.png,  000003.png,  000004.png ...
[T2 folder:]
000001.png,  000002.png,  000003.png,  000004.png ...
# Note that the images in the T1 and T2 folders correspond one to one.
```
(3) The random undersampling masks can be generated by **[data/generate_mask_random.py]**. Undersampled images are automatically generated in the dataloader process. 
### 2. Model training: 
Modify the data set path and training parameters in **[configs/modelx4.yaml]**, then run
```bash
sh train.sh
```

### 3. Model test:

Modify the test configurations in Python file **[test_psnr.py]**. Then run:
```bash
CUDA_VISIBLE_DEVICES=0 python test_PSNR.py
```
## Acknowledgement
Our code is built based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [MUSC](https://github.com/liutianlin0121/MUSC), thank them for releasing their code!
