import os
import torch
from pyramid_structure.Omi_LP import MSLT
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import dataset.ImageDataset as ImageDataset
import dataset.loaddata as loaddata
from batch_transformers import BatchRandomResolution, BatchToTensor, BatchRGBToYCbCr, YCbCrToRGB, BatchTestResolution
import argparse
from torch import nn, optim
from skimage.metrics import structural_similarity as ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as psnr
from pytorch_ssim.wavelet_ssim_loss import WSloss
import torchvision
import cv2 as cv
import thop

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = MSLT().cuda()
    # a = torch.zeros(1, 3, 3840, 2160).cuda()
    #a = torch.zeros(1, 3, 1024, 1024).cuda()
    a = torch.zeros(1, 3, 512, 512).cuda()
    flops, params = thop.profile(model,inputs=(a,))
    print("flops","params",flops,params)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(20):
        enhanced_image = model(a)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            torch.cuda.synchronize()
            starter.record()
            enhanced_image = model(a)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)


