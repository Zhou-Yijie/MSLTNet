import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
from . import network
import time
import cv2 as cv

class Lap_Pyramid(nn.Module):
    def __init__(self):
        super(Lap_Pyramid, self).__init__()
        self.de_conv_1 = nn.Conv2d(3, 3, 3,padding=1, stride=2)
        self.de_conv_2 = nn.Conv2d(3, 3, 3,padding=1,stride=2)
        self.de_conv_3 = nn.Conv2d(3, 3, 3,padding=1,stride=2)
        self.re_cov_1 = nn.Conv2d(3, 3, 3, padding=1, stride=1)
        self.re_cov_2 = nn.Conv2d(3, 3, 3, padding=1, stride=1)
        self.re_cov_3 = nn.Conv2d(3, 3, 3, padding=1, stride=1)

    def de_cov(self, x):
        seq = []
        level_1 = self.de_conv_1(x)
        level_2 = self.de_conv_2(level_1)
        level_3 = self.de_conv_3(level_2)
        seq_1 = x - nn.functional.interpolate(self.re_cov_1(level_1), size=(x.shape[2], x.shape[3]),mode='bilinear')
        seq_2 = level_1 - nn.functional.interpolate(self.re_cov_2(level_2), size=(level_1.shape[2], level_1.shape[3]), mode='bilinear')
        seq_3 = level_2 - nn.functional.interpolate(self.re_cov_3(level_3), size=(level_2.shape[2], level_2.shape[3]), mode='bilinear')
        seq.append(level_3)
        seq.append(seq_3)
        seq.append(seq_2)
        seq.append(seq_1)
        return seq
    def pyramid_recons(self, pyr):
        rec_1 = nn.functional.interpolate(self.re_cov_3(pyr[0]), size=(pyr[1].shape[2], pyr[1].shape[3]), mode='bilinear')
        image = rec_1 + pyr[1]
        rec_2 = nn.functional.interpolate(self.re_cov_2(image), size=(pyr[2].shape[2], pyr[2].shape[3]), mode='bilinear')
        image = rec_2 + pyr[2]
        rec_3 = nn.functional.interpolate(self.re_cov_1(image), size=(pyr[3].shape[2], pyr[3].shape[3]), mode='bilinear')
        image = rec_3 + pyr[3]
        return image
        
class Trans_high(nn.Module):
    def __init__(self, num_high=3):
        super(Trans_high, self).__init__()

        self.model = nn.Sequential(*[nn.Conv2d(9, 9, 1,1), nn.LeakyReLU(), nn.Conv2d(9, 3, 1, 1)])
        #self.model = nn.Sequential(*[nn.Conv2d(9, 9, 3,1), nn.LeakyReLU(), nn.Conv2d(9, 3, 3, 1)])

        self.trans_mask_block_1 = nn.Sequential(*[nn.Conv2d(3, 3, 1,1), nn.LeakyReLU(), nn.Conv2d(3, 3, 1,1)])
        self.trans_mask_block_2 = nn.Sequential(*[nn.Conv2d(3, 3, 1,1), nn.LeakyReLU(), nn.Conv2d(3, 3, 1,1)])

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []
        pyr_result.append(fake_low)

        mask = self.model(x)

        result_highfreq_1 = torch.mul(pyr_original[1], mask)
        pyr_result.append(result_highfreq_1)

        mask_1 = nn.functional.interpolate(mask, size=(pyr_original[2].shape[2], pyr_original[2].shape[3]))
        mask_1 = self.trans_mask_block_1(mask_1)
        result_highfreq_2 = torch.mul(pyr_original[2], mask_1)
        pyr_result.append(result_highfreq_2)

        mask_2 = nn.functional.interpolate(mask, size=(pyr_original[3].shape[2], pyr_original[3].shape[3]))
        mask_2 = self.trans_mask_block_1(mask_2)
        result_highfreq_3 = torch.mul(pyr_original[3], mask_2)
        pyr_result.append(result_highfreq_3)
        return pyr_result

        
class MSLT(nn.Module):
    def __init__(self, nrb_high=1, num_high=3):
        super(MSLT, self).__init__()

        self.lap_pyramid = Lap_Pyramid()
        trans_low = network.B_transformer()
        trans_high = Trans_high() 
        self.trans_low = trans_low.cuda()
        self.trans_high = trans_high.cuda()

    def forward(self, real_A_full):
        pyr_A = self.lap_pyramid.de_cov(real_A_full)
        fake_B_low = self.trans_low(pyr_A[0])
        real_A_up = nn.functional.interpolate(pyr_A[0], size=(pyr_A[1].shape[2], pyr_A[1].shape[3]))
        fake_B_up = nn.functional.interpolate(fake_B_low, size=(pyr_A[1].shape[2], pyr_A[1].shape[3]))
        high_with_low = torch.cat([pyr_A[1], real_A_up, fake_B_up], 1)
        pyr_A_trans = self.trans_high(high_with_low, pyr_A, fake_B_low)
        fake_B_full = self.lap_pyramid.pyramid_recons(pyr_A_trans)
        return fake_B_full


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model  = LPTNPaper().cuda()
    #a = torch.randn(1, 3, 3840, 2160).cuda()
    a = torch.randn(1, 3, 1024, 1024).cuda()
    #a = torch.randn(1, 3, 512, 512).cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings=np.zeros((repetitions,1))
#GPU-WARM-UP
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
