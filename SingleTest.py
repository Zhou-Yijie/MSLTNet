import torch
import torch.nn as nn
import cv2 as cv
import torchvision
import torch.optim
import os
from pyramid_structure.Omi_LP import MSLT
from torch.utils.data import DataLoader
import dataset.ImageDataset as ImageDataset
import dataset.loaddata as loaddata
from torchvision import transforms
from batch_transformers import BatchRandomResolution, BatchToTensor, BatchRGBToYCbCr, YCbCrToRGB, BatchTestResolution
import pandas as pd
from skimage.io import imread, imshow
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as psnr
import ssl
import lpips


Train_transform = transforms.Compose([
    BatchToTensor(),
])

#'snapshots/Epoch49.pth'
class util_of_lpips():
    def __init__(self, net, use_gpu=True):

        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img1_path, img2_path):
        # Load images
        img0 = lpips.im2tensor(lpips.load_image(img1_path))  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(img2_path))

        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        dist01 = self.loss_fn.forward(img0, img1)
        return dist01
        
def Test(Test_root, label_path,epoch):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    testmodel = MSLT().cuda()
    model = nn.DataParallel(testmodel)
    testmodel.load_state_dict(torch.load('snapshots/'+epoch))
    lp = util_of_lpips('alex')

    time1 = 0
    count = 1
    evl1 = 0
    evl2 = 0
    evl3 = 0
    evl_lpipsvgg = 0
    evl_lpipsalex = 0

    # 数据集处理
    train_transform = transforms.Compose([
        BatchToTensor(),
    ])

    # 数据集路径
    # datapath = "./data/"
    # 构建数据集

    train_data = loaddata.ImageSeqDataset(csv_file=os.path.join(Test_root, 'test.txt'),
                                 Train_img_seq_dir=Test_root,
                                 Label_img_dir=label_path,
                                 Train_transform=train_transform,
                                 Label_transform=transforms.ToTensor(),
                                 randomlist=False)

    train_loader = DataLoader(train_data,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=1)
    with torch.no_grad():
        for step, sample_batched in enumerate(train_loader):
            test_image, label_image = sample_batched['Train'], sample_batched['Lable']
    
            test_image = test_image.squeeze(0).cuda()
            print(label_image.shape)
            print(test_image.shape)
    
            label_image = label_image.cuda()
    
            for index in range(5):
                start = time.time()
                out4 = testmodel(test_image[index].unsqueeze(0))
                end_time = (time.time() - start)
                time1 = time1 + end_time
                average_time = time1 / (count)
                result_path= "./singleresults/"
                if not os.path.exists(result_path):
                    os.makedirs(result_path)

                torchvision.utils.save_image(out4, result_path + str(step + 1) +"_"+str(index+1)+ ".jpg")
                image = cv.imread(result_path + str(step + 1) +"_"+str(index+1)+ ".jpg")
                label = cv.imread(label_path + str(step + 1) + ".jpg")
    
                print(image.shape)
                print(label.shape)
    
                evl1 = evl1 + ssim(image, label, multichannel=True)
                evl2 = evl2 + psnr(image, label)
                evl3 = evl3 + lp.calc_lpips(result_path + str(step + 1) +"_"+str(index+1)+ ".jpg", label_path + str(step + 1) + ".jpg")
                
                evl_ssim = evl1 / (count)
                evl_psnr = evl2 / (count)
                evl_lpipsvgg = evl3 / (count)
                count = count+1
    
                print("time", average_time)
                print("psnr", evl_psnr)
                print("ssim", evl_ssim)
                print("lpips",evl_lpipsvgg )
            f = "test_metrics.txt"
            with open(f, "w") as file:  # ”w"代表着每次运行都覆盖内容
                file.write("ssim="+ str(evl_ssim)+ "\n")
                file.write("psnr=" +str(evl_psnr) + "\n")
                file.write("lpips=" +str(evl_lpipsvgg) + "\n")

if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    test_path = './data/testimage/'
    label_path = "./data/testimage/label/"
    epochlist = []
    epochlist.append("mslt.pth")
    for epoch in epochlist:
        Test(test_path, label_path, epoch)
