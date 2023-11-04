import os
import torch
from pyramid_structure.Omi_LP import MSLT
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


def train(config):
    model = MSLT()
    #model = nn.DataParallel(model, device_ids=[0, 1, 2])  
    model = model.cuda()
    if config.load_pretrain == True:
        model.load_state_dict(torch.load(config.pretrain_dir))
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    valid_transform = transforms.Compose([
        BatchToTensor(),
    ])
    
    
    l1_loss=nn.L1Loss(reduction='mean').cuda()
    MSE_loss=nn.MSELoss(reduction='mean').cuda()
    wsloss = WSloss().cuda()

    train_data = ImageDataset.ImageSeqDataset(csv_file=os.path.join(config.datapath, 'test.txt'),
                                 img_dir=config.datapath,
                                 transform=train_transform)
    train_loader = DataLoader(train_data,
                              batch_size=config.train_batch_size,
                              num_workers=config.num_workers,
                              pin_memory=True,
                              shuffle=True)

    valid_data = loaddata.ImageSeqDataset(csv_file=os.path.join(config.validpath, 'test.txt'),
                                 Train_img_seq_dir=config.validpath,
                                 Label_img_dir=config.validlabel,
                                 Train_transform=valid_transform,
                                 Label_transform=transforms.ToTensor(),
                                 randomlist=False)
    
    valid_loader = DataLoader(valid_data,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=1)
    iters = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=iters*2, T_mult=1)


    for epoch in range(config.num_epochs):
        for step, sample_batched in enumerate(train_loader):
            train_image, label_image = sample_batched['train'], sample_batched['label']
            train_image = train_image.cuda()
            label_image = label_image.cuda()
            optimizer.zero_grad()
 
            #content = torch.exp(content)
            
            output = model(train_image)
            total_loss = MSE_loss(output , label_image)
            total_loss.backward()
            optimizer.step()
            scheduler.step(epoch + step / iters)

            if ((step + 1) % config.display_iter) == 0:
                print("Loss at iteration", step + 1, ":", total_loss.item())
            if ((step + 1) % config.snapshot_iter) == 0:
                torch.save(model.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
                f = "loss.txt"
                with open(f, "a") as file:
                    file.write("epoch="+str(epoch)+"loss="+str(total_loss.item())+"lr="+str(optimizer.param_groups[0]['lr'])+ "\n")
        if epoch > 0 and epoch % 5 == 0:
            time1 = 0
            count = 1
            evl1 = 0
            evl2 = 0
            evl_lpipsvgg = 0
            evl_lpipsalex = 0
            for step, sample_batched in enumerate(valid_loader):
                test_image, label_image = sample_batched['Train'], sample_batched['Lable']
                test_image = test_image.squeeze(0).cuda()
                print(label_image.shape)
                print(test_image.shape)

                label_image = label_image.cuda()

                for index in range(5):
                    out4 = model(test_image[index].unsqueeze(0))
                    validate_path = "./validation/"
                    if not os.path.exists(validate_path):
                        os.makedirs(validate_path)

                    torchvision.utils.save_image(out4, validate_path + str(step + 1) + "_" + str(index + 1) + ".jpg")
                    image = cv.imread(validate_path + str(step + 1) + "_" + str(index + 1) + ".jpg")
                    label = cv.imread("/home/ubuntu/liangjin/SEC/validation/label/" + str(step + 1) + ".jpg")

                    print(image.shape)
                    print(label.shape)

                    evl1 = evl1 + ssim(image, label, multichannel=True)
                    evl2 = evl2 + psnr(image, label)
                    evl_ssim = evl1 / (count)
                    evl_psnr = evl2 / (count)

                    count = count + 1

                    print("psnr", evl_psnr)
                    print("ssim", evl_ssim)
            f = "./valid_record/valid.txt"
            if not os.path.exists("./valid_record/"):
                os.makedirs("./valid_record/")
            with open(f, "a") as file:  # ”w"代表着每次运行都覆盖内容
                file.write("epoch="+str(epoch)+"_"+"ssim="+str(evl_ssim)+"psnr="+str(evl_psnr)+"\n")





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--validpath', type=str, default="./data/validation/")
    parser.add_argument('--validlabel', type=str, default="./data/validation/label/")
    parser.add_argument('--datapath', type=str, default="./data/imagepatch_512/")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.9)
    parser.add_argument('--grad_clip_norm', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_dir', type=str, default= 'snapshots131/Epoch131.pth')

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    train(config)
