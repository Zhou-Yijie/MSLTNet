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

from skimage.io import imread, imshow
import time

# ME test Dataset
img_env_no = 2 # folder number
img_exposure_no = 1 # actual name of the image
image_dir_path = f"./data/MEdata/ME_test/{img_env_no}/"

test_images_paths = [] # 5 Element 
for i in range(1, 6):
    image_path = image_dir_path + f"{i}.jpg"
    test_images_paths.append(image_path)

# print(test_images_paths)
# Sample Test Image
test_image = imread(test_images_paths[img_exposure_no-1])

# Convert image to tensor
test_image = torch.from_numpy(test_image)
print(test_image.shape)
print(type(test_image))
        
def Inference(test_image: torch.Tensor):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_model = MSLT().to(device)
    model = nn.DataParallel(test_model)

    # Load pretrained model
    test_model.load_state_dict(torch.load('pretrained_model/mslt+.pth'))

    
    test_image = test_image.squeeze(0).to(device)
    print(test_image.shape)
    


    start = time.time()
    index = 0 # TODO: check this line
    output_image = test_model(test_image[index].unsqueeze(0)) # TODO: check this line
    inference_time = time.time() - start
    
    result_path= "./sasika_singleresults/"

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    torchvision.utils.save_image(output_image, result_path + f"{img_env_no}_{img_exposure_no}_.jpg")

    # show image using matplotlib
    image = cv.imread(result_path + f"{img_env_no}_{img_exposure_no}_.jpg")
    # label = cv.imread(label_path + f"{img_env_no}.jpg")
    imshow(image)
    # imshow(label)

    print(image.shape)

Inference(test_image)