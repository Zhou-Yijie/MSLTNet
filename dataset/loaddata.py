import os
import functools
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.io import imread, imshow
from batch_transformers import BatchRandomResolution, BatchToTensor, BatchRGBToYCbCr, YCbCrToRGB, BatchTestResolution
from torch.utils.data import Dataset
import random

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def randomlist(list):
	int = random.randint(1, 10)
	if int < len(list):
		newlist = random.sample(list, int)
	else:
		newlist = []
		for i in range(int):
			newlist.append(random.choice(list))
	return(newlist)


def has_file_allowed_extension(filename, extensions):
	"""Checks if a file is an allowed extension.
	Args:
		filename (string): path to a file
		extensions (iterable of strings): extensions to consider (lowercase)
	Returns:
		bool: True if the filename ends with one of given extensions
	"""
	filename_lower = filename.lower()
	return any(filename_lower.endswith(ext) for ext in extensions)


def image_seq_loader(img_seq_dir):
	img_seq_dir = os.path.expanduser(img_seq_dir)

	img_seq = []
	for root, _, fnames in sorted(os.walk(img_seq_dir)):
		for fname in sorted(fnames):
			if has_file_allowed_extension(fname, IMG_EXTENSIONS):
				image_name = os.path.join(root, fname)
				image = imread(image_name)
				img_seq.append(image)

	return img_seq


def get_default_img_seq_loader():
	return functools.partial(image_seq_loader)



class ImageSeqDataset(Dataset):
	def __init__(self, csv_file,
				 Train_img_seq_dir,
				 Label_img_dir,
				 Train_transform=None,
				 Label_transform=None,
				 get_loader=get_default_img_seq_loader,
				 randomlist = True):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			hr_img_seq_dir (string): Directory with all the high resolution image sequences.
			transform (callable, optional): transform to be applied on a sample.
		"""
		self.seqs = pd.read_csv(csv_file, sep='\n', header=None)
		self.Train_root = Train_img_seq_dir
		self.Label_img_dir = Label_img_dir
		self.Train_transform = Train_transform
		self.Label_transform = Label_transform
		self.loader = get_loader()
		self.randomlist = randomlist

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			samples: a Tensor that represents a video segment.
		"""
		Train_seq_dir = os.path.join(self.Train_root, str(self.seqs.iloc[index, 0]))
		I = self.loader(Train_seq_dir)
		if self.randomlist == True:
			I = randomlist(I)


		'''
		for i in range(4):
			cv2.imshow("seq" + str(i), seqs_2[i])
			cv2.waitKey()
		'''
		I = self.Train_transform(I)

		train = torch.stack(I, 0).contiguous()


		Label_image = imread(self.Label_img_dir + str(self.seqs.iloc[index, 0]) + ".jpg")

		Label = self.Label_transform(Label_image)


		sample = {'Train': train, 'Lable': Label}
		return sample

	def __len__(self):
		return len(self.seqs)

	@staticmethod
	def _reorderBylum(seq):
		I = torch.sum(torch.sum(torch.sum(seq, 1), 1), 1)
		_, index = torch.sort(I)
		result = seq[index, :]
		return result



if __name__=='__main__':
	# 数据集处理
	train_transform = transforms.Compose([
		BatchToTensor(),
	])

	train_data = ImageSeqDataset(csv_file=os.path.join("../traindata/trainimage/", 'train.txt'),
								 Train_img_seq_dir="../traindata//trainimage/",
								 Label_img_dir="../traindata/trainimage/label/",
								 Train_transform=train_transform,
								 Label_transform=transforms.ToTensor())

	train_loader = DataLoader(train_data,
							  batch_size=1,
							  shuffle=False,
							  pin_memory=True,
							  num_workers=1)
	for step, sample_batched in enumerate(train_loader):
		train_image, label_image = sample_batched['Train'], sample_batched['Lable']
		print("train", train_image.shape)
		print("train", label_image.shape)

