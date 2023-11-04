import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from skimage.io import imread
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import random

class ImageSeqDataset(Dataset):
	def __init__(self, csv_file,
				 img_dir,
				 label_dir,
				 transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			hr_img_seq_dir (string): Directory with all the high resolution image sequences.
			transform (callable, optional): transform to be applied on a sample.
		"""
		self.seqs = pd.read_csv(csv_file, sep='\n', header=None)
		self.image_root = img_dir
		self.label_root = label_dir
		self.transform = transform

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			samples: a Tensor that represents a video segment.
		"""
		train_image = imread(self.image_root + str(self.seqs.iloc[index, 0]) + ".JPG")
		label_image = imread(self.label_root + str(self.seqs.iloc[index, 0]) + ".JPG")

		if self.transform is not None:
			train_seq_image = self.transform(train_image)
			label_seq_image = self.transform(label_image)

		sample = {'Train': train_seq_image, 'Label': label_seq_image, 'index':str(self.seqs.iloc[index, 0])}
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
		transforms.ToTensor(),
	])

	train_data = ImageSeqDataset(csv_file=os.path.join("../cai/", 'train.txt'),
								 img_dir="../cai/train/",
								 label_dir="../cai/label/",
								 transform=train_transform)

	train_loader = DataLoader(train_data,
							  batch_size=1,
							  shuffle=False,
							  pin_memory=True,
							  num_workers=1)
	for step, sample_batched in enumerate(train_loader):
		train_image, label_image = sample_batched['Train'], sample_batched['Label']
		torchvision.utils.save_image(train_image, "../loaddata/"+ str(step) + "train.JPG")
		torchvision.utils.save_image(label_image, "../loaddata/" + str(step) + "label.JPG")