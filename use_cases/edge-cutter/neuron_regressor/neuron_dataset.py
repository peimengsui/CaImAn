import pickle
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numbers
import random 
from skimage import io
import os

def pad(img, padding, fill=0):
	"""Pad the given PIL Image on all sides with the given "pad" value.
	Args:
		img (PIL Image): Image to be padded.
		padding (int or tuple): Padding on each border. If a single int is provided this
			is used to pad all borders. If tuple of length 2 is provided this is the padding
			on left/right and top/bottom respectively. If a tuple of length 4 is provided
			this is the padding for the left, top, right and bottom borders
			respectively.
		fill: Pixel fill value. Default is 0. If a tuple of
			length 3, it is used to fill R, G, B channels respectively.
	Returns:
		PIL Image: Padded image.
	"""

	if not isinstance(padding, (numbers.Number, tuple)):
		raise TypeError('Got inappropriate padding arg')
	if not isinstance(fill, (numbers.Number, str, tuple)):
		raise TypeError('Got inappropriate fill arg')

	if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
		raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
						 "{} element tuple".format(len(padding)))

	return ImageOps.expand(img, border=padding, fill=fill)


def crop(img, i, j, h, w):
	"""Crop the given PIL Image.
	Args:
		img (PIL Image): Image to be cropped.
		i: Upper pixel coordinate.
		j: Left pixel coordinate.
		h: Height of the cropped image.
		w: Width of the cropped image.
	Returns:
		PIL Image: Cropped image.
	"""

	return img.crop((j, i, j + w, i + h))

class RandomCrop(object):
	"""Crop the given PIL Image at a random location.
	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
		padding (int or sequence, optional): Optional padding on each border
			of the image. Default is 0, i.e no padding. If a sequence of length
			4 is provided, it is used to pad left, top, right, bottom borders
			respectively.
	"""

	def __init__(self, size, padding=0):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.padding = padding

	@staticmethod
	def get_params(img, output_size):
		"""Get parameters for ``crop`` for a random crop.
		Args:
			img (PIL Image): Image to be cropped.
			output_size (tuple): Expected output size of the crop.
		Returns:
			tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
		"""
		w, h = img.size
		th, tw = output_size
		if w == tw and h == th:
			return 0, 0, h, w

		i = random.randint(0, h - th)
		j = random.randint(0, w - tw)
		return i, j, th, tw

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be cropped.
		Returns:
			PIL Image: Cropped image.
		"""
		if self.padding > 0:
			img = pad(img, self.padding)

		i, j, h, w = self.get_params(img, self.size)

		return crop(img, i, j, h, w), i, j

class NeuronDataset(Dataset):
	"""Neuron dataset."""

	def __init__(self, label_file, image_dir, transform=None):
		"""
		Args:
			label_file (string): Path to the label file.
			frame_file (string): Path to the image file
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.label = pickle.load(open(label_file, "rb"))
		self.transform = transform
		self.image_dir = image_dir


		self.cnt_0 = int(64*0.2)
		self.cnt_1 = int(64*0.4)
		self.cnt_2 = int(64*0.2)
		self.cnt_3 = 64 - self.cnt_0 - self.cnt_1 - self.cnt_2
		self.total_return = 0
		
	def __len__(self):
		return len(self.label)

	def getcrop(self, idx):
		
		while True:
			image = Image.open(os.path.join(self.image_dir, '{}.png'.format(idx)))
		
			img_arr = np.array(image)
		

			random_crop = RandomCrop(size = 64)
			cropped_image, center_i, center_j = random_crop(image)

			box = self.label[idx]
		
			if len(box) == 0:
				idx = np.random.randint(0, 5000)

				continue 
			bool_i = [np.abs(center_i-co[0]) < 32 for co in box]
			bool_j = [np.abs(center_j-co[1]) < 32 for co in box]
			count = float(sum(bool_i and bool_j))
		


			if self.transform:
				cropped_image = self.transform(cropped_image)

			return cropped_image, count

	def __getitem__(self, idx):
		
		'''
		@input: list of cropped_image and list of count
		@output: 10 crop images and corresponding count neurons
		'''
		#cropped_image, count = self.getcrop(idx)
		#print('count={}'.format(count))	
		#import pdb
		#pdb.set_trace()
		iteration = 0	
		while True:  
			if iteration < 300:    
				cropped_image, count = self.getcrop(idx)
			else:
				cropped_image, count = self.getcrop(np.random.randint(0,self.__len__()))
				if count = 0:
					continue
			iteration += 1
			#print('count={}'.format(count))
			#print('cnt_0={}'.format(self.cnt_0))
			#print('cnt_1={}'.format(self.cnt_1))
			#print('cnt_2={}'.format(self.cnt_2))
			#print('cnt_3={}'.format(self.cnt_3))
			#print(self.total_return)	
			if self.cnt_0 ==0 and self.cnt_1 ==0 and self.cnt_2 ==0 and self.cnt_3 ==0:
				self.cnt_0 = int(64*0.2)
				self.cnt_1 = int(64*0.4)
				self.cnt_2 = int(64*0.2)
				self.cnt_3 = 64 - self.cnt_0 - self.cnt_1 - self.cnt_2	


			if self.cnt_0 ==0 and self.cnt_1 == 0:
				if self.cnt_2 > 0:
					cropped_image, count = self.get_cnt_2_3(n=2)
					#self.cnt_2 -= 1
					self.total_return += 1
					return cropped_image, count
				else:
					cropped_image, count = self.get_cnt_2_3(n=3)
					#self.cnt_3 -= 1
					self.total_return += 1
					return cropped_image, count
	
			if count == 0:
				if self.cnt_0 > 0:
					self.cnt_0 -= 1
					self.total_return += 1
					return cropped_image, count

			elif count == 1:
				if self.cnt_1 > 0:
					self.cnt_1 -= 1
					self.total_return += 1
					return cropped_image, count

			elif count == 2:
				if self.cnt_2 > 0:
					self.cnt_2 -= 1
					self.total_return += 1
					return cropped_image, count

			elif count > 2:
				if self.cnt_3 > 0:
					self.cnt_3 -= 1
					self.total_return += 1
					return cropped_image, count

			else:
				pass




	def get_cnt_2_3(self, n = 2):

		while True:

			random_idx = np.random.randint(0, self.__len__())
			image = Image.open(os.path.join(self.image_dir, '{}.png'.format(random_idx)))
			image_arr = np.array(image)
			loc = self.label[random_idx]

			crop_list,crop_neual_cnt_list = self.crop_count_neural(image_arr, loc, 63)
			#print(max(crop_neual_cnt_list))
			if len(crop_neual_cnt_list)>0 and max(crop_neual_cnt_list) >= n:
			
				#idx_n = crop_neual_cnt_list.index(n)
				idx_n = next(x[0] for x in enumerate(crop_neual_cnt_list) if x[1] > n-1)
				crop_n = crop_list[idx_n]
				im = Image.fromarray(crop_n.astype('uint8'))
				#im.save('temp.png')
				if n == 2:
					self.cnt_2 -= 1
				else:
					self.cnt_3 -= 1
				return self.transform(Image.fromarray(crop_n.astype('uint8'))), crop_neual_cnt_list[idx_n]
				#return Image.open('temp.png'), crop_neual_cnt_list[idx_n]
			else:
				continue



	def crop_count_neural(self, image_frame, bbox_center_location, crop_size):


		def find_bounding_loc(center_index, crop_size = crop_size, image_frame = image_frame):
			'''
			@input: center_index, list of location
			@return: bounding_index, list-of-list [left_bot,left_top,right_bot,right_top]
			@logic: 
				1. given center_index location, return four corner location 
				2. truncate if boundary location out of image
			'''
			tmp = int(crop_size/2)

			left_bot_x = (center_index[0]- tmp) if (center_index[0]- tmp) >= 0 else 0 
			left_top_x = (center_index[0]- tmp) if (center_index[0]- tmp) >= 0 else 0 

			left_bot_y = (center_index[1]- tmp) if (center_index[1]- tmp) >= 0 else 0
			right_bot_y = (center_index[1]- tmp) if (center_index[1]- tmp) >= 0 else 0

			right_bot_x = (center_index[0]+ tmp) if (center_index[0]+ tmp) <= (image_frame.shape[0]-1) else (image_frame.shape[0]-1)
			right_top_x = (center_index[0]+ tmp) if (center_index[0]+ tmp) <= (image_frame.shape[0]-1) else (image_frame.shape[0]-1)

			left_top_y = (center_index[1]+ tmp) if (center_index[1]+ tmp) <= (image_frame.shape[1]-1) else (image_frame.shape[1]-1)  
			right_top_y = (center_index[1]+ tmp) if (center_index[1]+ tmp) <= (image_frame.shape[1]-1) else (image_frame.shape[1]-1)


			left_bot = [left_bot_x, left_bot_y]
			left_top = [left_top_x, left_top_y]
			right_bot = [right_bot_x, right_bot_y]
			right_top = [right_top_x, right_top_y]

			bounding_loc = [left_bot,left_top,right_bot,right_top]

			bounding_loc = [tuple(l) for l in bounding_loc]

			return bounding_loc

		def count_neural(location, bbox_center_location):
			'''
			count how many neurals in a bounding location
			'''
			neural_cnt = 0
			for bbox in bbox_center_location:
				if bbox[0] >= location[0][0] and bbox[0] <= location[2][0] and bbox[1] >= location[0][1] and bbox[1] <= location[1][1]:
					neural_cnt += 1
			return neural_cnt


		bbox_center_location = [tuple(l) for l in bbox_center_location]
		crop_loc_list = list(map(find_bounding_loc, bbox_center_location))
		crop_dict = dict(zip(bbox_center_location, crop_loc_list))
		
		crop_neural_dict = {} 
		for crop_key in crop_dict.keys():
			loction = crop_dict[crop_key]
			neural_cnt = count_neural(loction, bbox_center_location)
			crop_neural_dict[crop_key] = neural_cnt

		# deal with truncated crop, which is out of orginal image frame
		crop_blank_frame = np.zeros((crop_size+1)**2).reshape(crop_size+1,crop_size+1)
		crop_blank_frame = crop_blank_frame.astype(int)

		crop_list = []
		crop_neual_cnt_list = []
		for crop_key in crop_dict.keys():

			crop = image_frame[crop_dict[crop_key][0][0]:crop_dict[crop_key][2][0]+1, crop_dict[crop_key][0][1]:crop_dict[crop_key][1][1]+1]
			if crop.shape != crop_blank_frame.shape:
				tmp = crop_blank_frame.copy()
				tmp[0:crop.shape[0], 0:crop.shape[1]] = crop
				crop = tmp

			crop_neual_cnt = crop_neural_dict[crop_key]
			crop_list.append(crop)
			crop_neual_cnt_list.append(crop_neual_cnt)
		
		
		return (crop_list,crop_neual_cnt_list)
