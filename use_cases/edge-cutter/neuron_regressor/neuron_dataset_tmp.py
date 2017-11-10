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

    def __len__(self):
        return len(self.label)*10

    def getcrop(self, idx):
        cropped_image_list = []
        count_list = []
        image = Image.open(os.path.join(self.image_dir, '{}.png'.format(int(idx/10))))
        for i in range(0, 150):
            random_crop = RandomCrop(size = 64)
            cropped_image, center_i, center_j = random_crop(image)

            box = self.label[int(idx/10)]
            bool_i = [np.abs(center_i-co[0]) < 32 for co in box]
            bool_j = [np.abs(center_j-co[1]) < 32 for co in box]
            count = float(sum(bool_i and bool_j))
            cropped_image_list.append(cropped_image)
            count_list.append(count)

        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image_list, count_list

    def getitem(self, idx):
        
        '''
        @input: list of cropped_image and list of count
        @output: 10 crop images and corresponding count neurons
        '''
        
        # upsample distribution
        cnt_0 = 2 
        cnt_1 = 4
        cnt_2 = 2
        cnt_3 = 2
        
        # to choose 10 crops sample
        cropped_image = []
        count = []

        # 150 crops 
        cropped_image_list, count_list = self.getcrop(idx)
        indices = [i for i,x in enumerate(count_list) if x >= 3]
        if len(indices) >= 2: 
            # more than two count greater than 3
            cropped_image.append(cropped_image_list[indices[0]])
            cropped_image.append(cropped_image_list[indices[1]])
            count.append(count_list[indices[0]])
            count.append(count_list[indices[1]])
            cnt_3 = 0 
        elif len(indices) == 1: 
            # only one count greater than 3
            # append twice
            cropped_image.append(cropped_image_list[indices[0]])
            cropped_image.append(cropped_image_list[indices[0]])
            count.append(count_list[indices[0]])
            count.append(count_list[indices[0]])
            cnt_3 = 0
        else:
            # if no count greater than 3
            # to use cnt_2 crop 
            cnt_2 = 4
        
        counter = 0 
        while True:        
            index = random.randint(0, len(cropped_image_list))
            if cnt_0 != 0 and count_list[index] == 0:
                cnt_0 -= 1
                cropped_image.append(cropped_image_list[index])
                count.append(count_list[index])
                
            elif cnt_1 != 0 and count_list[index] == 1:
                cnt_1 -= 1
                cropped_image.append(cropped_image_list[index])
                count.append(count_list[index])
                
            elif cnt_2 != 0 and count_list[index] == 2:
                cnt_2 -= 1
                cropped_image.append(cropped_image_list[index])
                count.append(count_list[index])
            
            elif cnt_0 == 0 and cnt_1 == 0 and cnt_2 == 0 and cnt_3 == 0:
                return cropped_image, count
            else:
                if counter > 300:
                    for i in range(0, 10-len(cropped_image)):
                        cropped_image += [cropped_image_list[index]]
                        count += [count_list[index]]
                    return cropped_image, count
                    break
                else:
                    continue

    def __getitem__(self, idx):
        cropped_image, count = self.getitem(idx)
        index = random.randint(0, len(cropped_image))
        return cropped_image[index], count[index]