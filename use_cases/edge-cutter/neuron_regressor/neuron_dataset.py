import pickle
import torchvision
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numbers
import random 
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

    def __init__(self, label_file, frame_file, transform=None):
        """
        Args:
            label_file (string): Path to the label file.
            frame_file (string): Path to the image file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label = pickle.load(open(label_file, "rb"))
        self.frame = np.load(frame_file)['arr_0']
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = Image.fromarray(self.frame[idx,:,:], 'L')
        random_crop = RandomCrop(size = 64)
        cropped_image, center_i, center_j = random_crop(image)

        box = self.label[idx]
        bool_i = [np.abs(center_i-co[0]) < 30 for co in box]
        bool_j = [np.abs(center_j-co[1]) < 30 for co in box]
        count = float(sum(bool_i and bool_j))

        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, count
