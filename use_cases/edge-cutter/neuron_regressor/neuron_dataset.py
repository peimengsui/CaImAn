import pickle
import torchvision
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F

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
            img = F.pad(img, self.padding)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), i, j

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
        image = Image.fromarray(self.frame[idx,:,:])
        crop = RandomCrop(size = 64)
        cropped_image, center_i, center_j = image.crop()
        cropped_image = np.asarray(cropped_image)

        box = self.label[idx]
        bool_i = [(np.abs(center_i, co[0] < 30)) for co in box]
        bool_j = [(np.abs(center_j, co[1] < 30)) for co in box]
        count = sum(bool_i and bool_j)

        if self.transform:
            sample = self.transform(sample)

        return cropped_image, count