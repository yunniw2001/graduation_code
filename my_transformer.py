import os
import random
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
import cv2
import numpy
from torch.utils.data import Dataset


class MeanFiltersTransform:
    def __init__(self, p=1.0):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            tmp_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            tmp_img = cv2.blur(tmp_img, (1, 1))
            return Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        else:
            return img


class MeanFiltersTransformUnsharpMask:
    def __init__(self, p=1.0):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            tmp_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            blur_img = cv2.blur(tmp_img, (1, 1))
            tmp_img = tmp_img - blur_img + tmp_img
            return Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        else:
            return img


class MedianFiltersTransform:
    def __init__(self, p=1.0):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            tmp_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            tmp_img = cv2.medianBlur(tmp_img, 3)
            return Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        else:
            return img


class MedianFiltersTransformUnsharpMask:
    def __init__(self, p=1.0):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            tmp_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            blur_img = cv2.medianBlur(tmp_img, 3)
            tmp_img = tmp_img - blur_img + tmp_img
            return Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        else:
            return img


class GaussFiltersTransform:
    def __init__(self, p=1.0):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            tmp_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            tmp_img = cv2.GaussianBlur(tmp_img, (5, 5), sigmaX=0, sigmaY=0)
            return Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        else:
            return img


class GaussianFiltersTransformUnsharpMask:
    def __init__(self, p=1.0):
        assert isinstance(p, float)
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            tmp_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
            blur_img = cv2.GaussianBlur(tmp_img, (5, 5), sigmaX=0, sigmaY=0)
            tmp_img = tmp_img - blur_img + tmp_img
            return Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        else:
            return img


class FFT_converter:
    def __int__(self, len):
        self.p = len

    def __call__(self, img):
        # print(img)
        r, c = img.size
        blank_paper = np.full((r, c,3), 255, dtype=np.uint8)
        f = np.fft.fft2(img)
        f_paper = np.fft.fft2(blank_paper)
        fshift = np.fft.fftshift(f)
        fshift_blank = np.fft.fftshift(f_paper)
        center_r, center_c = int(r / 2), int(c / 2)
        # beta_r, beta_c = int(self.p * r), int(self.p * c)
        beta_r, beta_c = self.p,self.p
        fshift[center_r - beta_r:center_r + beta_r, center_c - beta_c:center_c + beta_c] = fshift_blank[
                                                                                           center_r - beta_r:center_r + beta_r,
                                                                                           center_c - beta_c:center_c + beta_c]
        ishift = np.fft.ifftshift(fshift)
        iimg = np.abs(np.fft.ifft2(ishift))
        return iimg


class MyDataset(Dataset):
    def __init__(self, img_path, txt_path, transform=None):
        super(MyDataset, self).__init__()
        self.img_path = img_path
        self.txt_path = txt_path
        f = open(self.txt_path, 'r')
        data = f.readlines()

        imgs = []
        labels = []
        for line in data:
            word = line.rstrip().split(' ')
            imgs.append(os.path.join(self.img_path, word[0]))
            labels.append(word[1])
        self.img = imgs
        self.label = labels
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]

        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label)

        return img, label
