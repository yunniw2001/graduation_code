import cv2
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from my_transformer import MeanFiltersTransform, MedianFiltersTransform, GaussFiltersTransform, \
    GaussianFiltersTransformUnsharpMask, MedianFiltersTransformUnsharpMask, MeanFiltersTransformUnsharpMask, MyDataset

from skimage.feature import hog

preprocessing = []
testprocessing = []
to_greyscale = None
batch_size = 50


def prepare_transform_for_image():
    global preprocessing
    global testprocessing
    global to_greyscale
    rotation = transforms.RandomRotation(5)
    resized_cropping = transforms.Resize((224, 224))
    contrast_brightness_adjustment = transforms.ColorJitter(brightness=0.5, contrast=0.5)
    smooth_or_sharpening = transforms.RandomChoice([
        MeanFiltersTransform(),
        MedianFiltersTransform(),
        GaussFiltersTransform(),
        GaussianFiltersTransformUnsharpMask(),
        MedianFiltersTransformUnsharpMask(),
        MeanFiltersTransformUnsharpMask()
    ])
    color_shift = transforms.ColorJitter(hue=0.14)
    preprocessing = transforms.Compose(
        [
            transforms.RandomApply(
                [rotation, contrast_brightness_adjustment, smooth_or_sharpening, color_shift], 0.6),
            resized_cropping,
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]
    )
    to_greyscale = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Grayscale(),
        ]
    )
    testprocessing = transforms.Compose(
        [resized_cropping,
         # transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])


class MyHogSVM:
    "Face Recognition for face using HOG+SVM"

    def __init__(self, name_dataset, dir_data, size_img, N_identity, N_training_img):
        super(MyHogSVM, self).__init__(name_dataset, dir_data, size_img, N_identity, N_training_img)

    def extract_HOG_img(self, img):
        HOG_cells = self.init_cell_img(img)
        HOG_blocks = self.combine_cells_to_block(HOG_cells)

        return HOG_blocks

    def init_cell_img(self, img, width_size=8, binsize=9):
        '''
        :param width_size: width and height of one cell
        :param binsize: number of orient
        :return: HOG of cells in a image
        '''
        self.width_size_cell = width_size
        self.binsize_cell = binsize
        height_size = width_size
        N_cells_height, N_cells_width = self.height // height_size, self.width // width_size
        # np.zeros可以生成任意大小的ndarray
        HOG_cells = np.zeros((N_cells_height, N_cells_width, binsize), dtype=np.int32)

        # cal histogram of cells
        for i in range(N_cells_height):
            # 标记要计算HOG的点
            row_cell = height_size * i
            for j in range(N_cells_width):
                col_cell = width_size * j
                HOG_cells[i, j] = self.cal_histogram_cell(
                    img[row_cell:row_cell + N_cells_width, col_cell:col_cell + N_cells_height])

        return HOG_cells

    def combine_cells_to_block(self, HOG_cells, width_size=2, height_size=2):
        assert (width_size == height_size)
        # block和cell一样都是堆叠的
        N_blocks_height, N_blocks_width = self.width_size_cell - width_size + 1, self.width_size_cell - height_size + 1
        self.width_size_block = width_size
        HOG_blocks = np.zeros((N_blocks_height, N_blocks_width, width_size * width_size * self.binsize_cell),
                              dtype=np.float32)
        for i in range(N_blocks_height):  # height 也就是行，秉承着先行后列的习惯，就先height后width了
            for j in range(N_blocks_width):
                # collection cells in a block
                # 把4个cell的HOG特征串联成新的block特征，LBP+h其实就是把图像分成了多个block
                HOG_block = HOG_cells[i:i + height_size, j:j + width_size].flatten().astype(np.float32)
                # L1归一化特征，这一步可能有点问题, 因为特征是向量，所有后续的所有东西都是向量或矩阵
                # 对光照和阴影获得更好的效果
                HOG_block /= np.sqrt(np.sum(abs(HOG_block)) + 1e-6)  # 防止0的平方根
                HOG_blocks[i, j] = HOG_block

        # 把blocks的HOG特征串联起来，就是整个图像的HOG特征
        self.size_HOG_feature = HOG_blocks.size
        # print('size of HOG for img is %d'%self.size_HOG_feature)
        return HOG_blocks.flatten()

    def cal_histogram_cell(self, img_cell):
        '''
        :param img_np: imread(path, 0), a cell of  gray image
        :return: the histogram of a cell
        '''
        img_np = img_cell
        #############cal value and angle of gradient###################
        dx = cv2.Sobel(img_np, cv2.CV_16S, 1, 0)
        dy = cv2.Sobel(img_np, cv2.CV_16S, 0, 1)
        # s is a small number avoiding /0
        s = 1e-3
        angle = np.int32(np.arctan(dy / (dx + s)) / np.pi * 180) + 90  # 为什么要加90

        # 将梯度转回uint8
        dy = cv2.convertScaleAbs(dy)
        dx = cv2.convertScaleAbs(dx)
        # 计算梯度大小，结合水平梯度和竖直梯度
        value_grd = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)
        # imshow('test_gra', value_grd)

        ###########collecting histogram of gradient#####################
        # hist is a 9D vector
        hist = np.zeros((self.binsize_cell), dtype=np.int32)
        # 180 or 360, 效果有什么不同
        step = 180 // self.binsize_cell
        seq_bins = angle // step
        seq_bins = seq_bins.flatten()
        value_grd = value_grd.flatten()
        for seq_i, val in zip(seq_bins.flatten(), value_grd.flatten()):
            hist[seq_i] += val
        return hist


prepare_transform_for_image()
dataset = 'IITD'
root_path = '/home/ubuntu/dataset/' + dataset + '/test_session/'
session1_dataset = MyDataset(root_path + 'session1/',
                             root_path + 'session1_label.txt', testprocessing)
session2_dataset = MyDataset(root_path + 'session2/',
                             root_path + 'session2_label.txt', testprocessing)
session1_dataloader = DataLoader(dataset=session1_dataset, batch_size=batch_size, shuffle=False)
session2_dataloader = DataLoader(dataset=session2_dataset, batch_size=batch_size, shuffle=True)
