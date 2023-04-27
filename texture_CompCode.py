import sys

import cv2
from scipy.signal import convolve2d
import numpy as np
import torchvision.transforms as transforms
from my_transformer import MeanFiltersTransform, MedianFiltersTransform, GaussFiltersTransform, \
    GaussianFiltersTransformUnsharpMask, MedianFiltersTransformUnsharpMask, MeanFiltersTransformUnsharpMask
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from sklearn.decomposition import PCA

preprocessing = None
testprocessing = None
my_gabor_filter = None


def prepare_transform_for_image():
    global preprocessing
    global testprocessing
    rotation = transforms.RandomRotation(5)
    resized_cropping = transforms.Resize((32, 32))
    contrast_brightness_adjustment = transforms.ColorJitter(brightness=0.5, contrast=0.5)
    color_shift = transforms.ColorJitter(hue=0.14)
    preprocessing = transforms.Compose(
        [
            transforms.RandomApply(
                [rotation, contrast_brightness_adjustment, color_shift], 0.6),
            MedianFiltersTransform(),
            resized_cropping,
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]
    )
    testprocessing = transforms.Compose(
        [MedianFiltersTransform(),
            resized_cropping,
         transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)
         ]
    )

class Gabor_filters:
    num_filters = 6
    num_points = 35
    sigma = 1.7
    filters = []
    frequency = 0.01
    band = np.pi/6


    def build_filters(self):
        for theta in range(self.num_filters):
            theta = theta/6. * np.pi
            kernel = np.real(gabor_kernel(self.frequency, theta=theta,bandwidth = self.band,
                                          sigma_x=self.sigma, sigma_y=self.sigma))
            self.filters.append(kernel)
        plt.figure(1)
        for temp in range(len(self.filters)):
            plt.subplot(2,3,temp+1)
            plt.imshow(self.filters[temp])
        plt.show()

    def extract_CompCode(self, image,show = False):
        gabor_responses = []
        for k,kernel in enumerate(self.filters):
            filtered = ndi.convolve(image,kernel,mode='wrap')
            gabor_responses.append(filtered)
        gabor_responses = np.array(gabor_responses)
        if show:
            plt.figure(2)
            for temp in range(len(gabor_responses)):
                plt.subplot(2,3,temp+1)
                plt.imshow(gabor_responses[temp],cmap='gray')
            plt.show()
        winner = np.argmin(gabor_responses,axis=0)
        # 标准化
        winner = (winner - np.max(np.max(winner))) * -1
        output = (winner / np.max(np.max(winner))) * 6
        # output = winner
        return output.reshape(1, -1)[0]

    def power(self,image):
        res = []
        for kernel in self.filters:
            res.append(np.sqrt(ndi.convolve(image,np.real(kernel),mode='wrap')**2+ndi.convolve(image,np.imag(kernel),mode='wrap')**2))
        plt.figure(3)
        for temp in range(len(res)):
            plt.subplot(2, 3, temp + 1)
            plt.imshow(res[temp], cmap='gray')
        plt.show()

my_gabor_filter = Gabor_filters()
my_gabor_filter.build_filters()




def read_image_and_label(labelpath, imgpath, state='train'):
    label_file = open(labelpath, 'r')
    content = label_file.readlines()
    code_g = []
    labels = []
    num = 0

    # print('===start read images and labels, generate uniform LBP gallery!===')
    for line in content:
        img_name, img_label = line.split(' ')[0], int(line.split(' ')[1])
        tmp_image_path = imgpath + img_name
        # print(tmp_image_path)
        if state == 'train':
            cur = testprocessing(Image.open(tmp_image_path).convert('L'))
        else:
            cur = testprocessing(Image.open(tmp_image_path).convert('L'))
        # print(cur.size())
        cur = cur.numpy()[0]
        if num ==0 and state == 'train':
            cur_code = my_gabor_filter.extract_CompCode(cur,True)
            my_gabor_filter.power(cur)
            num+=1
        else:
            cur_code = my_gabor_filter.extract_CompCode(cur)
        # print(cur_code.shape)
        code_g.append(cur_code)
        labels.append(img_label)
        num+=1
        if num%100 == 0:
            print('%d images Done!'%num)
    return code_g, labels


dataset = 'CASIA'
# 读入图像
img_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session1/'
label_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session1_label.txt'
save_gallery = '/home/ubuntu/graduation_model/gallery_texture.npy'
save_label = '/home/ubuntu/graduation_model/label_texture.npy'
prepare_transform_for_image()

label_file = open(label_PATH, 'r')
content = label_file.readlines()
code_gallery = []
already_processed = False

print('===start read images and labels, generate gallery!===')
if not already_processed:
    code_gallery, palmlabel = read_image_and_label(label_PATH, img_PATH)
    code_gallery = np.array(code_gallery)
    palmlabel = np.array(palmlabel)
    np.save(save_gallery,code_gallery)
    np.save(save_label,palmlabel)
else:
    code_gallery = np.load(save_gallery)
    palmlabel = np.load(save_label)
# # print(len(lbp_gallery))
# # print(image.shape)
# # plt.imshow(lbp,'gray')
# # plt.show()
# print('===DONE!===')
#
# 测试
print('===start test!===')
testimg_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session2/'
testlabel_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session2_label.txt'
print('===start load test image!===')
test_code_gallery, test_labels = read_image_and_label(testlabel_PATH, testimg_PATH,'test')
print(len(test_code_gallery))
test_code_gallery = np.array(test_code_gallery)
print('===load success! generate code success!===')

print('===start PCA!===')
# pca = PCA(n_components=200)

# print(code_gallery.shape)
# print(code_gallery.shape)
# code_gallery = pca.fit_transform(code_gallery)
# test_code_gallery = pca.fit_transform(test_code_gallery)
print(code_gallery.shape)
print(test_code_gallery.shape)
# sys.exit()

print('===start recognition===')
idx = 0
cur_correct = 0
total_correct = 0
batch = 0
while idx < len(test_code_gallery):
    tmp_code = test_code_gallery[idx].reshape(1,-1)
    # print(tmp_code.shape)
    # print(code_gallery.shape)
    cos_similarity = cosine_similarity(code_gallery, tmp_code)
    # print(cos_similarity.shape)
    best_match = np.argmax(cos_similarity)
    # print(best_match)
    # print('%d =? %d'%(palmlabel[best_match],test_labels[idx]))
    if palmlabel[best_match] == test_labels[idx]:
        cur_correct += 1
        total_correct += 1
    if (idx + 1) % 100 == 0:
        print('batch %d: correct rate = %.3f' % (batch, cur_correct / 100))
        cur_correct = 0
        batch += 1
    idx += 1
print('TOTAL CORRECT RATE: %.3f' % (total_correct / len(test_code_gallery)))
