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
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

preprocessing = None
testprocessing = None
my_gabor_filter = None
already_processed = True

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


def read_image_and_label(labelpath, imgpath, state='train'):
    label_file = open(labelpath, 'r')
    content = label_file.readlines()
    code_g = []
    labels = []
    testmatrix = []
    num = 0

    # print('===start read images and labels, generate uniform LBP gallery!===')
    for line in content:
        img_name, img_label = line.split(' ')[0], int(line.split(' ')[1])
        tmp_image_path = imgpath + img_name
        # print(tmp_image_path)
        cur = testprocessing(Image.open(tmp_image_path).convert('L'))
        # print(cur.size())
        tmp = cur.numpy()[0]
        # print(tmp.shape)
        cur_code = my_gabor_filter.extract_CompCode(tmp)

        cur = cur.permute(1, 2, 0).detach().numpy()
        testmatrix.append(cur.flatten())

        code_g.append(cur_code)
        labels.append(img_label)
        num+=1
        if num%100 == 0:
            print('%d images Done!'%num)
    return code_g,np.array(testmatrix), labels

def read_original_label(filepath):
    label_file = open(filepath, 'r')
    content = label_file.readlines()
    palmlabel = []
    for line in content:
        img_label = int(line.split(' ')[1])
        palmlabel.append(img_label)
    return palmlabel


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

    def extract_CompCode(self, image,show = False):
        gabor_responses = []
        for k,kernel in enumerate(self.filters):
            filtered = ndi.convolve(image,kernel,mode='wrap')
            gabor_responses.append(filtered)
        gabor_responses = np.array(gabor_responses)

        winner = np.argmin(gabor_responses,axis=0)
        # 标准化
        winner = (winner - np.max(np.max(winner))) * -1
        output = (winner / np.max(np.max(winner))) * 255
        return output.reshape(1, -1)[0]

    def power(self,image):
        res = []
        for kernel in self.filters:
            res.append(np.sqrt(ndi.convolve(image,np.real(kernel),mode='wrap')**2+ndi.convolve(image,np.imag(kernel),mode='wrap')**2))

dataset = 'tongji'
img_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session2/'
label_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session2_label.txt'
gallery_label_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session1_label.txt'
save_pca_PATH = '/home/ubuntu/graduation_model/palmmatrix_test_session1.joblib'
save_numpy_PATH = '/home/ubuntu/graduation_model/palmmatrix_test_session1_numpy.npy'
save_gallery = '/home/ubuntu/graduation_model/gallery_texture.npy'
save_label = '/home/ubuntu/graduation_model/label_texture.npy'

prepare_transform_for_image()
my_gabor_filter = Gabor_filters()
my_gabor_filter.build_filters()

palmmatrix = []
testmatrix = []
testlabel = []

# 读入原始标签
palmlabel = read_original_label(gallery_label_PATH)

print('===start load test image!===')
test_code_gallery, testmatrix, test_labels = read_image_and_label(label_PATH, img_PATH,'test')

# 读入gallery
code_gallery = np.load(save_gallery)
palmlabel = np.load(save_label)
# 生成特征脸
print('===start load pca!===')
palmmatrix = np.load(save_numpy_PATH)
pca = load(save_pca_PATH)
n_components = 50
eigenpalms = pca.components_[:n_components]
weights = eigenpalms @ (palmmatrix - pca.mean_).T
weights = weights.T

print(weights.shape)
print(code_gallery.shape)
merge_gallery = np.append(weights,code_gallery,axis=1)

# test
idx = 0
cur_correct = 0
total_correct = 0
batch = 0
print(testmatrix.shape)
while idx < len(testmatrix):
    tmp_compcode = test_code_gallery[idx].reshape(1, -1)
    query = testmatrix[idx].reshape(1, -1)
    # print(query.shape)
    # print(pca.mean_.shape)
    query_weight = eigenpalms @ (query - pca.mean_).T
    query_weight = query_weight.T
    # print(tmp_hist.shape)
    # print(query_weight.shape)
    merged_feature = np.append(query_weight,tmp_compcode,axis=1)
    # print(merged_feature.shape)
    # break
    cos_similarity = cosine_similarity(merge_gallery, merged_feature)
    best_match = np.argmax(cos_similarity)
    # print(best_match)
    # print('%d =? %d'%(palmlabel[best_match],test_labels[idx]))
    if palmlabel[best_match] == test_labels[idx]:
        cur_correct += 1
        total_correct += 1
    if (idx + 1) % 100 == 0:
        print('batch %d: correct rate = %.2f' % (batch, cur_correct / 100))
        cur_correct = 0
        batch += 1
    idx += 1
print('TOTAL CORRECT RATE: %.2f' % (total_correct / len(testmatrix)))
