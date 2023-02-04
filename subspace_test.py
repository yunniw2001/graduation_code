import cv2
import numpy as np
import torchvision
import torchvision.transforms as transforms
from my_transformer import MeanFiltersTransform, MedianFiltersTransform, GaussFiltersTransform, \
    GaussianFiltersTransformUnsharpMask, MedianFiltersTransformUnsharpMask, MeanFiltersTransformUnsharpMask
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from joblib import dump, load

preprocessing = None
testprocessing = None


def prepare_transform_for_image():
    global preprocessing
    global testprocessing
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
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]
    )
    testprocessing = transforms.Compose(
        [resized_cropping,
         transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])

def calculate_cos_similar(matrix,vector):
    num = np.dot(vector.T,matrix)  # 向量点乘
    denom = np.linalg.norm(matrix) * np.linalg.norm(vector)  # 求模长的乘积
    print(num.shape)
    res = num / denom
    # print(res.shape)
    # res[np.isneginf(res)] = 0
    res = 0.5+0.5*res
    return res[0]


# 读入图片
img_PATH = '/home/ubuntu/dataset/CASIA/test_session/session2/'
label_PATH = '/home/ubuntu/dataset/CASIA/test_session/session2_label.txt'
save_pca_PATH = '/home/ubuntu/graduation_model/palmmatrix_test_session1.joblib'
save_numpy_PATH = '/home/ubuntu/graduation_model/palmmatrix_test_session1_numpy.npy'
prepare_transform_for_image()
palms = []
palmmatrix = []
palmlabel = []

label_file = open(label_PATH, 'r')
content = label_file.readlines()
class_size = 0
min_size = 1000000

print('===start load test image!===')
for line in content:
    img_name, img_label = line.split(' ')[0], int(line.split(' ')[1])
    tmp_image_path = img_PATH + img_name
    if img_label > class_size:
        class_size = img_label
    if img_label < min_size:
        min_size = img_label
    cur = preprocessing(Image.open(tmp_image_path).convert('L'))
    # print(cur.size())
    cur = cur.permute(1, 2, 0).detach().numpy()
    palms.append(cur)
    palmmatrix.append(cur.flatten())
    palmlabel.append(img_label)
palmmatrix = np.array(palmmatrix)
print('===%d images Done! %d classes in total!===' % (len(content), class_size - min_size + 1))
# 导入pca
print('===start load pca!===')
palmmatrix = np.load(save_numpy_PATH)
pca = load(save_pca_PATH)
# 生成特征脸
n_components = 50
eigenpalms = pca.components_[:n_components]

# 权重
weights = eigenpalms @ (palmmatrix - pca.mean_).T

# 测试
print('===start test!===')

idx = 0
cur_correct = 0
total_correct = 0
batch = 0
while idx < len(palmmatrix):
    query = palmmatrix[idx].reshape(1, -1)
    query_weight = eigenpalms @ (query - pca.mean_).T
    # cos_similarity = calculate_cos_similar(weights,query_weight)
    cos_similarity = np.linalg.norm(weights - query_weight, axis=0)
    best_match = np.argmin(cos_similarity)
    # print(palmlabel[best_match])
    # print(len(palmlabel))
    # print(palmlabel[idx])
    if palmlabel[best_match] == palmlabel[idx]:
        cur_correct+=1
        total_correct+=1
    if (idx+1)%100 == 0:
        print('batch %d: correct rate = %.2f'%(batch,cur_correct/100))
        cur_correct=0
        batch+=1
    idx += 1
    # break

print('TOTAL CORRECT RATE: %.2f'%(total_correct/len(palmmatrix)))
