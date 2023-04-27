import sys

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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


preprocessing = None
testprocessing = None


def prepare_transform_for_image():
    global preprocessing
    global testprocessing
    rotation = transforms.RandomRotation(5)
    resized_cropping = transforms.Resize((32, 32))
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
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]
    )
    testprocessing = transforms.Compose(
        [resized_cropping,
         transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])


# 读入图片
dataset = 'CASIA'
img_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session2/'
label_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session2_label.txt'
gallery_label_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session1_label.txt'
save_pca_PATH = '/home/ubuntu/graduation_model/palmmatrix_test_session1.joblib'
save_numpy_PATH = '/home/ubuntu/graduation_model/palmmatrix_test_session1_numpy.npy'
save_svm_PATH = '/home/ubuntu/graduation_model/merge/'+dataset+'/svm.joblib'
save_kpca_PATH = '/home/ubuntu/graduation_model/kpca_palmmatrix_test_session1.joblib'
save_lda_PATH = '/home/ubuntu/graduation_model/lda_palmmatrix_test_session1.joblib'
prepare_transform_for_image()
# palms = []
palmmatrix = []
testmatrix = []
testlabel = []

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
    cur = testprocessing(Image.open(tmp_image_path).convert('L'))
    # print(cur.size())
    cur = cur.permute(1, 2, 0).detach().numpy()
    # palms.append(cur)
    testmatrix.append(cur.flatten())
    testlabel.append(img_label)
testmatrix = np.array(testmatrix)
print(testmatrix.shape)
# print(testlabel)
# print(palmmatrix.shape)
print('===%d images Done! %d classes in total!===' % (len(content), class_size - min_size + 1))
# 读入原始标签
label_file = open(gallery_label_PATH, 'r')
content = label_file.readlines()
palmlabel = []
for line in content:
    img_label = int(line.split(' ')[1])
    palmlabel.append(img_label)
# print(palmlabel)
# 导入pca
print('===start load pca!===')
palmmatrix = np.load(save_numpy_PATH)
pca = load(save_pca_PATH)
kpca = load(save_kpca_PATH)
lda = load(save_lda_PATH)
# 生成特征脸
n_components = 80
lda = pca
# eigenpalms = pca.components_[:n_components]

# 权重
# weights = eigenpalms @ (palmmatrix - pca.mean_).T
# weights = weights.T
# print(weights.shape)
# kpca_weights = kpca.eigenvectors_[:,:80]
# print(kpca_weights.shape)
# merge_sub_gallery = np.append(weights,kpca_weights,axis=1)
weights = lda.transform(palmmatrix)
# print(weights.shape)
# weights = pca.transform(palmmatrix)

# 训练svm
# svm = SVC(kernel='rbf')
# svm.fit(weights,palmlabel)
# dump(svm,save_svm_PATH)

# 测试
print('===start test!===')

idx = 0
cur_correct = 0
total_correct = 0
batch = 0
# query_kpca = kpca.transform(testmatrix)
# query = pca.transform(testmatrix)
# test_sub = np.append(query,query_kpca,axis=1)
print(testmatrix.shape)
plt.scatter(testmatrix[:, 0], testmatrix[:, 1],marker='o',c=testlabel)
plt.xlabel("xxxx")
plt.show()
query = lda.transform(testmatrix)
print(query.shape)
plt.scatter(query[:, 0], query[:, 1],marker='o',c=testlabel)
plt.show()
# print(query.shape)
# print(query_kpca.shape)
sys.exit()
while idx < len(query):
    # query = testmatrix[idx].reshape(1, -1)
    # print(query.shape)
    # query_weight = eigenpalms @ (query - pca.mean_).T

    # print(query_weight.shape)
    # print(query_weight.shape)
    # print(weights.shape)
    # cos_similarity = calculate_cos_similar(weights,query_weight)
    # best_match = svm.predict(query[idx].reshape(1,-1))
    # if best_match[0] == testlabel[idx]:
    cos_similarity = cosine_similarity(weights,query[idx].reshape(1,-1))
    best_match = np.argmax(cos_similarity)
    if palmlabel[best_match] == testlabel[idx]:
        cur_correct+=1
        total_correct+=1
    if (idx+1)%100 == 0:
        print('batch %d: correct rate = %.3f'%(batch,cur_correct/100))
        cur_correct=0
        batch+=1
    idx += 1
    break

print('TOTAL CORRECT RATE: %.3f'%(total_correct/len(testmatrix)))
