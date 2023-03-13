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
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
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


# 读入图像
dataset = 'IITD'
img_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session1/'
label_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session1_label.txt'
save_pca_PATH = '/home/ubuntu/graduation_model/palmmatrix_test_session1.joblib'
save_kpca_PATH = '/home/ubuntu/graduation_model/kpca_palmmatrix_test_session1.joblib'
save_lda_PATH = '/home/ubuntu/graduation_model/lda_palmmatrix_test_session1.joblib'
save_numpy_PATH = '/home/ubuntu/graduation_model/palmmatrix_test_session1_numpy.npy'
prepare_transform_for_image()
palms = []
palmmatrix = []
palmlabel = []
already_processed = False

label_file = open(label_PATH, 'r')
content = label_file.readlines()
class_size = 0
min_size = 1000000
if not already_processed:
    print('===start read train dataset!===')
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
        palms.append(cur)
        palmmatrix.append(cur.flatten())
        palmlabel.append(img_label)
        # if i >= 20:
        #     break
        # break
    # sys.exit()
    palmmatrix = np.array(palmmatrix)
    print(palmmatrix.shape)
    # PCA
    print('===start PCA!===')
    n_components = 80
    pca = PCA(n_components=n_components).fit(palmmatrix)
    kpca = KernelPCA(n_components=n_components).fit(palmmatrix)
    lda = LDA(n_components=n_components).fit(palmmatrix,palmlabel)
    print('===PCA DONE!===')
    np.save(save_numpy_PATH, palmmatrix)
    dump(kpca,save_kpca_PATH)
    dump(pca, save_pca_PATH)
    dump(lda,save_lda_PATH)
else:
    print('===start load pca!===')
    palmmatrix = np.load(save_numpy_PATH)
    pca = load(save_pca_PATH)
    kpca = load(save_kpca_PATH)
    print('===start read label file!===')
    for line in content:
        img_name, img_label = line.split(' ')[0], int(line.split(' ')[1])
        if img_label > class_size:
            class_size = img_label
        if img_label < min_size:
            min_size = img_label
        palmlabel.append(img_label)

# 预览图片
# fig, axes = plt.subplots(4,5,sharex=True,sharey=True,figsize=(8,10))
# for i in range(20):
#     axes[i%4][i//4].imshow(palms[i], cmap="gray")
# plt.show()

print('===%d images Done! %d classes in total!===' % (len(content), class_size - min_size + 1))
# print(palmmatrix)
# print(palmmatrix.shape)

# 预览特征脸
n_components = 80
eigenpalms = pca.components_[:n_components]
# k_eigenpalms = kpca.components_[:n_components]

# fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
# for i in range(16):
#     axes[i % 4][i // 4].imshow(eigenpalms[i].reshape(224, 224), cmap="gray")
# plt.show()

# 权重
weights = eigenpalms @ (palmmatrix - pca.mean_).T
# 测试
print('===start test!===')
goal = 4
query = palmmatrix[goal].reshape(1, -1)
query_weight = eigenpalms @ (query - pca.mean_).T
euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
best_match = np.argmin(euclidean_distance)
print("Best match %s with Euclidean distance %f" % (palmlabel[best_match], euclidean_distance[best_match]))
print('the correct answer is %d' % palmlabel[goal])


# print(palmmatrix)
# idx = 0
# cur_correct = 0
# total_correct = 0
# batch = 0
# while idx < len(palmmatrix):
#     query = palmmatrix[idx].reshape(1, -1)
#     query_weight = eigenpalms @ (query - pca.mean_).T
#     # cos_similarity = calculate_cos_similar(weights,query_weight)
#     cos_similarity = np.linalg.norm(weights - query_weight, axis=0)
#     best_match = np.argmin(cos_similarity)
#     # print(palmlabel[best_match])
#     # print(len(palmlabel))
#     # print(palmlabel[idx])
#     # print('%d ?= %d'%(palmlabel[best_match],palmlabel[idx]))
#     if palmlabel[best_match] == palmlabel[idx]:
#         cur_correct+=1
#         total_correct+=1
#     if (idx+1)%100 == 0:
#         print('batch %d: correct rate = %.2f'%(batch,cur_correct/100))
#         cur_correct=0
#         batch+=1
#     idx += 1
#     # break
#
# print('TOTAL CORRECT RATE: %.2f'%(total_correct/len(palmmatrix)))
