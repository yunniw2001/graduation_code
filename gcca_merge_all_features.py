import sys
import time

import numpy as np
from skimage.filters._gabor import gabor_kernel
# from sklearn.cross_decomposition import CCA
import torch
from sklearn.svm import SVC
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import ResNet
from ArcFace import ArcFace
from my_transformer import MeanFiltersTransform, MedianFiltersTransform, GaussFiltersTransform, \
    GaussianFiltersTransformUnsharpMask, MedianFiltersTransformUnsharpMask, MeanFiltersTransformUnsharpMask, MyDataset
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy import ndimage as ndi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mcca.gcca import G_CCA as CCA
import logging
import numpy as np

logging.root.setLevel(level=logging.INFO)
# random apply preprocessing
preprocessing = []
testprocessing = []
my_gabor_filter = None
to_greyscale = None

def feature_read(root_path):
    res_feature = np.load(root_path + 'dl_feature.npy')
    pca_feature = np.load(root_path + 'weights_pca.npy')
    lda_feature = np.load(root_path+'weights_lda.npy')
    compcode_feature = np.load(root_path + 'code_feature.npy')
    label_matrix = np.load(root_path + 'gallery_label.npy')
    return res_feature,pca_feature,lda_feature,compcode_feature,label_matrix





def calculate_accuracy(logits, label):
    _, pred = torch.max(logits.data, 1)
    i = 0
    while i < len(pred):
        pred[i] = gallery_label[pred[i]]
        i += 1
    return (label.data == pred).float().mean()


def balance_dimension(matrix_a,matrix_b):
    mixed = np.append(matrix_a, matrix_b, axis=1)
    return np.split(mixed, 2, axis=1)

def cca_merge_feature(tmp_cca,feature1,feature2,mode = 'test'):
    if not mode == 'test':
        tmp_cca.fit(feature1,feature2)
    tmp_cca.transform(feature1,feature2)
    return np.append(feature1,feature2,axis=1)

def normalization(feature_set):
    for i in range(len(feature_set)):  # 归一化每个特征
        feature_set[i] = feature_set[i]/ np.linalg.norm(feature_set[i])
    return feature_set

def feature_standard(X):
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return X

def feature_MinMax(X):
    _range = np.max(X,axis=0) - np.min(X,axis=0)
    return (X - np.min(X,axis=0)) / _range

batch_size = 40
num_class = 480
feature_size = 128
lr = 0.001
epochs = 1000

dataset = 'IITD'
root_path = '/home/ubuntu/dataset/'+dataset+'/session/'
if_need_norm = 'unitlength'
vote_state = 'majority'
n_max = 5

# print('===current dataset is: '+dataset+' and current mode is '+if_need_norm+'===')
write_file_root = '/home/ubuntu/graduation_project/output/'
write_name = 'gcca_rplc_svm_'+if_need_norm+'_'+dataset+'.txt'
print(write_file_root+write_name)
file = open(write_file_root+write_name,'w')
file.write('a')
file.close()
print('===current dataset is: '+dataset+' and current mode is '+if_need_norm+'===')
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



# 加载参数
if_need_balance=False
sv_path = '/home/ubuntu/graduation_model/features/'+dataset
print('===read session 1===')
print('read session1!')
feature_gallery,pca_weights,weights,code_gallery,gallery_label = feature_read(sv_path+'/session1/')
print('read session2!')
test_dl_feature,pca_query, query, test_code_feature, testlabel = feature_read(sv_path+'/session2/')
print("===completed!===")

print('===start merge features!===')


# print(feature_gallery.shape)
# print(palmmatrix.shape)
# classic_cca = CCA(n_components=120)
# dl_cca = CCA(n_components=60)
if if_need_norm == 'unitlength':
    feature_gallery =normalization(feature_gallery)
    pca_weights = normalization(pca_weights)
    code_gallery = normalization(code_gallery)
    weights = normalization(weights)
    test_dl_feature = normalization(test_dl_feature)
    test_code_feature = normalization(test_code_feature)
    pca_query = normalization(pca_query)
    query = normalization(query)
elif if_need_norm == 'minmax':
    feature_gallery = feature_MinMax(feature_gallery)
    pca_weights = feature_MinMax(pca_weights)
    code_gallery = feature_MinMax(code_gallery)
    weights = feature_MinMax(weights)
    test_dl_feature = feature_MinMax(test_dl_feature)
    test_code_feature = feature_MinMax(test_code_feature)
    pca_query = feature_MinMax(pca_query)
    query = feature_MinMax(query)

elif if_need_norm == 'meanstd':
    feature_gallery = feature_standard(feature_gallery)
    pca_weights = feature_standard(pca_weights)
    code_gallery = feature_standard(code_gallery)
    weights = feature_standard(weights)
    test_dl_feature = feature_standard(test_dl_feature)
    test_code_feature = feature_standard(test_code_feature)
    pca_query = feature_standard(pca_query)
    query = feature_standard(query)


cca = CCA(n_components=80)
cca.fit(feature_gallery,pca_weights,weights,code_gallery)
cca.transform(feature_gallery,pca_weights,weights,code_gallery)
mergeallfeature_gallery = np.concatenate((np.array(feature_gallery),np.array(pca_weights),np.array(weights),np.array(code_gallery)),axis=1)
print('===start test!===')
cca.transform(test_dl_feature,pca_query,query,test_code_feature)
mergeallfeature_test = np.concatenate((np.array(test_dl_feature),np.array(pca_query),np.array(query),np.array(test_code_feature)),axis=1)
svm = SVC(kernel='sigmoid')
svm.fit(mergeallfeature_gallery,gallery_label)
idx = 0
total_correct = 0
cur_correct = 0
batch = 0
start_time = time.perf_counter()
while idx < len(mergeallfeature_test):
    # print(merge_gallery.shape)
    # print(test_merge[idx].shape)
    # break
    # cos_similarity = cosine_similarity(mergeallfeature_gallery,mergeallfeature_test[idx].reshape(1,-1))
    # best_match = np.argmax(cos_similarity)
    best_match = svm.predict(mergeallfeature_test[idx].reshape(1,-1))
    if best_match[0] == testlabel[idx]:
    # print(best_match)
    # print('%d =? %d'%(palmlabel[best_match],test_labels[idx]))

    # if gallery_label[best_match] == testlabel[idx]:
        cur_correct += 1
        total_correct += 1
    if (idx + 1) % 100 == 0:
        print('batch %d: correct rate = %.3f' % (batch, cur_correct / 100))
        cur_correct = 0
        batch += 1
    idx += 1
end_time = time.perf_counter()
run_time = end_time-start_time
print('===total time: %f***average time: %f==='%(run_time,run_time/len(testlabel)))
print('TOTAL CORRECT RATE: %.3f' % (total_correct / len(mergeallfeature_test)))







