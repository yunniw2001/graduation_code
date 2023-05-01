import sys

import numpy as np
from skimage.filters._gabor import gabor_kernel
from sklearn.cross_decomposition import CCA
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import ResNet
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from ArcFace import ArcFace
from my_transformer import MeanFiltersTransform, MedianFiltersTransform, GaussFiltersTransform, \
    GaussianFiltersTransformUnsharpMask, MedianFiltersTransformUnsharpMask, MeanFiltersTransformUnsharpMask, MyDataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity
from scipy import ndimage as ndi
from joblib import dump, load
import time

# random apply preprocessing
preprocessing = []
testprocessing = []
my_gabor_filter = None
to_greyscale = None
gallery_label = None


def normalization(feature_set):
    for i in range(len(feature_set)):  # 归一化每个特征
        feature_set[i] = feature_set[i]/ np.linalg.norm(feature_set[i])
    return feature_set




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

def vote_score(vote_box:dict, matrix, vector, weight):
    cos_similarity = cosine_similarity(matrix, vector)
    best_match = np.argmax(cos_similarity)
    best_match = gallery_label[best_match]
    if vote_box.get(best_match):
        vote_box[best_match] += weight
    else:
        vote_box[best_match] = weight
    return vote_box

def vote_n_max_decision(vote_box:dict,matrix,vector,n):
    cos_similarity = cosine_similarity(matrix, vector).ravel()
    # print(cos_similarity)
    best_match = np.argsort(cos_similarity)[::-1]
    # print(np.argmax(cos_similarity))
    # print(best_match)
    best_match = best_match[0:n]
    # print(best_match)
    best_match = gallery_label[best_match]
    # print(best_match)
    for possible_class in best_match:
        # print(possible_class)
        if vote_box.get(possible_class):
            vote_box[possible_class] += 1
        else:
            vote_box[possible_class] = 1
    return vote_box
def vote_svm(vote_box:dict,vector,weight,svm_object):
    # cos_similarity = cosine_similarity(matrix, vector)
    best_match = svm_object.predict(vector)[0]
    if vote_box.get(best_match):
        vote_box[best_match] += weight
    else:
        vote_box[best_match] = weight
    return vote_box

def calculate_weight(accuracy):
    accuracy = np.array(accuracy)
    # normalization
    tot = sum(accuracy)
    x = accuracy/tot
    # beta_k
    accuracy_mean = np.mean(accuracy)
    sigma = np.sqrt(np.sum(np.power(accuracy-accuracy_mean,2)))
    miu = np.fabs(1-2.5*sigma)
    beta = np.exp(-np.power(x-miu,2)/(2*(sigma**2)))/(sigma*np.sqrt(2*np.pi))
    print(beta)
    return beta[0],beta[1],beta[2]

def feature_standard(X):
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return X

def feature_MinMax(X):
    _range = np.max(X,axis=0) - np.min(X,axis=0)
    return (X - np.min(X,axis=0)) / _range

def feature_read(root_path):
    res_feature = np.load(root_path + 'dl_feature.npy')
    pca_feature = np.load(root_path + 'weights_pca.npy')
    lda_feature = np.load(root_path+'weights_lda.npy')
    compcode_feature = np.load(root_path + 'code_feature.npy')
    label_matrix = np.load(root_path + 'gallery_label.npy')
    return res_feature,pca_feature,lda_feature,compcode_feature,label_matrix

def cca_merge(tmp_cca,matrix1,matrix2,mode='test'):
    if not mode == 'test':
        tmp_cca.fit(matrix1,matrix2)
    cca1, cca2 = tmp_cca.transform(matrix1, matrix2)
    merge_gallery = np.append(cca1, cca2, axis=1)
    return tmp_cca,merge_gallery

batch_size = 40
num_class = 480
feature_size = 128
lr = 0.001
epochs = 1000


dataset = 'tongji'
if_need_balance=False
if_need_norm = 'unitlength'
vote_state = 'majority'
n_max = 5
model_folder = '/home/ubuntu/graduation_model/merge/'+dataset+'/'
already_prepared = True
test_mode = False
root_path = '/home/ubuntu/dataset/'+dataset+'/session/'
if test_mode:
    dataset = 'IITD'
    root_path = root_path = '/home/ubuntu/dataset/'+dataset+'/test_session/'
session1_dataset = MyDataset(root_path+'session1/',
                             root_path+'session1_label.txt', testprocessing)
session2_dataset = MyDataset(root_path+'session2/',
                             root_path+'session2_label.txt', testprocessing)
session1_dataloader = DataLoader(dataset=session1_dataset, batch_size=batch_size, shuffle=False)
session2_dataloader = DataLoader(dataset=session2_dataset, batch_size=batch_size, shuffle=True)

print('===current dataset is: '+dataset+' and current mode is '+if_need_norm+'===')
write_file_root = '/home/ubuntu/graduation_project/output/'
write_name = 'majority_'+if_need_norm+'_'+dataset+'.txt'
print(write_file_root+write_name)
log_file = open(write_file_root+write_name,'w')
log_file.write('a')
log_file.close()

# oldstdout = sys.stdout
# sys.stdout = file

print('===current dataset is:'+dataset+' and current normalization mode is '+if_need_norm+'===')
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# 加载参数
sv_path = '/home/ubuntu/graduation_model/features/'+dataset
if test_mode:
    sv_path = sv_path+'/small'
# print("===start generating gallery-dl-feature===")
if already_prepared:
    print('read session1!')
    feature_gallery,pca_weights,weights,code_gallery,gallery_label = feature_read(sv_path+'/session1/')
    print('read session2!')
    test_dl_feature,pca_query, query, test_code_feature, testlabel = feature_read(sv_path+'/session2/')

    if if_need_norm == 'unitlength':
        feature_gallery = normalization(feature_gallery)
        pca_weights = normalization(pca_weights)
        test_dl_feature = normalization(test_dl_feature)
        pca_query = normalization(pca_query)
    elif if_need_norm == 'minmax':
        feature_gallery = feature_MinMax(feature_gallery)
        pca_weights = feature_MinMax(pca_weights)
        test_dl_feature = feature_MinMax(test_dl_feature)
        pca_query = feature_MinMax(pca_query)
    elif if_need_norm == 'meanstd':
        feature_gallery = feature_standard(feature_gallery)
        pca_weights = feature_standard(pca_weights)
        test_dl_feature = feature_standard(test_dl_feature)
        pca_query = feature_standard(pca_query)

else:
    print('haven\'t prepared! check it!')
# weights = feature_norm(weights)
# pca_weights = feature_norm(pca_weights)
# 标准化

# svm_merge = SVC(kernel='sigmoid')
# svm_merge = LinearSVC()
# svm_merge.fit(weights,gallery_label)



# test_dl_feature = test_dl_feature.cpu().numpy()
# query = feature_norm(query)
# pca_query = feature_norm(pca_query)
# test_dl_feature = feature_norm(test_dl_feature)
if if_need_norm:
    test_dl_feature = normalization(test_dl_feature)
    pca_query = normalization(pca_query)
print('===start merge session1 features!===')
start_time = time.perf_counter()
classic_cca = CCA(n_components=120)
classic_cca,merge_gallery = cca_merge(classic_cca,feature_gallery,pca_weights,'train')
end_time = time.perf_counter()
run_time = end_time-start_time
print('===total time: %f***average time: %f==='%(run_time,run_time/len(gallery_label)))
print('===start merge session2 features!===')
start_time = time.perf_counter()
_,test_merge = cca_merge(classic_cca,test_dl_feature,pca_query)
end_time = time.perf_counter()
run_time = end_time-start_time
print('===total time: %f***average time: %f==='%(run_time,run_time/len(testlabel)))

# calculate dynamic weight

dl_weight,merge_weight, lda_weight, compcode_weight = [0.90,0.901, 0.804, 0.8]

# print(dl_weight)


idx = 0
total_correct = 0
cur_correct = 0
batch = 0
start_time = time.perf_counter()

while idx < len(test_dl_feature):
    # print(merge_gallery.shape)
    # print(test_merge[idx].shape)
    # break
    vote_box = {}
    if vote_state == 'default':
        vote_box = vote_score(vote_box, merge_gallery, test_merge[idx].reshape(1, -1), merge_weight)
        vote_box = vote_score(vote_box, weights, query[idx].reshape(1, -1), lda_weight)
        vote_box = vote_score(vote_box, code_gallery, test_code_feature[idx].reshape(1, -1), compcode_weight)
    elif vote_state == 'majority':
        vote_box = vote_n_max_decision(vote_box, merge_gallery, test_merge[idx].reshape(1, -1),n_max)
        vote_box = vote_n_max_decision(vote_box, weights, query[idx].reshape(1, -1),n_max)
        vote_box = vote_n_max_decision(vote_box, code_gallery, test_code_feature[idx].reshape(1, -1),n_max)
    # dl-vote
    # vote_box = vote(vote_box,feature_gallery,test_dl_feature[idx].reshape(1,-1),dl_weight)
    # vote_box = vote_svm(vote_box, test_merge[idx].reshape(1, -1), merge_weight,svm_merge)

    # vote_box = vote_svm(vote_box, query[idx].reshape(1, -1), lda_weight,svm_merge)
    # print(vote_box)

    # vote_box = vote_svm(vote_box, test_code_feature[idx].reshape(1, -1), compcode_weight,svm_merge)
    # print(vote_box)
    # break
    best_match = max(vote_box,key=vote_box.get)
    # print(best_match)
    # print('%d =? %d'%(palmlabel[best_match],test_labels[idx]))
    if best_match == testlabel[idx]:
        cur_correct += 1
        total_correct += 1
    # else:
    #     print(vote_box,end='')
    #     print('     correct answer is :%d'%testlabel[idx])
    if (idx + 1) % 100 == 0:
        print('batch %d: correct rate = %.3f' % (batch, cur_correct / 100))
        cur_correct = 0
        batch += 1
    idx += 1
print('TOTAL CORRECT RATE: %.3f' % (total_correct / len(testlabel)))
end_time = time.perf_counter()
run_time = end_time-start_time
print('===total time: %f***average time: %f==='%(run_time,run_time/len(testlabel)))
# sys.stdout = oldstdout







