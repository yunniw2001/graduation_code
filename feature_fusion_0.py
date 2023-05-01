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
        output = (winner / np.max(np.max(winner))) * 6
        # output = winner
        return output.reshape(1, -1)[0]

    def process_images(self,images):
        res = []
        for image in images:
            res.append(self.extract_CompCode(image[0]))
        return res


    def power(self,image):
        res = []
        for kernel in self.filters:
            res.append(np.sqrt(ndi.convolve(image,np.real(kernel),mode='wrap')**2+ndi.convolve(image,np.imag(kernel),mode='wrap')**2))



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

def vote(vote_box:dict,matrix,vector,weight):
    cos_similarity = cosine_similarity(matrix, vector)
    best_match = np.argmax(cos_similarity)
    best_match = gallery_label[best_match]
    if vote_box.get(best_match):
        vote_box[best_match] += weight
    else:
        vote_box[best_match] = weight
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

def feature_norm(X):
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


dataset = 'IITD'
print('===current dataset is:'+dataset+'!===')
model_folder = '/home/ubuntu/graduation_model/merge/'+dataset+'/'
already_prepared = True
test_mode = False
root_path = '/home/ubuntu/dataset/'+dataset+'/session/'
if test_mode:
    root_path = root_path = '/home/ubuntu/dataset/'+dataset+'/test_session/'
session1_dataset = MyDataset(root_path+'session1/',
                             root_path+'session1_label.txt', testprocessing)
session2_dataset = MyDataset(root_path+'session2/',
                             root_path+'session2_label.txt', testprocessing)
session1_dataloader = DataLoader(dataset=session1_dataset, batch_size=batch_size, shuffle=False)
session2_dataloader = DataLoader(dataset=session2_dataset, batch_size=batch_size, shuffle=True)
if_need_balance=False
if_need_norm = 'none'

# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
net = ResNet.resnet34()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

write_file_root = '/home/ubuntu/graduation_project/output/'
write_name = 'score_svmcode_'+if_need_norm+'_'+dataset+'.txt'
print(write_file_root+write_name)
file = open(write_file_root+write_name,'w')
file.write('a')
file.close()
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

else:
    print('haven\'t prepared! check it!')
# weights = feature_norm(weights)
# pca_weights = feature_norm(pca_weights)
# 标准化





# test_dl_feature = test_dl_feature.cpu().numpy()
# query = feature_norm(query)
# pca_query = feature_norm(pca_query)
# test_dl_feature = feature_norm(test_dl_feature)
if if_need_norm =='unitlength':
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
svm_merge = SVC(kernel='sigmoid')
# svm_merge = LinearSVC()
svm_merge.fit(merge_gallery,gallery_label)
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
    # dl-vote
    # vote_box = vote(vote_box,feature_gallery,test_dl_feature[idx].reshape(1,-1),dl_weight)
    vote_box = vote_svm(vote_box, test_merge[idx].reshape(1, -1), merge_weight,svm_merge)
    # vote_box = vote(vote_box,merge_gallery,test_merge[idx].reshape(1,-1),merge_weight)
    vote_box =vote(vote_box,weights,query[idx].reshape(1,-1),lda_weight)
    # vote_box = vote_svm(vote_box, query[idx].reshape(1, -1), lda_weight,svm_merge)
    # print(vote_box)
    vote_box = vote(vote_box,code_gallery,test_code_feature[idx].reshape(1,-1),compcode_weight)
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








