import numpy as np
from sklearn.metrics import roc_curve
import numpy as np
from skimage.filters._gabor import gabor_kernel
# from sklearn.cross_decomposition import CCA
from mcca.cca import CCA as CCA
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve, auc
# from sklearn.cross_decomposition import CCA
import torch
from sklearn.preprocessing import normalize
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
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 10,
    "mathtext.fontset":'stix',
}
rcParams.update(config)

def calculate_cosine_similarity(features):
    num_faces = len(features)
    similarities = np.zeros((num_faces, num_faces))

    for i in range(num_faces):
        for j in range(i+1, num_faces):
            # 计算余弦相似度
            similarity = np.dot(features[i], features[j]) / (np.linalg.norm(features[i]) * np.linalg.norm(features[j]))
            similarities[i][j] = similarities[j][i] = similarity

    return similarities


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
    # accuracy_mean = np.mean(accuracy)
    # sigma = np.sqrt(np.sum(np.power(accuracy-accuracy_mean,2)))
    # miu = np.fabs(1-2.5*sigma)
    # beta = np.exp(-np.power(x-miu,2)/(2*(sigma**2)))/(sigma*np.sqrt(2*np.pi))
    # print(beta)
    return x[0],x[1],x[2]

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

def display_one_dim(arr):
    for i in arr:
        print(i,end=' ')
    print('')

def compu_roc(class_in,class_each):
    FRR = []
    FAR = []
    thresld = np.arange(0, 1, 0.001)  # 生成模型阈值的等差列表
    eer = 1
    for i in range(len(thresld)):
        frr = np.sum(class_in < thresld[i]) / len(class_in)
        FRR.append(frr)

        far = np.sum(class_each > thresld[i]) / len(class_each)
        FAR.append(far)

        if (abs(frr - far) < 0.02):  # frr和far值相差很小时认为相等
            eer = abs(frr + far) / 2
    return eer
def compute_eer(fpr,tpr,threshold):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    print(eer_threshold)
    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    # eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(eer)
    # print(thresh)
    # eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    return eer

def get_score_matrix(gallery,test_batch):
    cosine_matrix = cosine_similarity(gallery,test_batch)
    tmp = np.argmax(cosine_matrix,axis=0)
    tmp = np.expand_dims(tmp,axis=0)
    out = np.zeros_like(cosine_matrix)
    np.put_along_axis(out,tmp,1,axis=0)
    return out

def get_positive_label(match_result,gallery_label,testlabel):
    if_postive = np.zeros_like(match_result)
    col, row = if_postive.shape
    for line in range(col):
        for j in range(row):
            if gallery_label[line] == testlabel[j]:
                if_postive[line][j] = 1
    if_postive = if_postive.flatten()
    score = match_result.flatten()
    print(max(score))
    return if_postive,score

batch_size = 40
num_class = 480
feature_size = 128
lr = 0.001
epochs = 1000

def ccosine_similarity(feature_gallery,test_feature):
    cosine_matrix = cosine_similarity(feature_gallery,test_feature)
    tmp = np.argmax(cosine_matrix, axis=0)
    tmp = np.expand_dims(tmp, axis=0)
    out = np.zeros_like(cosine_matrix)
    np.put_along_axis(out, tmp, 1, axis=0)
    return out

dataset = 'tongji'
print('===current dataset is:'+dataset+'!===')
model_folder = '/home/ubuntu/graduation_model/merge/'+dataset+'/'
already_prepared = True
test_mode = True
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
if_need_norm = 'unitlength'

# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
net = ResNet.resnet34()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

write_file_root = '/home/ubuntu/graduation_project/output/'
write_name = 'score_fast_'+if_need_norm+'_'+dataset+'.txt'
# print(write_file_root+write_name)
# file = open(write_file_root+write_name,'w')
# file.write('a')
# file.close()
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

# merge_weight, lda_weight, compcode_weight = calculate_weight([0.9,0.8, 0.8])
# svm_merge = SVC(kernel='sigmoid')
merge_weight, lda_weight, compcode_weight = [0.9,0.8, 0.8]
# svm_merge = LinearSVC()
# svm_merge.fit(merge_gallery,gallery_label)
# print(dl_weight)


idx = 0
total_correct = 0
cur_correct = 0
batch = 0
start_time = time.perf_counter()
# 确定batch
batch_size = 100

resnet_match_result = cosine_similarity(feature_gallery,test_dl_feature)
cca_match_result = ccosine_similarity(merge_gallery,test_merge)
lda_match_result = ccosine_similarity(weights,query)
code_match_result = ccosine_similarity(code_gallery,test_code_feature)
pca_match_result = cosine_similarity(pca_weights,pca_query)

final_match_result = merge_weight*cca_match_result+lda_weight*lda_match_result+compcode_weight*code_match_result
final_match_result = feature_norm(final_match_result)
res_match_result = feature_norm(resnet_match_result)
cca_match_result = feature_norm(cca_match_result)
lda_match_result = feature_norm(lda_match_result)
code_match_result = feature_norm(code_match_result)
pca_match_result = feature_norm(pca_match_result)

print(final_match_result.shape)
pred_label = gallery_label[np.argmax(final_match_result,axis=0)]

our_positive,our_score = get_positive_label(final_match_result,gallery_label,testlabel)
class_in = []
class_each = []
thre = 0.635545550667493
ans =0
# for i in range(len(our_positive)):
#     if our_positive[i] == 1:
#         label1 = i//1200
#         label2 = i%1200
#         if gallery_label[label1] == testlabel[label2] and our_score[i] <thre:
#             ans+=1
# print(ans)


# display_one_dim(final_match_result[:, 0])

total_correct = np.sum(pred_label == testlabel)
test_score = np.sum(our_score > thre)
print('a:%d'%test_score)
# print(testlabel)
print(total_correct)
print(len(our_positive))
our_fpr, our_tpr, our_thresholds = roc_curve(our_positive, our_score, pos_label=1)
print(max(our_thresholds))

# fn_num = 0
# where_0001 = np.argmin(abs(0.01-1+our_fpr))
# print(our_tpr[where_0001])


# display_one_dim(our_fpr)
# res_positive,res_score = get_positive_label(resnet_match_result,gallery_label,testlabel)
# res_fpr, res_tpr, res_thresholds = roc_curve(res_positive, res_score, pos_label=1)
# pca_positive,pca_score = get_positive_label(pca_match_result,gallery_label,testlabel)
# pca_fpr, pca_tpr, pca_thresholds = roc_curve(pca_positive, pca_score, pos_label=1)
# lda_positive,lda_score = get_positive_label(lda_match_result,gallery_label,testlabel)
# lda_fpr, lda_tpr, lda_thresholds = roc_curve(lda_positive, lda_score, pos_label=1)
# code_positive,code_score = get_positive_label(code_match_result,gallery_label,testlabel)
# code_fpr, code_tpr, code_thresholds = roc_curve(code_positive, code_score, pos_label=1)
# cca_positive,cca_score = get_positive_label(cca_match_result,gallery_label,testlabel)
# cca_fpr, cca_tpr, cca_thresholds = roc_curve(cca_positive, cca_score, pos_label=1)
# print(fpr)
# res_fpr
our_auc =auc(our_fpr,our_tpr)
# print(our_auc)


# plt.figure()
l1, = plt.plot(our_fpr, our_tpr)
# l2, = plt.plot(our_thresholds, our_fpr)
# plt.ylim((0,0.1))
# lengend = plt.legend(handles=[l1,l2],labels=['fnr','far'],loc='best')
plt.show()
# l2, = plt.plot(res_fpr, res_tpr)
# l3, = plt.plot(lda_fpr, lda_tpr)
# l4, = plt.plot(pca_fpr, pca_tpr)
# l5, = plt.plot(code_fpr, code_tpr)
# l6, = plt.plot(cca_fpr, cca_tpr)
# plt.xlabel('False Accept Rate')
# plt.ylabel('Genuine Accept Rate')
# plt.xlim((0,0.04))
# plt.ylim((0.96,1))
# lengend = plt.legend(handles=[l1,l2,l3,l4,l5,l6],labels=['ours','ResNet-34','LDA','PCA','CompCode','cca fused feature'],loc='best')
# # plt.title('ROC')
# # plt.plot(fpr, tpr)
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.savefig('/home/ubuntu/output_image/'+dataset+'_roc_curve.pdf')
# plt.show()
#
#
our_eer = compute_eer(our_fpr,our_tpr,our_thresholds)
# res_eer = compute_eer(res_fpr,res_tpr,res_thresholds)
# pca_eer = compute_eer(pca_fpr,pca_tpr,pca_thresholds)
# lda_eer = compute_eer(lda_fpr,lda_tpr,lda_thresholds)
# code_eer = compute_eer(code_fpr,code_tpr,code_thresholds)
# cca_eer = compute_eer(cca_fpr,cca_tpr,cca_thresholds)
print("OUR_EER:", our_eer)
# print("Res_EER:", res_eer)
# print("Code_EER:", code_eer)
# print("LDA_EER:", lda_eer)
# print("PCA_EER:", pca_eer)
# print("CCA_EER:", cca_eer)
# plt.figure()
# plt.plot(1 - tpr, thresholds,label = 'far')
# plt.plot(fpr, thresholds,label = 'fpr')
# plt.legend()
# plt.xlabel('thresh')
# plt.ylabel('far/fpr')
# plt.title(' eer')
# plt.show()

