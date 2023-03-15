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
from ArcFace import ArcFace
from my_transformer import MeanFiltersTransform, MedianFiltersTransform, GaussFiltersTransform, \
    GaussianFiltersTransformUnsharpMask, MedianFiltersTransformUnsharpMask, MeanFiltersTransformUnsharpMask, MyDataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity
from scipy import ndimage as ndi
from joblib import dump, load

# random apply preprocessing
preprocessing = []
testprocessing = []
my_gabor_filter = None
to_greyscale = None

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


def calculate_accuracy(logits, label):
    _, pred = torch.max(logits.data, 1)
    i = 0
    while i < len(pred):
        pred[i] = gallery_label[pred[i]]
        i += 1
    return (label.data == pred).float().mean()

def read_image_and_label(dataloader):
    labels = []
    code_g = []
    dl_g = []
    testmatrix = []

    for i, data in enumerate(dataloader):
        images, label = data
        cur = to_greyscale(images)

        tmp = cur.numpy()
        # print(tmp.shape)
        cur = cur.detach().numpy()
        # print(cur.shape)

        cur_code = my_gabor_filter.process_images(tmp)
        cur = np.squeeze(cur)
        # print(cur.shape)

        images = images.to(device)
        label = label.to(device)
        labels.extend(label)
        feats, normalized_feature = net(images, None)
        if i == 0:
            testmatrix = cur
            code_g = cur_code
            dl_g = normalized_feature
            dl_g = torch.cat([normalized_feature, torch.flip(normalized_feature, [1])], 1)
        else:
            testmatrix = np.append(testmatrix, cur, axis=0)
            code_g = np.append(code_g,cur_code,axis=0)
            tmp = torch.cat([normalized_feature, torch.flip(normalized_feature, [1])], 1)
            dl_g = torch.cat([dl_g, tmp], 0)
        # code_g.append(cur_code)
        # print(testmatrix.shape)

        if (i + 1) % 10 == 0:
            print('[batch: %d DONE!]' % (i))
    testmatrix = testmatrix.reshape(*testmatrix.shape[:-2],-1)
    print("===completed!===")
    return dl_g,code_g, testmatrix, labels

def balance_dimension(matrix_a,matrix_b):
    mixed = np.append(matrix_a, matrix_b, axis=1)
    return np.split(mixed, 2, axis=1)

def vote(vote_box:dict,matrix,vector,weight):
    cos_similarity = cosine_similarity(matrix, vector)
    best_match = np.argmax(cos_similarity)
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



batch_size = 40
num_class = 480
feature_size = 128
lr = 0.001
epochs = 1000

prepare_transform_for_image()
my_gabor_filter = Gabor_filters()
my_gabor_filter.build_filters()

dataset = 'tongji'
model_folder = '/home/ubuntu/graduation_model/merge/'+dataset+'/'
already_prepared = False
root_path = '/home/ubuntu/dataset/'+dataset+'/test_session/'
session1_dataset = MyDataset(root_path+'session1/',
                             root_path+'session1_label.txt', testprocessing)
session2_dataset = MyDataset(root_path+'session2/',
                             root_path+'session2_label.txt', testprocessing)
session1_dataloader = DataLoader(dataset=session1_dataset, batch_size=batch_size, shuffle=False)
session2_dataloader = DataLoader(dataset=session2_dataset, batch_size=batch_size, shuffle=True)
if_need_balance=False
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
net = ResNet.resnet34()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# 加载参数
PATH_NET = '/home/ubuntu/graduation_model/deeplearning/model_net_419.pt'
print("===start load param===")
net.load_state_dict(torch.load(PATH_NET))
net.eval()
print("===successfully load net===")

print("===palm print recognition test===")
with torch.no_grad():
    # print("===start generating gallery-dl-feature===")
    if not already_prepared:
        feature_gallery,code_gallery,palmmatrix,gallery_label = read_image_and_label(session1_dataloader)
        pca = PCA(n_components=80).fit(palmmatrix)
        # n_components = 200
        # eigenpalms = pca.components_[:n_components]
        # weights = eigenpalms @ (palmmatrix - pca.mean_).T
        # weights = weights.T
        pca_weights = pca.transform(palmmatrix)
        gallery_label = torch.tensor(gallery_label, device = 'cpu').numpy()
        lda = LDA(n_components=80).fit(palmmatrix,gallery_label)
        weights = lda.transform(palmmatrix)
        print("===completed!===")

        # if if_need_balance:
        #     print('===begin balance feature dimension===')
        #     code_gallery,weights =balance_dimension(code_gallery,weights)
        #     print(code_gallery.shape)
        #     print(weights.shape)
        feature_gallery = feature_gallery.cpu().numpy()
        # feature_gallery = feature_norm(feature_gallery)
        # code_gallery = feature_norm(code_gallery)

        print('===start merge features!===')
        classic_cca = CCA(n_components=80)
        classic_cca.fit(feature_gallery, pca_weights)
        dl_cca,pca_cca = classic_cca.transform(feature_gallery, pca_weights)
        merge_gallery = np.append(dl_cca,pca_cca,axis=1)
        np.save(model_folder+'palmmatrix.npy',palmmatrix)
        np.save(model_folder + 'gallery_label.npy', torch.tensor(gallery_label, device='cpu'))
        np.save(model_folder+'dl_feature.npy',feature_gallery)
        np.save(model_folder+'code_feature.npy',code_gallery)
        dump(classic_cca,model_folder+'cca.joblib')
        np.save(model_folder+'merge_feature.npy',merge_gallery)
        dump(pca,model_folder+'pca.joblib')
        dump(lda,model_folder+'lda.joblib')
    else:
        feature_gallery = np.load(model_folder+'dl_feature.npy')
        code_gallery = np.load(model_folder+'code_feature.npy')
        gallery_label = np.load(model_folder + 'gallery_label.npy')
        merge_gallery= np.load(model_folder + 'merge_feature.npy')
        palmmatrix = np.load(model_folder+'palmmatrix.npy')
        classic_cca = load(model_folder+'cca.joblib')
        lda = load(model_folder+'lda.joblib')
        pca = load(model_folder+'pca.joblib')
        n_components = 80
        # eigenpalms = pca.components_[:n_components]
        # weights = eigenpalms @ (palmmatrix - pca.mean_).T
        # weights = weights.T
        weights = lda.transform(palmmatrix)
        pca_weights = pca.transform(palmmatrix)
    # weights = feature_norm(weights)
    # pca_weights = feature_norm(pca_weights)
    # 标准化



    print('===start test!===')
    test_dl_feature,test_code_feature,testmatrix,testlabel = read_image_and_label(session2_dataloader)
    query = lda.transform(testmatrix)
    pca_query = pca.transform(testmatrix)


    test_dl_feature = test_dl_feature.cpu().numpy()
    # query = feature_norm(query)
    # pca_query = feature_norm(pca_query)
    # test_dl_feature = feature_norm(test_dl_feature)

    test_dl_cca,test_pca_cca = classic_cca.transform(test_dl_feature, pca_query)
    test_merge = np.append(test_dl_cca,test_pca_cca,axis=1)



    # calculate dynamic weight
    # dl_weight,lda_weight,compcode_weight = calculate_weight([0.908,0.804,0.77])
    dl_weight, lda_weight, compcode_weight = [0.908, 0.804, 0.8]

    # print(dl_weight)


    idx = 0
    total_correct = 0
    cur_correct = 0
    batch = 0
    while idx < len(test_dl_feature):
        # print(merge_gallery.shape)
        # print(test_merge[idx].shape)
        # break
        vote_box = {}
        # dl-vote
        vote_box = vote(vote_box,feature_gallery,test_dl_feature[idx].reshape(1,-1),0.91)
        vote_box = vote(vote_box,merge_gallery,test_merge[idx].reshape(1,-1),dl_weight)
        vote_box =vote(vote_box,weights,query[idx].reshape(1,-1),lda_weight)
        # print(vote_box)
        vote_box = vote(vote_box,code_gallery,test_code_feature[idx].reshape(1,-1),compcode_weight)
        # print(vote_box)
        # break
        best_match = max(vote_box,key=vote_box.get)
        # print(best_match)
        # print('%d =? %d'%(palmlabel[best_match],test_labels[idx]))
        if gallery_label[best_match] == testlabel[idx]:
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
    print('TOTAL CORRECT RATE: %.3f' % (total_correct / len(testmatrix)))








