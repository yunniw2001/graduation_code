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
from sklearn.metrics.pairwise import cosine_similarity
from scipy import ndimage as ndi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


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
        output = (winner / np.max(np.max(winner))) * 255
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

def cca_merge_feature(tmp_cca,feature1,feature2,mode = 'test'):
    if not mode == 'test':
        tmp_cca.fit(feature1,feature2)
    cca1,cca2 = tmp_cca.transform(feature1,feature2)
    return np.append(cca1,cca2,axis=1)

batch_size = 40
num_class = 480
feature_size = 128
lr = 0.001
epochs = 1000

prepare_transform_for_image()
my_gabor_filter = Gabor_filters()
my_gabor_filter.build_filters()

dataset = 'IITD'
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
# device = "cpu"
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
    feature_gallery,code_gallery,palmmatrix,gallery_label = read_image_and_label(session1_dataloader)
    # pca = PCA().fit(palmmatrix)
    # n_components = 200
    # eigenpalms = pca.components_[:n_components]
    # weights = eigenpalms @ (palmmatrix - pca.mean_).T
    # weights = weights.T
    gallery_label = torch.tensor(gallery_label, device='cpu').numpy()
    lda = LDA(n_components=80).fit(palmmatrix,gallery_label)
    pca = PCA(n_components=120).fit(palmmatrix)
    weights = pca.transform(palmmatrix)
    # weights = lda.transform(palmmatrix)
    print("===completed!===")

    # if if_need_balance:
    #     print('===begin balance feature dimension===')
    #     code_gallery,weights =balance_dimension(code_gallery,weights)
    #     print(code_gallery.shape)
    #     print(weights.shape)


    print('===start merge features!===')
    feature_gallery = feature_gallery.cpu().numpy()
    # print(feature_gallery.shape)
    # print(palmmatrix.shape)
    classic_cca = CCA(n_components=120)
    dl_cca = CCA(n_components=60)
    # print(code_gallery.shape)
    # print(weights.shape)
    # code_gallery =feature_gallery
    # weights = feature_gallery
    # classic_cca.fit(feature_gallery, weights)
    # code_cca,pca_cca = classic_cca.transform(feature_gallery, weights)
    merge_gallery = cca_merge_feature(classic_cca,feature_gallery,weights,'train')
    # mergeallfeature_gallery = cca_merge_feature(dl_cca,merge_gallery,code_gallery,'train')
    mergeallfeature_gallery = merge_gallery
    print('===start test!===')
    test_dl_feature,test_code_feature,testmatrix,testlabel = read_image_and_label(session2_dataloader)
    # query = lda.transform(testmatrix)
    query = pca.transform(testmatrix)
    test_dl_feature = test_dl_feature.cpu().numpy()

    test_merge = cca_merge_feature(classic_cca,test_dl_feature,query)

    # mergeallfeature_test = cca_merge_feature(dl_cca,test_merge,test_code_feature)
    mergeallfeature_test = test_merge

    idx = 0
    total_correct = 0
    cur_correct = 0
    batch = 0
    while idx < len(test_merge):
        # print(merge_gallery.shape)
        # print(test_merge[idx].shape)
        # break
        cos_similarity = cosine_similarity(mergeallfeature_gallery,mergeallfeature_test[idx].reshape(1,-1))
        best_match = np.argmax(cos_similarity)
        # print(best_match)
        # print('%d =? %d'%(palmlabel[best_match],test_labels[idx]))
        if gallery_label[best_match] == testlabel[idx]:
            cur_correct += 1
            total_correct += 1
        if (idx + 1) % 100 == 0:
            print('batch %d: correct rate = %.3f' % (batch, cur_correct / 100))
            cur_correct = 0
            batch += 1
        idx += 1
    print('TOTAL CORRECT RATE: %.3f' % (total_correct / len(testmatrix)))








