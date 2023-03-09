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

        # images = images.to(device)
        # label = label.to(device)
        labels.extend(label)
        # feats, normalized_feature = net(images, None)
        if i == 0:
            testmatrix = cur
            code_g = cur_code
            # dl_g = normalized_feature
            # dl_g = torch.cat([normalized_feature, torch.flip(normalized_feature, [1])], 1)
        else:
            testmatrix = np.append(testmatrix, cur, axis=0)
            code_g = np.append(code_g,cur_code,axis=0)
            # tmp = torch.cat([normalized_feature, torch.flip(normalized_feature, [1])], 1)
            # dl_g = torch.cat([dl_g, tmp], 0)
        # code_g.append(cur_code)
        # print(testmatrix.shape)

        if (i + 1) % 10 == 0:
            print('[batch: %d DONE!]' % (i))
    testmatrix = testmatrix.reshape(*testmatrix.shape[:-2],-1)
    print("===completed!===")
    return code_g, testmatrix, labels

def balance_dimension(matrix_a,matrix_b):
    mixed = np.append(matrix_a, matrix_b, axis=1)
    return np.split(mixed, 2, axis=1)



batch_size = 40
num_class = 480
feature_size = 128
lr = 0.001
epochs = 1000

prepare_transform_for_image()
my_gabor_filter = Gabor_filters()
my_gabor_filter.build_filters()

dataset = 'CASIA'
root_path = '/home/ubuntu/dataset/'+dataset+'/test_session/'
session1_dataset = MyDataset(root_path+'session1/',
                             root_path+'session1_label.txt', testprocessing)
session2_dataset = MyDataset(root_path+'session2/',
                             root_path+'session2_label.txt', testprocessing)
session1_dataloader = DataLoader(dataset=session1_dataset, batch_size=batch_size, shuffle=False)
session2_dataloader = DataLoader(dataset=session2_dataset, batch_size=batch_size, shuffle=True)
if_need_balance=False
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# net = ResNet.resnet34()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net.to(device)

# 加载参数
# PATH_NET = '/home/ubuntu/graduation_model/deeplearning/model_net_419.pt'
# print("===start load param===")
# net.load_state_dict(torch.load(PATH_NET))
# net.eval()
# print("===successfully load net===")

print("===palm print recognition test===")
with torch.no_grad():
    # print("===start generating gallery-dl-feature===")
    code_gallery,palmmatrix,gallery_label = read_image_and_label(session1_dataloader)
    pca = PCA().fit(palmmatrix)
    n_components = 200
    eigenpalms = pca.components_[:n_components]
    weights = eigenpalms @ (palmmatrix - pca.mean_).T
    weights = weights.T
    print("===completed!===")

    if if_need_balance:
        print('===begin balance feature dimension===')
        code_gallery,weights =balance_dimension(code_gallery,weights)
        print(code_gallery.shape)
        print(weights.shape)


    print('===start merge features!===')

    # print(feature_gallery.shape)
    # print(palmmatrix.shape)
    cca = CCA(n_components=125)
    # print(code_gallery.shape)
    # print(weights.shape)
    cca.fit(code_gallery,weights)
    # dl_r,pca_r = cca.x_weights_,cca.y_weights_
    # print(dl_r.shape)
    # print(pca_r.shape)
    # dl_cca =np.matmul(feature_gallery,cca.x_weights_)
    # pca_cca = np.matmul(weights,cca.y_weights_)
    code_cca,pca_cca = cca.transform(code_gallery,weights)
    merge_gallery = np.append(code_cca,pca_cca,axis=1)
    # print(dl_cca.shape)
    # print(pca_cca.shape)
    print(merge_gallery.shape)
    print('===start test!===')
    test_code_feature,testmatrix,testlabel = read_image_and_label(session2_dataloader)
    query = eigenpalms @ (testmatrix-pca.mean_).T
    query = query.T

    if if_need_balance:
        test_code_feature,query=balance_dimension(test_code_feature,query)
    test_code_cca,test_pca_cca = cca.transform(test_code_feature,query)
    test_merge = np.append(test_code_cca,test_pca_cca,axis=1)
    idx = 0
    total_correct = 0
    cur_correct = 0
    batch = 0
    while idx < len(test_merge):
        # print(merge_gallery.shape)
        # print(test_merge[idx].shape)
        # break
        cos_similarity = cosine_similarity(merge_gallery,test_merge[idx].reshape(1,-1))
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








