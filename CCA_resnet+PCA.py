import numpy as np
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

# random apply preprocessing
preprocessing = []
testprocessing = []
to_greyscale = None


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


def calculate_cos_similarity(test_feature):
    # print(test_feature.size())
    # print(feature_gallery.size())
    cosine_similarity = torch.cosine_similarity(feature_gallery, test_feature.unsqueeze(0), dim=1, eps=1e-08)
    return cosine_similarity


def calculate_accuracy(logits, label):
    _, pred = torch.max(logits.data, 1)
    i = 0
    while i < len(pred):
        pred[i] = gallery_label[pred[i]]
        i += 1
    return (label.data == pred).float().mean()

def read_image_and_label(dataloader):
    labels = []
    dl_g = []
    testmatrix = []

    print("===start generating gallery-dl-feature===")
    for i, data in enumerate(dataloader):
        images, label = data
        cur = to_greyscale(images)

        cur = cur.detach().numpy()
        cur = np.squeeze(cur)
        # print(cur.shape)

        images = images.to(device)
        label = label.to(device)
        labels.extend(label)
        feats, normalized_feature = net(images, None)
        if i == 0:
            testmatrix = cur
            # dl_g = normalized_feature
            dl_g = torch.cat([normalized_feature, torch.flip(normalized_feature, [1])], 1)
        else:
            testmatrix = np.append(testmatrix, cur, axis=0)
            tmp = torch.cat([normalized_feature, torch.flip(normalized_feature, [1])], 1)
            dl_g = torch.cat([dl_g, tmp], 0)

        # print(testmatrix.shape)

        if (i + 1) % 10 == 0:
            print('[batch: %d DONE!]' % (i))
    testmatrix = testmatrix.reshape(*testmatrix.shape[:-2],-1)
    print("===completed!===")
    return dl_g, testmatrix, labels



batch_size = 40
num_class = 480
feature_size = 128
lr = 0.001
epochs = 1000

prepare_transform_for_image()
dataset = 'tongji'
root_path = '/home/ubuntu/dataset/'+dataset+'/test_session/'
session1_dataset = MyDataset(root_path+'session1/',
                             root_path+'session1_label.txt', testprocessing)
session2_dataset = MyDataset(root_path+'session2/',
                             root_path+'session2_label.txt', testprocessing)
session1_dataloader = DataLoader(dataset=session1_dataset, batch_size=batch_size, shuffle=False)
session2_dataloader = DataLoader(dataset=session2_dataset, batch_size=batch_size, shuffle=True)
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
    feature_gallery,palmmatrix,gallery_label = read_image_and_label(session1_dataloader)
    pca = PCA().fit(palmmatrix)
    n_components = 200
    eigenpalms = pca.components_[:n_components]
    weights = eigenpalms @ (palmmatrix - pca.mean_).T
    weights = weights.T
    print("===completed!===")

    print('===start merge features!===')
    feature_gallery = feature_gallery.cpu().numpy()
    # print(feature_gallery.shape)
    # print(palmmatrix.shape)
    cca = CCA(n_components=125)
    cca.fit(feature_gallery,weights)
    # dl_r,pca_r = cca.x_weights_,cca.y_weights_
    # print(dl_r.shape)
    # print(pca_r.shape)
    # dl_cca =np.matmul(feature_gallery,cca.x_weights_)
    # pca_cca = np.matmul(weights,cca.y_weights_)
    dl_cca,pca_cca = cca.transform(feature_gallery,weights)
    merge_gallery = np.append(dl_cca,pca_cca,axis=1)
    # print(dl_cca.shape)
    # print(pca_cca.shape)
    print(merge_gallery.shape)
    print('===start test!===')
    test_dl_feature,testmatrix,testlabel = read_image_and_label(session2_dataloader)
    query = eigenpalms @ (testmatrix-pca.mean_).T
    query = query.T
    test_dl_feature = test_dl_feature.cpu().numpy()
    test_dl_cca,test_pca_cca = cca.transform(test_dl_feature,query)
    test_merge = np.append(test_dl_cca,test_pca_cca,axis=1)
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








