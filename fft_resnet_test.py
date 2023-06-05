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
    GaussianFiltersTransformUnsharpMask, MedianFiltersTransformUnsharpMask, MeanFiltersTransformUnsharpMask, MyDataset, \
    FFT_converter

# random apply preprocessing
preprocessing = []
gallery_preprocessing = []


def prepare_transform_for_image():
    global preprocessing
    global gallery_preprocessing
    rotation = transforms.RandomRotation(6)
    resized_cropping = transforms.Resize((224, 224))
    my_fft_converter = FFT_converter()
    my_fft_converter.__int__(12)
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
            # my_fft_converter(0.03),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    gallery_preprocessing = transforms.Compose(
        [resized_cropping,
         my_fft_converter,
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])


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


batch_size = 40
num_class = 480
feature_size = 128
lr = 0.001
epochs = 1000

prepare_transform_for_image()
dataset = 'CASIA'
root_path = '/home/ubuntu/dataset/' + dataset + '/test_session/'
session1_dataset = MyDataset(root_path + 'session1/',
                             root_path + 'session1_label.txt', gallery_preprocessing)
session2_dataset = MyDataset(root_path + 'session2/',
                             root_path + 'session2_label.txt', gallery_preprocessing)
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
writer = SummaryWriter(log_dir='/home/ubuntu/tensorboard_data/')
data_iter = iter(session1_dataloader)
images, labels = next(data_iter)
img_grid = torchvision.utils.make_grid(images)
writer.add_image('preview some pictures after', img_grid)
print("===palm print recognition test===")
with torch.no_grad():
    print("===start generating gallery===")
    gallery_label = []
    feature_gallery = []
    for i, data in enumerate(session1_dataloader):
        images, label = data
        images = images.type(torch.FloatTensor)
        images = images.to(device)
        label = label.to(device)
        gallery_label.extend(label)
        feats, normalized_feature = net(images, None)
        if i == 0:
            feature_gallery = torch.cat([normalized_feature, torch.flip(normalized_feature, [1])], 1)
        else:
            tmp = torch.cat([normalized_feature, torch.flip(normalized_feature, [1])], 1)
            feature_gallery = torch.cat([feature_gallery, tmp], 0)
        if (i + 1) % 10 == 0:
            print('[batch: %d DONE!]' % (i))
    print("===generate gallery completed!===")
    print("===start test===")
    average_accuracy = 0
    total = 0
    total_i = 0
    for i, data in enumerate(session2_dataloader):
        images, label = data
        images = images.type(torch.FloatTensor)
        images = images.to(device)
        label = label.to(device)
        feature, normalized_feature = net(images, None)
        tmp = torch.cat([normalized_feature, torch.flip(normalized_feature, [1])], 1)
        prec_logits = []
        k = 0
        for tmp_feature in tmp:
            if k == 0:
                prec_logits = calculate_cos_similarity(tmp_feature)
                prec_logits = torch.unsqueeze(prec_logits, 0)
            else:
                tmp = calculate_cos_similarity(tmp_feature)
                prec_logits = torch.cat([prec_logits, torch.unsqueeze(tmp, 0)], 0)
            k += 1
        average_accuracy += calculate_accuracy(prec_logits, label)
        total += calculate_accuracy(prec_logits, label)
        total_i += 1
        if (i + 1) % 10 == 0:
            print('[batch: %d] accuracy: %.3f' % (i, average_accuracy / 10))
            average_accuracy = 0
    print("general average accuracy: %.3f" % (total / total_i))
