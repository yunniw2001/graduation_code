import torch
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image

import ResNet
from my_transformer import MeanFiltersTransform, MedianFiltersTransform, GaussFiltersTransform, \
    GaussianFiltersTransformUnsharpMask, MedianFiltersTransformUnsharpMask, MeanFiltersTransformUnsharpMask

preprocessing = []
test_preprocessing = []


def prepare_transform_for_image():
    global preprocessing
    global test_preprocessing
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
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    test_preprocessing = transforms.Compose(
        [resized_cropping,
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


features = {}
def get_activation(name):
    def hook_func(module, input, output):
        features[name] = output.detach()
    return hook_func


def show_features(features):
    for keys, values in features.items():
        values = values.permute(1,0,2,3)
        image_name = '/home/ubuntu/output_img/'+keys+'.png'
        torchvision.utils.save_image(values,image_name)

prepare_transform_for_image()
writer = SummaryWriter(log_dir='/home/ubuntu/tensorboard_data/')
path_img = '/home/ubuntu/dataset/tongji/train/00001.bmp'
img = Image.open(path_img).convert('RGB')
if test_preprocessing is not None:
    img = test_preprocessing(img).unsqueeze(0)

net = ResNet.resnet34()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# 加载参数
PATH_NET = '/home/ubuntu/graduation_model/deeplearning/model_net_419.pt'
print("===start load param===")
net.load_state_dict(torch.load(PATH_NET))
net.eval()
print("===successfully load net===")

# 注册钩子
net.layer3.register_forward_hook(get_activation('layer3'))

# 前向传播
with torch.no_grad():
    img = img.cuda()
    feats, normalized_feature = net(img, None)

show_features(features)



