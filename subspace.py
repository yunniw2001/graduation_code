import cv2
import numpy as np
import torchvision
import torchvision.transforms as transforms
from my_transformer import MeanFiltersTransform, MedianFiltersTransform, GaussFiltersTransform, \
    GaussianFiltersTransformUnsharpMask, MedianFiltersTransformUnsharpMask, MeanFiltersTransformUnsharpMask
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

preprocessing = None


def prepare_transform_for_image():
    global preprocessing
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
                [rotation, contrast_brightness_adjustment, smooth_or_sharpening,color_shift], 0.6),
            resized_cropping,
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]
    )


# 读入图像
img_PATH = '/home/ubuntu/dataset/CASIA/train_ROI/'
label_PATH = '/home/ubuntu/dataset/CASIA/train_label.txt'
save_PATH = '/home/ubuntu/graduation_model/palmmatrix.npy'
prepare_transform_for_image()
palms = []
palmmatrix = []
palmlabel = []
already_processed = True

label_file = open(label_PATH, 'r')
content = label_file.readlines()
class_size = 0
if not already_processed:
    print('===start read train dataset!===')
    for line in content:
        img_name, img_label = line.split(' ')[0], int(line.split(' ')[1])
        tmp_image_path = img_PATH + img_name
        if img_label>class_size:
            class_size = img_label
        cur = preprocessing(Image.open(tmp_image_path).convert('L'))
        # print(cur.size())
        cur = cur.permute(1, 2, 0).detach().numpy()
        palms.append(cur)
        palmmatrix.append(cur.flatten())
        palmlabel.append(img_label)
        # if i >= 20:
        #     break
        # break
    palmmatrix = np.array(palmmatrix)
    np.save(save_PATH, palmmatrix)
else:
    print('===start load palm matrix!===')
    palmmatrix = np.load(save_PATH)
    print('===start read label file!===')
    for line in content:
        img_name, img_label = line.split(' ')[0], int(line.split(' ')[1])
        if img_label>class_size:
            class_size = img_label
        palmlabel.append(img_label)

# 预览图片
# fig, axes = plt.subplots(4,5,sharex=True,sharey=True,figsize=(8,10))
# for i in range(20):
#     axes[i%4][i//4].imshow(palms[i], cmap="gray")
# plt.show()

print('===%d images Done! %d classes in total!==='%(len(content),class_size+1))
print(palmmatrix)
print(palmmatrix.shape)
# PCA
# pca = PCA().fit(palmmatrix)


