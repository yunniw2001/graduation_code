import numpy as np
import torchvision.transforms as transforms
from my_transformer import MeanFiltersTransform, MedianFiltersTransform, GaussFiltersTransform, \
    GaussianFiltersTransformUnsharpMask, MedianFiltersTransformUnsharpMask, MeanFiltersTransformUnsharpMask
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

preprocessing = None
testprocessing = None


def prepare_transform_for_image():
    global preprocessing
    global testprocessing
    rotation = transforms.RandomRotation(5)
    resized_cropping = transforms.Resize((224, 224))
    contrast_brightness_adjustment = transforms.ColorJitter(brightness=0.5, contrast=0.5)
    color_shift = transforms.ColorJitter(hue=0.14)
    preprocessing = transforms.Compose(
        [
            transforms.RandomApply(
                [rotation, contrast_brightness_adjustment, color_shift], 0.6),
            MedianFiltersTransform(),
            resized_cropping,
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]
    )
    testprocessing = transforms.Compose(
        [MedianFiltersTransform(),
            resized_cropping,
         transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)
         ]
    )


def read_image_and_label(labelpath, imgpath, state='train'):
    label_file = open(labelpath, 'r')
    content = label_file.readlines()
    class_size = 0
    min_size = 1000000
    lbp_g = []
    labels = []

    # print('===start read images and labels, generate uniform LBP gallery!===')
    for line in content:
        img_name, img_label = line.split(' ')[0], int(line.split(' ')[1])
        tmp_image_path = imgpath + img_name
        # print(tmp_image_path)
        if img_label > class_size:
            class_size = img_label
        if img_label < min_size:
            min_size = img_label
        if state == 'train':
            cur = testprocessing(Image.open(tmp_image_path).convert('L'))
        else:
            cur = testprocessing(Image.open(tmp_image_path).convert('L'))
        # print(cur.size())
        cur = cur.numpy()[0]
        cur_lbp = local_binary_pattern(cur, 8, 2,'default')
        lbp_g.append(cur_lbp)
        labels.append(img_label)
    return lbp_g, labels


radius = 1
n_points = 8 * radius
dataset = 'CASIA'
# 读入图像
img_PATH = '/home/ubuntu/dataset/'+dataset+'/session/session1/'
label_PATH = '/home/ubuntu/dataset/'+dataset+'/session/session1_label.txt'
save_gallery = '/home/ubuntu/graduation_model/gallery_texture.npy'
save_label = '/home/ubuntu/graduation_model/label_texture.npy'
prepare_transform_for_image()

label_file = open(label_PATH, 'r')
content = label_file.readlines()
class_size = 0
min_size = 1000000
lbp_gallery = []
palmlabel = []
already_processed = False

print('===start read images and labels, generate uniform LBP gallery!===')
if not already_processed:
    lbp_gallery, palmlabel = read_image_and_label(label_PATH, img_PATH)
    lbp_gallery = np.array(lbp_gallery)
    palmlabel = np.array(palmlabel)
    # np.save(save_gallery,lbp_gallery)
    # np.save(save_label,palmlabel)
else:
    lbp_gallery = np.load(save_gallery)
    palmlabel = np.load(save_label)
# print(len(lbp_gallery))
# print(image.shape)
# plt.imshow(lbp,'gray')
# plt.show()
print('===DONE!===')
# 直方图
hist_gallery = []
n_bins = int(lbp_gallery[0].max() + 1)
print('===start generating histogram!===')
for tmp_lbp in lbp_gallery:
    hist, bins = np.histogram(tmp_lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist_gallery.append(hist)
hist_gallery = np.array(hist_gallery)
# hist_gallery = lbp_gallery
print('===DONE!===')
# print(hist)
# plt.hist(lbp.ravel(),bins=n_bins,range=(0,n_bins),density=True)
# plt.show()

# 测试
print('===start test!===')
testimg_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session2/'
testlabel_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session2_label.txt'
print('===start load test image!===')
test_lbp_gallery, test_labels = read_image_and_label(testlabel_PATH, testimg_PATH)
print(len(test_lbp_gallery))
print('===load success! generate LBP success!===')

test_hist = test_lbp_gallery
print('===start generating test histogram!===')
for tmp_lbp in test_lbp_gallery:
    hist, bins = np.histogram(tmp_lbp.ravel(), bins=n_bins, range=(0, n_bins))
    test_hist.append(hist)
# print(len(test_hist))
print('===DONE!===')

print('===start recognition===')
idx = 0
cur_correct = 0
total_correct = 0
batch = 0
while idx < len(test_lbp_gallery):
    tmp_hist = test_hist[idx].reshape(1, -1)
    # print(tmp_hist.shape)
    # print(hist_gallery.shape)
    cos_similarity = cosine_similarity(hist_gallery, tmp_hist)
    # print(cos_similarity.shape)
    best_match = np.argmax(cos_similarity)
    # print(best_match)
    # print('%d =? %d'%(palmlabel[best_match],test_labels[idx]))
    if palmlabel[best_match] == test_labels[idx]:
        cur_correct += 1
        total_correct += 1
    if (idx + 1) % 100 == 0:
        print('batch %d: correct rate = %.2f' % (batch, cur_correct / 100))
        cur_correct = 0
        batch += 1
    idx += 1
print('TOTAL CORRECT RATE: %.2f' % (total_correct / len(test_hist)))
