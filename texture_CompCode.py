from scipy.signal import convolve2d
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

def extract_CompCode(image):
    num_filters = 6
    num_points = 35
    sigma = 1.5

    theta_L =np.arange(1,num_filters)*np.pi/num_filters
    (x,y) = np.meshgrid(np.arange(0,35,1),np.arange(0,35,1))
    x0,y0 = np.shape(x)[0]/2,np.shape(y)[0]/2 # why?
    kappa = np.sqrt(2*np.log(2))*((np.power(2,sigma)+1)/(np.power(2,sigma)-1))
    omega = kappa/sigma

    Psi = {}  # where the filters are stored
    gabor_responses = []
    for i in range(0, len(theta_L)):
        xp = (x - x0) * np.cos(theta_L[i]) + (y - y0) * np.sin(theta_L[i])
        yp = -(x - x0) * np.sin(theta_L[i]) + (y - y0) * np.cos(theta_L[i])
        # Directional Gabor Filter
        Psi[str(i)] = (-omega / (np.sqrt(2 * np.pi)) * kappa) * \
                      np.exp(
                          (-np.power(omega, 2) / (8 * np.power(kappa, 2))) * (4 * np.power(xp, 2) + np.power(yp, 2))) * \
                      (np.cos(omega * xp) - np.exp(-np.power(kappa, 2) / 2))

        # # used for debugging... #1
        # plt.subplot(2,3,i+1)
        # plt.imshow(Psi[str(i)], cmap='jet')
        filtered = convolve2d(image, Psi[str(i)], mode='same', boundary='symm')
        # # used for debugging #2
        # plt.imshow(filtered)
        # plt.show()
        gabor_responses.append(filtered)

    # plt.show() #1
    gabor_responses = np.array(gabor_responses)

    compcode_orientations = np.argmin(gabor_responses, axis=0)
    compcode_magnitude = np.min(gabor_responses, axis=0)

    return compcode_orientations, compcode_magnitude

def read_image_and_label(labelpath, imgpath, state='train'):
    label_file = open(labelpath, 'r')
    content = label_file.readlines()
    code_g = []
    labels = []
    num = 0

    # print('===start read images and labels, generate uniform LBP gallery!===')
    for line in content:
        img_name, img_label = line.split(' ')[0], int(line.split(' ')[1])
        tmp_image_path = imgpath + img_name
        # print(tmp_image_path)
        if state == 'train':
            cur = testprocessing(Image.open(tmp_image_path).convert('L'))
        else:
            cur = testprocessing(Image.open(tmp_image_path).convert('L'))
        # print(cur.size())
        cur = cur.numpy()[0]
        cur_code,_ = extract_CompCode(cur)
        code_g.append(cur_code)
        labels.append(img_label)
        num+=1
        if num%100 == 0:
            print('%d images Done!'%num)
    return code_g, labels


dataset = 'IITD'
# 读入图像
img_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session1/'
label_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session1_label.txt'
save_gallery = '/home/ubuntu/graduation_model/gallery_texture.npy'
save_label = '/home/ubuntu/graduation_model/label_texture.npy'
prepare_transform_for_image()

label_file = open(label_PATH, 'r')
content = label_file.readlines()
code_gallery = []
already_processed = False

print('===start read images and labels, generate gallery!===')
if not already_processed:
    code_gallery, palmlabel = read_image_and_label(label_PATH, img_PATH)
    code_gallery = np.array(code_gallery)
    palmlabel = np.array(palmlabel)
    np.save(save_gallery,code_gallery)
    np.save(save_label,palmlabel)
else:
    code_gallery = np.load(save_gallery)
    palmlabel = np.load(save_label)
# print(len(lbp_gallery))
# print(image.shape)
# plt.imshow(lbp,'gray')
# plt.show()
print('===DONE!===')

# 测试
print('===start test!===')
testimg_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session2/'
testlabel_PATH = '/home/ubuntu/dataset/'+dataset+'/test_session/session2_label.txt'
print('===start load test image!===')
test_code_gallery, test_labels = read_image_and_label(testlabel_PATH, testimg_PATH)
print(len(test_code_gallery))
print('===load success! generate code success!===')


print('===start recognition===')
idx = 0
cur_correct = 0
total_correct = 0
batch = 0
while idx < len(test_code_gallery):
    tmp_code = test_code_gallery[idx].reshape(1, -1)
    # print(tmp_hist.shape)
    # print(hist_gallery.shape)
    cos_similarity = cosine_similarity(code_gallery, tmp_code)
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
print('TOTAL CORRECT RATE: %.2f' % (total_correct / len(test_code_gallery)))
