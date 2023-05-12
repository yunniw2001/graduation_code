import sys

import matplotlib.pyplot as plt
import torch
import ResNet
import cv2 as cv
import numpy as np


# load net
net = ResNet.resnet34()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
PATH_NET = '/home/ubuntu/graduation_model/deeplearning/model_net_419.pt'
print("===start load param===")
net.load_state_dict(torch.load(PATH_NET))
net.eval()
print("===successfully load net===")

# read pic
img_path = '/home/ubuntu/dataset/tongji/test_session/session2/14801.bmp'
img = cv.imread(img_path,0)
# print(img.shape)
r,c = img.shape
blank_paper = np.full((r,c),255, dtype = np.uint8)
# print(blank_paper)

# plt.subplot(221)
# plt.imshow(img, cmap="gray", vmin=0,vmax=255)
# plt.title('Original Palm')
# plt.axis('off')
# plt.subplot(223)
# plt.imshow(blank_paper,cmap="gray", vmin=0,vmax=255)
# plt.title('blank paper')
# plt.axis('off')
# plt.show()
# sys.exit()
# fft
f = np.fft.fft2(img)
f_paper = np.fft.fft2(blank_paper)
# 通过将零频分量移动到数组中心
fshift = np.fft.fftshift(f)
fshift_blank = np.fft.fftshift(f_paper)
plt.subplot(142)
plt.imshow(20*np.log(np.abs(fshift)),cmap="gray", vmin=0,vmax=255)
plt.title('fft shift')
plt.axis('off')
# plt.subplot(224)
# plt.imshow(np.abs(fshift_blank),cmap="gray", vmin=0,vmax=255)
# plt.title('fft shift blank paper')
# plt.axis('off')
# plt.show()

# fshift_blank = np.full((r,c),0, dtype = np.uint8)
# filter
beta = 0.01
center_r,center_c = int(r/2),int(c/2)
beta_r,beta_c = int(beta*r),int(beta*c)
fshift[center_r-beta_r:center_r+beta_r,center_c-beta_c:center_c+beta_c] = fshift_blank[center_r-beta_r:center_r+beta_r,center_c-beta_c:center_c+beta_c]
plt.subplot(143)
plt.imshow(np.abs(fshift),cmap="gray", vmin=0,vmax=255)
plt.title('merge')
plt.axis('off')
# plt.show()

ishift = np.fft.ifftshift(fshift)
iimg = np.abs(np.fft.ifft2(ishift))
plt.subplot(141)
plt.imshow(img,cmap="gray", vmin=0,vmax=255)
plt.title('original image')
plt.axis('off')
plt.subplot(144)
plt.imshow(iimg,cmap="gray", vmin=0,vmax=255)
plt.title('ifft')
plt.axis('off')
plt.suptitle('beta = %2f'%beta)
plt.show()

