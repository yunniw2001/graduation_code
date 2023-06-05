import sys
import matplotlib
import matplotlib.pyplot as plt
import torch
import ResNet
import cv2 as cv
import numpy as np
import scipy.fftpack as fftpack
from PIL import Image

def display(a):
    for i in a:
        for j in i:
            print(j,end=' ')
        print('')

# read pic
img_path = '/home/ubuntu/dataset/tongji/test_session/session2/14801.bmp'
root = '/home/ubuntu/output_image/'
# img_path = '/home/ubuntu/output_image/doc.png'
# img = Image.open(img_path).convert('RGB')
# print(img.shape)
img = cv.imread(img_path,0)
r,c = img.shape
blank_paper = np.full((r,c),255, dtype = np.uint8)

f = np.fft.fft2(img)
f_paper = np.fft.fft2(blank_paper)

# display(np.abs(f_paper))
# # 通过将零频分量移动到数组中心
fshift = np.fft.fftshift(f)
fshift_blank = np.fft.fftshift(f_paper)
# fshift_blank = f_paper

plt.subplot(232)
# plt.imshow(np.abs(fshift),cmap="gray")
plt.imshow(np.log(1 + np.abs(fshift)),cmap='gray')
plt.title('fft shift')
plt.axis('off')

beta = 0.06
center_r,center_c = int(r/2),int(c/2)
beta_r,beta_c = int(beta*r),int(beta*c)
fshift[center_r-beta_r:center_r+beta_r,center_c-beta_c:center_c+beta_c] = fshift_blank[center_r-beta_r:center_r+beta_r,center_c-beta_c:center_c+beta_c]

# plt.show()

print(blank_paper)
print(np.abs(f_paper))
print(fshift_blank)
ishift = np.fft.ifftshift(fshift)
iimg = np.abs(np.fft.ifft2(ishift))
iimg = (iimg-np.min(iimg))/(np.max(iimg)-np.min(iimg))*255
plt.subplot(231)
plt.imshow(img,cmap='gray')
plt.title('original image')
plt.axis('off')
plt.subplot(234)
plt.imshow(blank_paper,cmap='gray',vmin=0,vmax=255)
plt.title('blank paper')
plt.axis('off')

print(np.max(20*np.log(1 + np.abs(fshift_blank))))
# a = Image.fromarray(f_paper)
# a.save("/home/ubuntu/output_image/hh.jpeg")
plt.subplot(235)
# plt.show()
plt.imshow(20*np.log(1 + np.abs(fshift_blank)),cmap='gray',vmin=0,vmax=255)
plt.title('fft shift blank paper')
plt.axis('off')
plt.subplot(236)
# display(iimg)
plt.imshow(iimg,cmap='gray',vmin=0,vmax=255)
plt.title('ifft')
plt.axis('off')
plt.suptitle('beta = %.3f'%beta)
plt.subplot(233)
plt.imshow(np.log(1 + np.abs(fshift)),cmap='gray')
plt.title('merge')
plt.axis('off')
plt.show()

# print(iimg.shape)
# cv.imwrite(root+'res_blank.jpg',20*np.log(1 + np.abs(fshift_blank)))

