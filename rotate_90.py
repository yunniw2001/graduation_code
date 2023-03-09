import os

import cv2
def rotate_image(ori_path,dest_path):
    img = cv2.imread(ori_path)

    print('Before rotate image shape is', img.shape)
    img90 = cv2.flip(img, 0)  # 垂直翻转图像
    img90 = cv2.transpose(img90)  # 转置图像
    print('After rotate image shape is', img90.shape)
    cv2.imwrite(dest_path, img90) # 保存旋转后的图像
    return

orifolder = '/home/ubuntu/dataset/CASIA/test_session0/session2/'
dest_folder = '/home/ubuntu/dataset/CASIA/test_session/session2/'
files = os.listdir(orifolder)
for item in files:
    if not item.endswith('.jpg'):
        continue
    # tmp = item.split('_')
    # if not tmp[2] =='l':
    #     continue
    # print(item+' start!')
    rotate_image(orifolder+item,dest_folder+item)
    # break
# rotate_image('/home/ubuntu/dataset/IITD/train/001_l_1.bmp','/home/ubuntu/dataset/IITD/test_session2/session1/001_l_1.bmp')

