import os
import shutil
import sys


# 同济
def make_tongji_test_label(img_dir, text_dir):
    files = os.listdir(img_dir)
    write_file = open(text_dir, 'a+')
    num = 0
    for item in files:
        # if num>=4800:
        #     break
        num += 1
        tmp_item = list(item[0:5])
        if tmp_item[0] == '1':
            tmp_item[0] = '0'
        tmp_item = ''.join(tmp_item)
        tmp_item = int(tmp_item)
        belong_index = (tmp_item - 1) // 10
        write_file.write(item + ' ' + str(belong_index) + '\n')
    write_file.close()


# CASIA
def move_to_root_dir(root_path, cur_path):
    for filename in os.listdir(cur_path):
        if os.path.isfile(os.path.join(cur_path, filename)):
            shutil.move(os.path.join(cur_path, filename), os.path.join(root_path, filename))
        elif os.path.isdir(os.path.join(cur_path, filename)):
            move_to_root_dir(root_path, os.path.join(cur_path, filename))
        else:
            sys.exit("Should never reach here.")
    # remove empty folders
    if cur_path != root_path:
        os.rmdir(cur_path)


def make_CASIA_test_label(img_dir, text_dir):
    files = os.listdir(img_dir)
    write_file = open(text_dir, 'a+')
    for item in files:
        if not item.endswith('.jpg'):
            continue
        flag = 0
        tmp = item.split('_')
        idx = int(tmp[0])
        belong_index = (idx - 1) * 2
        if tmp[2] == 'r' and flag == 0:
            belong_index += 1
            flag = 1
        write_file.write(item + ' ' + str(belong_index) + '\n')
        # print(item + ' ' + str(belong_index) + '\n')
        # break
    write_file.close()


def make_IITD_test_label(img_dir, text_dir):
    files = os.listdir(img_dir)
    write_file = open(text_dir, 'a+')
    for item in files:
        if not item.endswith('.bmp'):
            continue
        flag = 0
        tmp = item.split('_')
        idx = int(tmp[0])
        belong_index = (idx - 1) * 2
        if tmp[1] == 'r' and flag == 0:
            belong_index += 1
            flag = 1
        write_file.write(item + ' ' + str(belong_index) + '\n')
        # print(item + ' ' + str(belong_index) + '\n')
        # break
    write_file.close()


def make_session_CASIA(ori_path, session1_path, session2_path):
    files = os.listdir(ori_path)
    for item in files:
        if not item.endswith('jpg'):
            continue
        tmp = item.split('_')
        print(item)
        # print(tmp[3].split('.'))
        if int(tmp[3].split('.')[0]) > 4:
            shutil.copy(ori_path + item, session2_path)
        else:
            shutil.copy(ori_path + item, session1_path)


def move_session_IITD(ori_path, session_root_path):
    files = os.listdir(ori_path)
    for item in files:
        if not item.endswith('.bmp'):
            continue
        tmp = item.split('_')
        if int(tmp[2].split('.')[0]) > 3:
            shutil.move(ori_path + item, session_root_path + 'session2/')
        else:
            shutil.move(ori_path + item, session_root_path + 'session1/')


if __name__ == '__main__':
    # make_tongji_test_label('/home/ubuntu/dataset/tongji/test/session2/',
    #                          '/home/ubuntu/dataset/tongji/test/session2_label.txt')
    # root_path = '/Users/wuyihang/Documents/dataset/CASIA-PalmprintV1/train/'
    # to_path = '/Users/wuyihang/Documents/dataset/CASIA-PalmprintV1/train/'
    # lst = os.listdir(root_path)
    # for item in lst:
    #     if item == '.DS_Store' or item.endswith('.jpg'):
    #         continue
    #     cur_path = root_path+str(item)
    #     move_to_root_dir(to_path,cur_path)
    # CASIA标签
    make_CASIA_test_label('/home/ubuntu/dataset/CASIA/test_session/session1/',
                          '/home/ubuntu/dataset/CASIA/test_session/session1_label.txt')
    # gallery分组
    # root_path = '/home/ubuntu/dataset/CASIA/'
    # make_session_CASIA(root_path+'test_ROI/',root_path+'test_session/session1/',root_path+'test_session/session2/')
    # IITD session
    # move_session_IITD('/home/ubuntu/dataset/IITD/test_session/','/home/ubuntu/dataset/IITD/test_session/')
    # make_IITD_test_label('/home/ubuntu/dataset/IITD/test_session/session2',
    #                      '/home/ubuntu/dataset/IITD/test_session/session2_label.txt')