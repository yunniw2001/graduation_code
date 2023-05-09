import os
import shutil
import sys


# 同济
def make_tongji_test_label(img_dir, text_dir):
    files = os.listdir(img_dir)
    write_file = open(text_dir, 'w')
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

def make_session_tongji(ori_path, session1_path, session2_path):
    files = os.listdir(ori_path)
    for item in files:
        if not item.endswith('bmp'):
            continue
        tmp = item.split('.')[0][0]
        # print(item)
        # print(tmp[3].split('.'))
        if int(tmp[0]) == 1:
            print(item)
            shutil.copy(ori_path + item, session2_path)
        else:
            shutil.copy(ori_path + item, session1_path)
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
    write_file = open(text_dir, 'w')
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
def make_session_CASIA(ori_path, session1_path, session2_path):
    files = os.listdir(ori_path)
    for item in files:
        if not item.endswith('jpg'):
            continue
        tmp = item.split('_')
        # print(item)
        # print(tmp[3].split('.'))
        if int(tmp[3].split('.')[0]) > 4:
            shutil.copy(ori_path + item, session2_path)
        else:
            shutil.copy(ori_path + item, session1_path)

# IITD
def make_IITD_test_label(img_dir, text_dir):
    files = os.listdir(img_dir)
    write_file = open(text_dir, 'w')
    # print(files)
    files.sort()
    # print(files)
    cur_idx = 0
    for item in files:
        if not item.endswith('.bmp'):
            continue
        tmp = item.split('_')
        idx = int(tmp[0])
        belong_index = (idx - 1) * 2
        if tmp[1] == 'r':
            belong_index += 1
        # elif tmp[1] =='l' and cur_idx != idx:
        #     cur_idx = idx
        write_file.write(item + ' ' + str(belong_index) + '\n')
        # print(item + ' ' + str(belong_index) + '\n')
        # break
    write_file.close()


def make_left_right_IITD(ori_path,direction,to_dir):
    files = os.listdir(ori_path)
    for item in files:
        if not item.endswith('.bmp'):
            continue
        tmp = item.split('_')
        new_name = tmp[0]+'_'+direction+'_'+tmp[1]
        shutil.move(ori_path + item, to_dir + new_name)


def move_session_IITD(ori_path, session_root_path):
    files = os.listdir(ori_path)
    for item in files:
        if not item.endswith('.bmp'):
            continue
        tmp = item.split('_')
        if int(tmp[2].split('.')[0]) > 5:
            shutil.copy(ori_path + item, session_root_path + 'session2/')
        else:
            shutil.copy(ori_path + item, session_root_path + 'session1/')
    make_IITD_test_label('/home/ubuntu/dataset/IITD/session/session2',
                         '/home/ubuntu/dataset/IITD/session/session2_label.txt')
    make_IITD_test_label('/home/ubuntu/dataset/IITD/session/session1',
                         '/home/ubuntu/dataset/IITD/session/session1_label.txt')

def check_label(s1,s2,direction):
    session1 = os.listdir(s1)
    session2 = os.listdir(s2)

    for i in range(231):
        file_name = "{:03d}".format(i)+'_'+direction
        if any(x.startswith(file_name) for x in session2):
            if any(y.startswith(file_name) for y in session1):
                continue
            else:
                print(s2+file_name)
                del_list = [idx for idx in session2 if idx.startswith(file_name)]
                for item in del_list:
                    print(item)
                    os.remove(s2+item)
def IITD_copy(from_dir,to_dir):
    file_list = os.listdir(from_dir)
    for item in file_list:
        index = item.split('_')[0]
        if int(index)>=180:
            shutil.copy(from_dir+item,to_dir)

def remove_line():
    line = input()
    res = []
    while line:
        item = line.split('_')
        res.append(item[0].split('}')[0]+'}'+item[1])
        line = input()
    for i in res:
        print(i)
if __name__ == '__main__':
    remove_line()
    # make_session_tongji('/home/ubuntu/dataset/tongji/images/','/home/ubuntu/dataset/tongji/session/session1/','/home/ubuntu/dataset/tongji/session/session2/')
    # make_tongji_test_label('/home/ubuntu/dataset/tongji/session/session2/',
    #                          '/home/ubuntu/dataset/tongji/session/session2_label.txt')
    # root_path = '/Users/wuyihang/Documents/dataset/CASIA-PalmprintV1/train/'
    # to_path = '/Users/wuyihang/Documents/dataset/CASIA-PalmprintV1/train/'
    # lst = os.listdir(root_path)
    # for item in lst:
    #     if item == '.DS_Store' or item.endswith('.jpg'):
    #         continue
    #     cur_path = root_path+str(item)
    #     move_to_root_dir(to_path,cur_path)
    # CASIA标签
    # make_CASIA_test_label('/home/ubuntu/dataset/CASIA/session/session1/',
    #                       '/home/ubuntu/dataset/CASIA/session/session1_label.txt')
    # gallery分组
    # root_path = '/home/ubuntu/dataset/CASIA/'
    # make_session_CASIA('/home/ubuntu/dataset/CASIA/images/','/home/ubuntu/dataset/CASIA/session/session1/','/home/ubuntu/dataset/CASIA/session/session2/')
    # IITD session
    # move_session_IITD('/home/ubuntu/dataset/IITD/images/','/home/ubuntu/dataset/IITD/session/')
    # check_label('/home/ubuntu/dataset/IITD/session/session1/','/home/ubuntu/dataset/IITD/session/session2/','r')
    # make_IITD_test_label('/home/ubuntu/dataset/IITD/session/session1',
    #                      '/home/ubuntu/dataset/IITD/session/session1_label.txt')
    # make_left_right_IITD('/home/ubuntu/dataset/IITD/origin/right/','r','/home/ubuntu/dataset/IITD/images/')
    # IITD_copy('/home/ubuntu/dataset/IITD/session/session2/','/home/ubuntu/dataset/IITD/test_session/session2/')