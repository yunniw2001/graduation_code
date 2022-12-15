import os


def make_tongji_test_label(img_dir, text_dir):
    files = os.listdir(img_dir)
    write_file = open(text_dir, 'a+')
    num = 0
    for item in files:
        # if num>=4800:
        #     break
        num+=1
        tmp_item = list(item[0:5])
        if tmp_item[0] == '1':
            tmp_item[0] = '0'
        tmp_item = ''.join(tmp_item)
        tmp_item = int(tmp_item)
        belong_index = (tmp_item - 1) // 10
        write_file.write(item + ' ' + str(belong_index) + '\n')
    write_file.close()

if __name__ == '__main__':
    make_tongji_test_label('/home/ubuntu/dataset/tongji/test/session2/',
                             '/home/ubuntu/dataset/tongji/test/session2_label.txt')