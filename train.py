import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import ResNet
from ArcFace import ArcFace
from my_transformer import MeanFiltersTransform, MedianFiltersTransform, GaussFiltersTransform, \
    GaussianFiltersTransformUnsharpMask, MedianFiltersTransformUnsharpMask, MeanFiltersTransformUnsharpMask, MyDataset

# random apply preprocessing
preprocessing = None


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


def read_txt(text_dir):
    file = open(text_dir, 'r')
    lines = file.readlines()
    epoch = int(lines[0])
    per_idx = int(lines[1])
    cur_idx = lines[2][1:-2].split(',')
    for i in range(len(cur_idx)):
        cur_idx[i] = int(cur_idx[i])
    train_loss = lines[3][1:-2].split(',')
    accuracy = lines[4][1:-2].split(',')
    for i in range(len(train_loss)):
        train_loss[i] = float(train_loss[i])
        writer.add_scalar('train/loss', train_loss[i], cur_idx[i])  # 画loss，横坐标为epoch
    for i in range(len(accuracy)):
        accuracy[i] = float(accuracy[i])
        writer.add_scalar('train/accuracy', accuracy[i], cur_idx[i])
    return epoch, per_idx, cur_idx, train_loss, accuracy

def make_text_save(text_dir, epoch, pre_idx, tain_idx, train_loss, accuracy):
    write_file = open(text_dir, 'w')
    write_file.write(str(epoch) + '\n')
    write_file.write(str(pre_idx) + '\n')
    write_file.write(str(tain_idx) + '\n')
    write_file.write(str(train_loss) + '\n')
    write_file.write(str(accuracy) + '\n')
    write_file.close()
batch_size = 60
num_class = 480
feature_size = 128
lr = 0.001
epochs = 1000

prepare_transform_for_image()
train_dataset = MyDataset('/home/ubuntu/dataset/CASIA/train/',
                          '/home/ubuntu/dataset/CASIA/train_label.txt', preprocessing)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
net = ResNet.resnet34()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# 预览输入图像
writer = SummaryWriter(log_dir='/home/ubuntu/tensorboard_data/')
data_iter = iter(train_dataloader)
images, labels = next(data_iter)
img_grid = torchvision.utils.make_grid(images)
writer.add_image('preview_some_pictures', img_grid)
# images = images.cuda()
# writer.add_graph(net, images)
# 损失函数
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.0005)

torch.autograd.set_grad_enabled(True)
train_loss = []
train_idx = []
per_idx = 0
train_accuracy = []
epoch = 0
record_size = 15
flag = False
# 加载参数
PATH_NET = '/home/ubuntu/graduation_model/deeplearning/model_net_134.pt'
# PATH_ARC = '/home/ubuntu/graduation_model/deeplearning/model_arcloss_139.pt'
flag = False
print("===start load param===")
if flag == True:
    net.load_state_dict(torch.load(PATH_NET))
net.train()
# param = torch.load(PATH_ARC)
print("===successfully load net===")


print("===start_training===")
while epoch < epochs:
    running_loss = 0.0
    accuracy = 0.0
    if flag == True:
        epoch, per_idx, train_idx, train_loss, train_accuracy = read_txt('/home/ubuntu/graduation_model/deeplearning/CASIA_data.txt')
        # epoch, per_idx, train_idx, train_loss,train_accuracy = read_txt('/home/ubuntu/project/data_v1.0.txt')
        flag = False
    for i, data in enumerate(train_dataloader):
        images, label = data
        images = images.to(device)
        label = label.to(device)
        logits,tmp_accuracy = net.forward(images,label)
        # out = torch.log(logits)
        loss = criterion(logits,label)
        # 计算
        accuracy += tmp_accuracy
        # print(loss.item())
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % record_size == 0:
            print('[epoch:%d  %5d]   loss: %.5f accuracy: %.5f' % (
                epoch + 1, i + 1, running_loss / record_size, accuracy / record_size))
            # print(accuracy/record_size)
            train_loss.append(running_loss / record_size)
            train_accuracy.append(accuracy.item() / record_size)
            train_idx.append(per_idx)
            writer.add_scalar('train/loss', running_loss / record_size, per_idx)  # 画loss，横坐标为epoch
            writer.add_scalar('train/accuracy', accuracy.item() / record_size, per_idx)
            accuracy = 0
            per_idx += 1
            running_loss = 0
    if (epoch + 1) % 15 == 0:
        save_path = '/home/ubuntu/graduation_model/deeplearning/CASIA_data.txt'
        make_text_save(save_path, epoch, per_idx, train_idx, train_loss, train_accuracy)
        PATH = "/home/ubuntu/graduation_model/deeplearning/CASIA/model_net_" + str(epoch) + ".pt"
        # Save
        torch.save(net.state_dict(), PATH)
    epoch+=1

