import torch.nn as nn
import torch.nn.functional as F



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        # conv1
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # conv3.x
        self.layer2 = self._make_layer(block, 128, layers[1])
        # conv4.x
        self.layer3 = self._make_layer(block, 256, layers[2])
        # conv5.x
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512 * block.expansion, 128, bias=False)

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion), )
        layers = [block(self.in_channel, channel, stride, downsample)]
        self.in_channel = channel
        for i in range(1, blocks):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, x, label=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        feature = self.fc(x)

        # feature = self.fc(x)
        if label is None:
            return feature, F.normalize(feature)
        else:

            _, mlogits, _, _ = self.lmcl(feature, label)
            return feature,mlogits


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
