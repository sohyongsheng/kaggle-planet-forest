# Code from Heng CherKeng
# https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/33559
#  download from:
#       https://raw.githubusercontent.com/pytorch/vision/master/torchvision/models/resnet.py
#
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18' : 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34' : 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50' : 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_shape=(3,244,244), num_classes=1000):
        self.inplanes = 64

        super(ResNet, self).__init__()
        in_channels, height, width = in_shape

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1  = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = F.dropout(x8,p=0.5,training=self.training)
        #x = nn.AvgPool2d(kernel_size=7)
        #x = F.adaptive_avg_pool2d(x,output_size=1) + F.adaptive_max_pool2d(x,output_size=1)
        #x = F.adaptive_max_pool2d(x8,output_size=1)
        x = F.adaptive_avg_pool2d(x, output_size = (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.sigmoid(x)

        return x


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

if __name__ == '__main__':
    torch.cuda.set_device(2)
    # Make sure that modified ResNet works properly for other image sizes.
    batch_size = 1
    number_of_channels = 3
    image_height = 256
    image_width = 256
    number_of_classes = 17
    inputs = torch.randn(batch_size, number_of_channels, image_height, image_width)
    labels = torch.randn(batch_size, number_of_classes)
    (inputs, labels) = (Variable(inputs.cuda()), Variable(labels.cuda()))
    
    model = resnet34(in_shape = (number_of_channels, image_height, image_width), num_classes = number_of_classes)
    model = model.cuda()
    
    model.train()
    outputs = model(inputs)
    loss = F.binary_cross_entropy(outputs, labels)
    loss.backward()
    
#    print(type(model))
#    print(model)
#    print('Input size:', inputs.size())
#    print('Output size:', outputs.size())

#    parameters = list(model.parameters())
#    print(parameters[0])
#    for params in model.parameters():
#        print(type(params))
