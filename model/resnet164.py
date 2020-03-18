"""
Author: Yawei Li
Date: 06/11/2019
ResNet164 for CIFAR10 and CIFAR100 classification
"""

from torchvision.models.resnet import conv3x3, conv1x1, Bottleneck
import math
import torch.nn as nn
from IPython import embed
import torch.utils.model_zoo as model_zoo
from model import common


def make_model(args, parent=False):
    return ResNet164(args[0])


class ResNet164(nn.Module):

    def __init__(self, args):
        super(ResNet164, self).__init__()
        block = Bottleneck

        self.args = args
        self.depth = args.depth
        n = (self.depth - 2) // 9
        width = 16
        self.inplanes = width

        if args.data_train == 'CIFAR10':
            num_classes = 10
        elif args.data_train == 'CIFAR100':
            num_classes = 100
        else:
            raise NotImplementedError('The module is not designed for dataset ' + args.data_train)

        self.conv1 = conv3x3(3, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, width, n)
        self.layer2 = self._make_layer(block, width * 2, n, stride=2)
        self.layer3 = self._make_layer(block, width * 4, n, stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # if self.args.pretrained:
        #     self.load(args, strict=True)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=nn.BatchNorm2d))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=nn.BatchNorm2d))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
