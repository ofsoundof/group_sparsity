# This is the wide resnet module.
__author__ = 'yawli'
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import common
#from IPython import embed


def make_model(args, parent=False):
    return Wide_ResNet(args[0])


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv):
        super(Wide_ResNet, self).__init__()
        self.args = args
        depth = self.args.depth
        widen_factor = self.args.widen_factor
        dropout_rate = self.args.dropout_rate
        num_classes = int(args.data_train[5:]) if args.data_train.find('CIFAR') >= 0 else 200
        # if self.args.data_train == 'CIFAR10':
        #     num_classes = 10
        # elif self.args.data_train == 'CIFAR100':
        #     num_classes = 100
        # else:
        #     raise NotImplementedError('Wide ResNet is not applied to dataset {}'.format(self.args.data_train))
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4) // 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        stride = 1 if args.data_train.find('CIFAR') >= 0 else 2
        self.conv1 = conv3x3(3, nStages[0], stride=stride)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def load(self, pretrain, strict=True):
        if pretrain:
            self.load_state_dict(torch.load(pretrain), strict=strict)
