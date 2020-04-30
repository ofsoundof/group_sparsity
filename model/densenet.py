import math

import torch
import torch.nn as nn
from model import common

def make_model(args, parent=False):
    return DenseNet(args[0])

# reference: torchvision
class Dense(nn.Module):
    def __init__(
            self, in_channels, k, kernel_size=3, bias=False,
            conv=common.default_conv,
            norm=common.default_norm,
            act=common.default_act):
        super(Dense, self).__init__()

        module_list = []
        # This is a pre-activation module
        if norm: module_list.append(norm(in_channels))
        module_list.append(act())
        module_list.append(conv(in_channels, k, kernel_size, bias=bias))

        self.body = nn.Sequential(*module_list)

    def forward(self, x):
        out = self.body(x)
        out = torch.cat((x, out), 1)

        return out

class BottleNeck(nn.Module):
    def __init__(
            self, in_channels, k, kernel_size=3, bias=False,
            conv=common.default_conv,
            norm=common.default_norm,
            act=common.default_act):

        super(BottleNeck, self).__init__()

        module_list = []
        if norm: module_list.append(norm(in_channels))
        module_list.append(act())
        module_list.append(nn.Conv2d(in_channels, 4 * k, 1, bias=bias))
        if norm: module_list.append(norm(4 * k))
        module_list.append(act())
        module_list.append(conv(4 * k, k, kernel_size, bias=bias))

        self.body = nn.Sequential(*module_list)

    def forward(self, x):
        out = self.body(x)
        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, bias=False,
            norm=common.default_norm,
            act=common.default_act,
            conv1x1=common.default_conv):

        module_list = []
        if norm: module_list.append(norm(in_channels))
        module_list.append(act())
        module_list.append(conv1x1(in_channels, out_channels, 1, bias=bias))
        module_list.append(nn.AvgPool2d(2))

        super(Transition, self).__init__(*module_list)

class DenseNet(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv, conv1x1=common.default_conv):
        super(DenseNet, self).__init__()

        n_blocks = (args.depth - 4) // 3
        if args.bottleneck: n_blocks //= 2

        k = args.k
        c_in = 2 * k

        def _dense_block(in_channels):
            module_list = []
            for _ in range(n_blocks):
                if args.bottleneck:
                    module_list.append(BottleNeck(in_channels, k, conv=conv3x3))
                else:
                    module_list.append(Dense(in_channels, k, conv=conv3x3))
                in_channels += k

            return nn.Sequential(*module_list)

        module_list = []
        module_list.append(conv3x3(args.n_colors, c_in, 3, bias=False))

        for i in range(3):
            module_list.append(_dense_block(c_in))
            c_in += k * n_blocks
            if i < 2:
                c_out = int(math.floor(args.reduction * c_in))
                module_list.append(Transition(c_in, c_out, conv1x1=conv1x1))
                c_in = c_out

        module_list.append(common.default_norm(c_in))
        module_list.append(common.default_act())
        module_list.append(nn.AvgPool2d(8))
        self.features = nn.Sequential(*module_list)

        if args.data_train == 'ImageNet':
            n_classes = 1000
        else:
            if args.data_train.find('CIFAR') >= 0:
                n_classes = int(args.data_train[5:])

        self.classifier = nn.Linear(c_in, n_classes)

        common.init_kaiming(self)
        # if conv3x3 == common.default_conv:
        #     self.load(args, strict=True)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.squeeze())

        return x

    def load(self, pretrain, strict=True):
        # used by compression methods (Hinge).
        if pretrain:
            self.load_state_dict(torch.load(pretrain), strict=strict)


def gen(target):
    '''
         a generator for iterating default Conv2d
    '''
    def _criterion(m):
        if isinstance(m, nn.Conv2d):
            return m.in_channels >= 24 and m.kernel_size[0] != 1

        return False

    gen = (m for m in target.modules() if _criterion(m))

    return gen

