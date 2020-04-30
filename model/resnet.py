import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model import common
#from IPython import embed


def make_model(args, parent=False):
    return ResNet(args[0])


class ResBlock(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, stride=1, conv3x3=common.default_conv, downsample=None, args=None):
        super(ResBlock, self).__init__()

        self.stride = stride
        m = [conv3x3(in_channels, planes, kernel_size, stride=stride, bias=False, args=args),
             nn.BatchNorm2d(planes),
             nn.ReLU(inplace=True),
             conv3x3(planes, planes, kernel_size, bias=False, args=args),
             nn.BatchNorm2d(planes)]

        self.body = nn.Sequential(*m)
        self.downsample = downsample
        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.body(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, stride=1, conv3x3=common.default_conv,
                 conv1x1=common.default_conv, downsample=None, args=None):
        super(BottleNeck, self).__init__()

        expansion = 4
        m = [conv1x1(in_channels, planes, 1, bias=False),
             nn.BatchNorm2d(planes),
             nn.ReLU(inplace=True),
             conv3x3(planes, planes, kernel_size, stride=stride, bias=False, args=args),
             nn.BatchNorm2d(planes),
             nn.ReLU(inplace=True),
             conv1x1(planes, expansion * planes, 1, bias=False),
             nn.BatchNorm2d(expansion * planes)]

        self.body = nn.Sequential(*m)
        self.downsample = downsample
        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.body(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out


class DownSampleA(nn.Module):
    def __init__(self):
        super(DownSampleA, self).__init__()

    def forward(self, x):
        # identity shortcut with 'zero padding' in the channel dimension
        c = x.size(1) // 2
        pool = F.avg_pool2d(x, 2)
        out = F.pad(pool, [0, 0, 0, 0, c, c], 'constant', 0)

        return out


class DownSampleC(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, conv1x1=common.default_conv):
        m = [conv1x1(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels)]
        super(DownSampleC, self).__init__(*m)


class ResNet(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv, conv1x1=common.default_conv):
        super(ResNet, self).__init__()

        n_classes = int(args.data_train[5:]) if args.data_train.find('CIFAR') >= 0 else 200
        kernel_size = args.kernel_size
        if args.depth == 50:
            self.expansion = 4
            self.block = BottleNeck
            self.n_blocks = (args.depth - 2) // 9
        elif args.depth <= 56:
            self.expansion = 1
            self.block = ResBlock
            self.n_blocks = (args.depth - 2) // 6
        else:
            self.expansion = 4
            self.block = BottleNeck
            self.n_blocks = (args.depth - 2) // 9
        self.in_channels = 16
        self.downsample_type = args.downsample_type
        bias = not args.no_bias

        kwargs = {'conv3x3': conv3x3,
                  'conv1x1': conv1x1,
                  'args': args}
        stride = 1 if args.data_train.find('CIFAR') >= 0 else 2
        m = [common.BasicBlock(args.n_colors, 16, kernel_size=kernel_size, stride=stride, bias=bias, conv3x3=conv3x3, args=args),
             self.make_layer(self.n_blocks, 16, kernel_size, **kwargs),
             self.make_layer(self.n_blocks, 32, kernel_size, stride=2, **kwargs),
             self.make_layer(self.n_blocks, 64, kernel_size, stride=2, **kwargs),
             nn.AvgPool2d(8)]
        fc = nn.Linear(64 * self.expansion, n_classes)

        self.features = nn.Sequential(*m)
        self.classifier = fc

    def make_layer(self, blocks, planes, kernel_size, stride=1, conv3x3=common.default_conv, conv1x1=common.default_conv,
                   args=None):
        out_channels = planes * self.expansion
        if stride != 1 or self.in_channels != out_channels:
            if self.downsample_type == 'A':
                downsample = DownSampleA()
            elif self.downsample_type == 'C':
                downsample = DownSampleC(self.in_channels, out_channels, stride=stride, conv1x1=conv1x1)
        else:
            downsample = None
        kwargs = {'conv3x3': conv3x3,
                  'args': args}
        if self.block == BottleNeck:
            kwargs['conv1x1'] = conv1x1

        m = [self.block(self.in_channels, planes, kernel_size, stride=stride, downsample=downsample, **kwargs)]
        self.in_channels = out_channels
        for _ in range(blocks - 1):
            m.append(self.block(self.in_channels, planes, kernel_size, **kwargs))

        return nn.Sequential(*m)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.squeeze())
        return x

    def load(self, args, strict=True):
        if args.pretrain:
            self.load_state_dict(torch.load(args.pretrain), strict=strict)

    # def load_state_dict(self, state_dict, strict=True):
    #     """
    #     load state dictionary
    #     """
    #     if strict:
    #         # used to load the model parameters during training
    #         super(ResNet, self).load_state_dict(state_dict, strict)
    #     else:
    #         # used to load the model parameters during test
    #         own_state = self.state_dict(keep_vars=True)
    #         for (name, param), (name_o, param_o) in zip(state_dict.items(), own_state.items()):
    #             # if name in own_state:
    #             if isinstance(param, nn.Parameter):
    #                 param = param.data
    #             # if param.size() != own_state[name_o].size():
    #             #     own_state[name_o].data = param
    #             # else:
    #             own_state[name_o].data.copy_(param)
