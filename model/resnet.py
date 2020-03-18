import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model import common
from IPython import embed


def make_model(args, parent=False):
    return ResNet(args[0])


# reference: torchvision
class ResBlock(nn.Module):
    def __init__(
        self, in_channels, planes, kernel_size, stride=1,
        conv3x3=common.default_conv,
        conv1x1=common.default_conv,
        norm=common.default_norm,
        act=common.default_act,
        downsample=None):

        super(ResBlock, self).__init__()
        m = [conv3x3(
            in_channels, planes, kernel_size, stride=stride, bias=False
        )]
        self.stride = stride
        if norm: m.append(norm(planes))
        m.append(act())
        m.append(conv3x3(planes, planes, kernel_size, bias=False))
        if norm: m.append(norm(planes))

        self.body = nn.Sequential(*m)
        self.downsample = downsample
        self.act_out = act()

    def forward(self, x):
        out = self.body(x)
        # print(out.shape, x.shape)
        if self.downsample is not None: x = self.downsample(x)
        # print(self.downsample, self.stride)
        # print('Out size {}; x size {}'.format(out.size(), x.size()))
        # if x.shape[1] == 16:
        #     embed()
        out += x
        out = self.act_out(out)

        return out


class BottleNeck(nn.Module):
    def __init__(
        self, in_channels, planes, kernel_size, stride=1,
        conv3x3=common.default_conv,
        conv1x1=common.default_conv,
        norm=common.default_norm,
        act=common.default_act,
        downsample=None):

        super(BottleNeck, self).__init__()
        m = [conv1x1(in_channels, planes, 1, bias=False)]
        if norm: m.append(norm(planes))
        m.append(act())
        m.append(conv3x3(planes, planes, kernel_size, stride=stride, bias=False))
        if norm: m.append(norm(planes))
        m.append(act())
        m.append(conv1x1(planes, 4 * planes, 1, bias=False))
        if norm: m.append(norm(4 * planes))

        self.body = nn.Sequential(*m)
        self.downsample = downsample
        self.act_out = act()

    def forward(self, x):
        out = self.body(x)
        if self.downsample is not None: x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out


class DownSampleA(nn.Module):
    def __init__(self):
        super(DownSampleA, self).__init__()

    def forward(self, x):
        # identity shortcut with 'padding zero'
        c = x.size(1) // 2
        pool = F.avg_pool2d(x, 2)
        out = F.pad(pool, (0, 0, 0, 0, c, c), 'constant', 0)

        return out


class DownSampleC(nn.Sequential):
    def __init__(
        self, in_channels, out_channels,
        stride=1, conv1x1=common.default_conv):

        m = [
            conv1x1(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        super(DownSampleC, self).__init__(*m)


class ResNet(nn.Module):
    def __init__(
        self, args, conv3x3=common.default_conv, conv1x1=common.default_conv):

        super(ResNet, self).__init__()

        m = []
        if args.data_train.find('CIFAR') >= 0:
            self.block = ResBlock
            self.expansion = 1

            self.n_blocks = (args.depth - 2) // 6
            self.in_channels = 16
            self.downsample_type = 'A'
            n_classes = int(args.data_train[5:])

            kwargs = {
                'kernel_size': args.kernel_size,
                'conv3x3': conv3x3,
            }
            m.append(common.BasicBlock(args.n_colors, 16, **kwargs))
            m.append(self.make_layer(16, self.n_blocks, **kwargs))
            m.append(self.make_layer(32, self.n_blocks, stride=2, **kwargs))
            m.append(self.make_layer(64, self.n_blocks, stride=2, **kwargs))
            m.append(nn.AvgPool2d(8))

            fc = nn.Linear(64 * self.expansion, n_classes)

        elif args.data_train == 'ImageNet':
            block_config = {
                18: ([2, 2, 2, 2], ResBlock, 1),
                34: ([3, 4, 6, 3], ResBlock, 1),
                50: ([3, 4, 6, 3], BottleNeck, 4),
                101: ([3, 4, 23, 3], BottleNeck, 4),
                152: ([3, 8, 36, 3], BottleNeck, 4)
            }
            n_blocks, self.block, self.expansion = block_config[args.depth]

            self.in_channels = 64
            self.downsample_type = 'C'
            n_classes = 1000
            kwargs = {
                'conv3x3': conv3x3,
                'conv1x1': conv1x1,
            }
            m.append(common.BasicBlock(
                args.n_colors, 64, 7, stride=2, conv3x3=conv3x3, bias=False
            ))
            m.append(nn.MaxPool2d(3, 2, padding=1))
            m.append(self.make_layer(64, n_blocks[0], 3, **kwargs))
            m.append(self.make_layer(128, n_blocks[1], 3, stride=2, **kwargs))
            m.append(self.make_layer(256, n_blocks[2], 3, stride=2, **kwargs))
            m.append(self.make_layer(512, n_blocks[3], 3, stride=2, **kwargs))
            m.append(nn.AvgPool2d(7, 1))

            fc = nn.Linear(512 * self.expansion, n_classes)

        self.features= nn.Sequential(*m)
        self.classifier = fc

        # only if when it is child model
        # if conv3x3 == common.default_conv:
            # traning phase of network pruning
            # if args.model.lower().find('prune') >= 0 and not args.test_only and not args.load:
            #     self.load(args, strict=True)

    def make_layer(
        self, planes, blocks, kernel_size, stride=1,
        conv3x3=common.default_conv,
        conv1x1=common.default_conv,
        bias=False):

        out_channels = planes * self.expansion
        if stride != 1 or self.in_channels != out_channels:
            if self.downsample_type == 'A':
                downsample = DownSampleA()
            elif self.downsample_type == 'C':
                downsample = DownSampleC(
                    self.in_channels,
                    out_channels,
                    stride=stride,
                    conv1x1=conv1x1
                )
        else:
            downsample = None

        kwargs = {
            'conv3x3': conv3x3,
            'conv1x1': conv1x1
        }
        m = [self.block(
            self.in_channels, planes, kernel_size,
            stride=stride, downsample=downsample, **kwargs
        )]
        self.in_channels = out_channels

        for _ in range(blocks - 1):
            m.append(self.block(
                self.in_channels, planes, kernel_size, **kwargs
            ))

        return nn.Sequential(*m)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.squeeze())

        return x

    def load(self, args, strict=True):
        if args.data_train == 'ImageNet':
            if args.pretrain == 'download' or args.extend == 'download':
                state = getattr(models, 'resnet{}'.format(args.depth))(
                    pretrain=True
                )
            elif args.extend:
                state = torch.load(args.extend)
            else:
                common.init_kaiming(self)
                return

            source = state.state_dict()
            target = self.state_dict()
            for s, t in zip(source.keys(), target.keys()):
                target[t].copy_(source[s])
        else:
            if args.pretrain:
                self.load_state_dict(torch.load(args.pretrain), strict=strict)