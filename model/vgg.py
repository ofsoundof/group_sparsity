import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model import common


def make_model(args, parent=False):
    return VGG(args[0])

# reference: torchvision
class VGG(nn.Module):
    def __init__(self, args, conv3x3=common.default_conv, conv1x1=None):
        super(VGG, self).__init__()
        # we use batch noramlization for VGG
        norm = None
        self.norm = norm
        bias = not args.no_bias

        configs = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            'ef': [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']
        }

        body_list = []
        in_channels = args.n_colors
        if args.data_train.find('CIFAR') >= 0 or args.data_train.find('Tiny') >= 0:

            for i, v in enumerate(configs[args.vgg_type]):
                if v == 'M':
                    body_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    stride = 2 if i == 0 and args.data_train.find('Tiny') >= 0 else 1
                    body_list.append(common.BasicBlock(in_channels, v, args.kernel_size, stride=stride,
                                                       bias=bias, conv3x3=conv3x3))
                    in_channels = v
        else:
            for i, v in enumerate(configs[args.vgg_type]):
                if v == 'M':
                    body_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    conv2d = conv3x3(in_channels, v, kernel_size=3)
                    if norm is not None:
                        body_list += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        body_list += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v


        # assert(args.data_train.find('CIFAR') >= 0)
        self.features = nn.Sequential(*body_list)
        if args.data_train.find('CIFAR') >= 0:
            n_classes = int(args.data_train[5:])
            if args.template.find('linear3') >= 0:
                self.classifier = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Linear(in_channels, in_channels),
                                                nn.Linear(in_channels, n_classes))
            else:
                self.classifier = nn.Linear(in_channels, n_classes)
        if args.data_train.find('Tiny') >= 0:
            n_classes = 200
            self.classifier = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Linear(in_channels, in_channels),
                                            nn.Linear(in_channels, n_classes))
        elif args.data_train == 'ImageNet':
            n_classes = 1000
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, n_classes),
            )
            # self.classifier = nn.Sequential(
            #     nn.Linear(512 * 7 * 7, n_classes)
            # )
        # print(conv3x3, conv3x3 == common.default_conv or conv3x3 == nn.Conv2d)

        # if conv3x3 == common.default_conv or conv3x3 == nn.Conv2d:
        #     self.load(args, strict=True)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        return x

    def load(self, args, strict=True):
        model_dir = os.path.join('..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        if args.data_train.find('CIFAR') >= 0:
            if args.pretrain == 'download' or args.extend == 'download':
                url = (
                    'https://cv.snu.ac.kr/'
                    'research/clustering_kernels/models/vgg16-89711a85.pt'
                )

                state = model_zoo.load_url(url, model_dir=model_dir)
            elif args.pretrain:
                state = torch.load(args.pretrain)
            elif args.extend:
                state = torch.load(args.extend)
            else:
                common.init_vgg(self)
                return
        elif args.data_train == 'ImageNet':
            if args.pretrain == 'download':
                #print('pretrain download')
                if self.norm is not None:
                    url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
                else:
                    url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
                state = model_zoo.load_url(url, model_dir=model_dir)
            else:
                common.init_vgg(self)
                return
        elif args.data_train == 'Tiny_ImageNet':
            common.init_vgg(self)
            return
        else:
            raise NotImplementedError('Unavailable dataset {}'.format(args.data_train))
        #print(state['features.0.bias'])
        self.load_state_dict(state, strict=strict)
