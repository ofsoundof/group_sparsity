import math
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from IPython import embed
count_data = 0
import numpy as np
from torch.nn import init

# round2 over-parameterize
class ROPConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1, expansion=4, args=None, latent_trainable=True):
        super(ROPConv1, self).__init__()
        self.expansion = expansion
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        s = kernel_size + 3 * 2
        if latent_trainable:
            latent_vector = init.kaiming_normal_(torch.Tensor(1, 1, s, s))
            self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        else:
            latent_vector = init.normal_(torch.Tensor(1, 1, s, s))
            self.register_buffer('latent_vector', latent_vector)

        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        layer1 = [nn.Conv2d(1, out_channels, kernel_size), nn.BatchNorm2d(out_channels), nn.ReLU()]
        layer2 = [nn.Conv2d(1, in_channels, kernel_size), nn.BatchNorm2d(in_channels), nn.ReLU()]
        # embed()
        layer3 = [nn.Conv2d(in_channels, in_channels * expansion, kernel_size), nn.BatchNorm2d(in_channels * expansion)]
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)

    def forward(self, x):
        weight = self.layer1(self.latent_vector).transpose(0, 1)
        weight = self.layer3(self.layer2(weight))
        weight = weight.reshape(self.out_channels, self.in_channels, self.expansion, self.kernel_size, self.kernel_size).sum(dim=2)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


class ROPConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1, expansion=4, args=None, latent_trainable=True):
        super(ROPConv2, self).__init__()
        self.expansion = expansion
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        s = kernel_size + 2 * 2
        self.s = s
        if latent_trainable:
            latent_vector = init.kaiming_normal_(torch.Tensor(1, 1, s, s))
            self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        else:
            latent_vector = init.normal_(torch.Tensor(1, 1, s, s))
            # self.register_parameter('latent_vector', nn.Parameter(latent_vector, requires_grad=False))
            self.register_buffer('latent_vector', latent_vector)

        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        layer1 = [nn.Conv2d(1, in_channels * out_channels, kernel_size), nn.BatchNorm2d(in_channels * out_channels), nn.ReLU()]
        layer2 = [nn.Conv2d(in_channels, in_channels * expansion, kernel_size), nn.BatchNorm2d(in_channels * expansion)]
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)

    def forward(self, x):
        weight = self.layer1(self.latent_vector).reshape(self.out_channels, self.in_channels, self.s - 2, self.s - 2)
        weight = self.layer2(weight)
        weight = weight.reshape(self.out_channels, self.in_channels, self.expansion, self.kernel_size, self.kernel_size).sum(dim=2)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


class ROPConv3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1, expansion=4, latent_trainable=True):
        super(ROPConv3, self).__init__()
        self.expansion = expansion
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        if latent_trainable:
            latent_vector = init.kaiming_normal_(torch.Tensor(1, in_channels))
            self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        else:
            latent_vector = init.normal_(torch.Tensor(1, in_channels))
            # self.register_parameter('latent_vector', nn.Parameter(latent_vector, requires_grad=False))
            self.register_buffer('latent_vector', latent_vector)

        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        hypernet = [nn.Linear(in_channels, in_channels * kernel_size), nn.ReLU(),
                    nn.Linear(in_channels * kernel_size, in_channels * kernel_size ** 2), nn.ReLU(),
                    nn.Linear(in_channels * kernel_size ** 2, self.expansion * in_channels * kernel_size ** 2 *out_channels),
                    ]
        self.batchnorm = nn.BatchNorm2d(in_channels * expansion)
        self.hypernet = nn.Sequential(*hypernet)

    def forward(self, x):
        weight = self.hypernet(self.latent_vector)
        weight = self.batchnorm(weight.reshape(self.out_channels, self.in_channels * self.expansion, self.kernel_size, self.kernel_size))
        weight = weight.reshape(self.out_channels, self.in_channels, self.expansion, self.kernel_size,
                                self.kernel_size).sum(dim=2)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


class ROPConv4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1, expansion=4, latent_trainable=True):
        super(ROPConv4, self).__init__()
        self.expansion = expansion
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        if latent_trainable:
            latent_vector = init.kaiming_normal_(torch.Tensor(1, in_channels))
            self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        else:
            latent_vector = init.normal_(torch.Tensor(1, in_channels))
            # self.register_parameter('latent_vector', nn.Parameter(latent_vector, requires_grad=False))
            self.register_buffer('latent_vector', latent_vector)

        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        layer1 = [nn.Linear(in_channels, in_channels * out_channels), nn.ReLU()]
        layer2 = [nn.Linear(in_channels, in_channels * kernel_size ** 2 * expansion)]
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.batchnorm = nn.BatchNorm2d(in_channels * expansion)

    def forward(self, x):
        weight = self.layer1(self.latent_vector).reshape(self.out_channels, self.in_channels)
        weight = self.layer2(weight)
        weight = self.batchnorm(weight.reshape(self.out_channels, self.in_channels * self.expansion, self.kernel_size, self.kernel_size))
        weight = weight.reshape(self.out_channels, self.in_channels, self.expansion, self.kernel_size,
                                self.kernel_size).sum(dim=2)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


# round1 over-parameterize
class OPConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1):
        super(OPConv1, self).__init__()
        self.expansion = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        latent_vector = init.kaiming_normal_(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        hypernet = [nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2), nn.BatchNorm2d(in_channels), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2), nn.BatchNorm2d(in_channels), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels * self.expansion, kernel_size, padding=kernel_size//2),
                              nn.BatchNorm2d(in_channels * self.expansion)]
        self.hypernet = nn.Sequential(*hypernet)

    def forward(self, x):
        weight = self.hypernet(self.latent_vector)
        weight = weight.reshape(self.out_channels, self.in_channels, self.expansion, self.kernel_size, self.kernel_size).sum(dim=2)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


class OPConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1):
        super(OPConv2, self).__init__()
        self.expansion = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        s = kernel_size + 4 * 2
        latent_vector = init.kaiming_normal_(torch.Tensor(1, 1, s, s))
        self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        hypernet = [nn.Conv2d(1, in_channels, kernel_size), nn.BatchNorm2d(in_channels), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size), nn.BatchNorm2d(in_channels), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size), nn.BatchNorm2d(in_channels), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels * out_channels * self.expansion, kernel_size),
                              nn.BatchNorm2d(in_channels * out_channels * self.expansion)]
        self.hypernet = nn.Sequential(*hypernet)

    def forward(self, x):
        weight = self.hypernet(self.latent_vector)
        weight = weight.reshape(self.out_channels, self.in_channels, self.expansion, self.kernel_size, self.kernel_size).sum(dim=2)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


class OPConv3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1):
        super(OPConv3, self).__init__()
        self.expansion = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        s = kernel_size + 4 * 2
        latent_vector = init.kaiming_normal_(torch.Tensor(out_channels, 1, s, s))
        self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        hypernet = [nn.Conv2d(1, in_channels, kernel_size), nn.BatchNorm2d(in_channels), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size), nn.BatchNorm2d(in_channels), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size), nn.BatchNorm2d(in_channels), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels * self.expansion, kernel_size),
                              nn.BatchNorm2d(in_channels * self.expansion)]
        self.hypernet = nn.Sequential(*hypernet)

    def forward(self, x):
        weight = self.hypernet(self.latent_vector)
        weight = weight.reshape(self.out_channels, self.in_channels, self.expansion, self.kernel_size, self.kernel_size).sum(dim=2)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


class OPConv4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1):
        super(OPConv4, self).__init__()
        self.expansion = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        s = kernel_size + 4 * 2
        latent_vector = init.kaiming_normal_(torch.Tensor(out_channels, 1, s, s))
        self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        hypernet = [nn.Conv2d(1, in_channels, kernel_size), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels * self.expansion, kernel_size),
                              nn.BatchNorm2d(in_channels * self.expansion)]
        self.hypernet = nn.Sequential(*hypernet)

    def forward(self, x):
        weight = self.hypernet(self.latent_vector)
        weight = weight.reshape(self.out_channels, self.in_channels, self.expansion, self.kernel_size, self.kernel_size).sum(dim=2)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


class OPConv5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1):
        super(OPConv5, self).__init__()
        self.expansion = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        s = kernel_size + 4 * 2
        latent_vector = init.kaiming_normal_(torch.Tensor(out_channels, 1, s, s))
        self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        hypernet = [nn.Conv2d(1, in_channels, kernel_size), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels * self.expansion, kernel_size),
                    ]
        self.hypernet = nn.Sequential(*hypernet)

    def forward(self, x):
        weight = self.hypernet(self.latent_vector)
        weight = weight.reshape(self.out_channels, self.in_channels, self.expansion, self.kernel_size, self.kernel_size).sum(dim=2)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


class OPConv6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1):
        super(OPConv6, self).__init__()
        self.expansion = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        latent_vector = init.kaiming_normal_(torch.Tensor(1, 1, out_channels * kernel_size, in_channels * kernel_size))
        self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        hypernet = [nn.Conv2d(1, in_channels, kernel_size, padding=kernel_size//2), nn.BatchNorm2d(in_channels), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2), nn.BatchNorm2d(in_channels), nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2), nn.BatchNorm2d(in_channels), nn.ReLU(),
                    nn.Conv2d(in_channels, 1, kernel_size, padding=kernel_size//2)]
        self.hypernet = nn.Sequential(*hypernet)

    def forward(self, x):
        weight = self.hypernet(self.latent_vector)
        weight = weight.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


class OPConv7(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1):
        super(OPConv7, self).__init__()
        self.expansion = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        latent_vector = init.kaiming_normal_(torch.Tensor(1, in_channels))
        self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        hypernet = [nn.Linear(in_channels, in_channels * kernel_size), nn.ReLU(),
                    nn.Linear(in_channels * kernel_size, in_channels * kernel_size ** 2), nn.ReLU(),
                    nn.Linear(in_channels * kernel_size ** 2, in_channels * kernel_size ** 2 *out_channels),
                    ]
                    # nn.Linear(in_channels * kernel_size ** 2 *out_channels, in_channels * kernel_size ** 2 *out_channels),
                    #     nn.BatchNorm2d(in_channels * kernel_size ** 2 *out_channels)]
        self.hypernet = nn.Sequential(*hypernet)

    def forward(self, x):
        # embed()
        weight = self.hypernet(self.latent_vector)
        weight = weight.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


class OPConv8(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1):
        super(OPConv8, self).__init__()
        self.expansion = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        latent_vector = init.kaiming_normal_(torch.Tensor(1, in_channels))
        self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        hypernet = [nn.Linear(in_channels, in_channels * kernel_size), nn.ReLU(),
                    nn.Linear(in_channels * kernel_size, in_channels * kernel_size ** 2), nn.ReLU(),
                    nn.Linear(in_channels * kernel_size ** 2, self.expansion * in_channels * kernel_size ** 2 *out_channels),
                    ]
                    # nn.Linear(in_channels * kernel_size ** 2 *out_channels, in_channels * kernel_size ** 2 *out_channels),
                    #     nn.BatchNorm2d(in_channels * kernel_size ** 2 *out_channels)]
        self.hypernet = nn.Sequential(*hypernet)

    def forward(self, x):
        # embed()
        weight = self.hypernet(self.latent_vector)
        weight = weight.reshape(self.out_channels, self.in_channels, self.expansion, self.kernel_size,
                                self.kernel_size).sum(dim=2)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


class OPConv9(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1):
        super(OPConv9, self).__init__()
        self.expansion = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        latent_vector = init.kaiming_normal_(torch.Tensor(out_channels, in_channels))
        self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        hypernet = [nn.Linear(in_channels, in_channels * kernel_size), nn.ReLU(),
                    nn.Linear(in_channels * kernel_size, in_channels * kernel_size ** 2), nn.ReLU(),
                    nn.Linear(in_channels * kernel_size ** 2, in_channels * kernel_size ** 2), nn.ReLU(),
                    nn.Linear(in_channels * kernel_size ** 2, in_channels * kernel_size ** 2)
                    ]
                    # nn.Linear(in_channels * kernel_size ** 2 *out_channels, in_channels * kernel_size ** 2 *out_channels),
                    #     nn.BatchNorm2d(in_channels * kernel_size ** 2 *out_channels)]
        self.hypernet = nn.Sequential(*hypernet)

    def forward(self, x):
        # embed()
        weight = self.hypernet(self.latent_vector)
        weight = weight.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


class OPConv10(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1):
        super(OPConv10, self).__init__()
        self.expansion = 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        latent_vector = init.kaiming_normal_(torch.Tensor(out_channels, in_channels))
        self.register_parameter('latent_vector', nn.Parameter(latent_vector))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
        hypernet = [nn.Linear(in_channels, in_channels * kernel_size), nn.ReLU(),
                    nn.Linear(in_channels * kernel_size, in_channels * kernel_size ** 2), nn.ReLU(),
                    nn.Linear(in_channels * kernel_size ** 2, self.expansion * in_channels * kernel_size ** 2),
                    ]
                    # nn.Linear(in_channels * kernel_size ** 2 *out_channels, in_channels * kernel_size ** 2 *out_channels),
                    #     nn.BatchNorm2d(in_channels * kernel_size ** 2 *out_channels)]
        self.hypernet = nn.Sequential(*hypernet)

    def forward(self, x):
        # embed()
        weight = self.hypernet(self.latent_vector)
        weight = weight.reshape(self.out_channels, self.in_channels, self.expansion, self.kernel_size,
                                self.kernel_size).sum(dim=2)
        x = F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x


## Channel Attention (CA) Layer
class ECALayer1(nn.Module):
    def __init__(self, channel, expansion):
        super(ECALayer1, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Linear(channel, channel * expansion, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(channel * expansion, channel * expansion, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        # embed()
        x = self.conv_du(x)
        return x


class ECALayer2(nn.Module):
    def __init__(self, channel, expansion):
        super(ECALayer2, self).__init__()
        # global average pooling: feature --> point
        pool_dim  = int(np.ceil(np.sqrt(expansion)))
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_dim)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Linear(channel * pool_dim ** 2, channel * expansion, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(channel * expansion, channel * expansion, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        # embed()
        x = self.conv_du(x)
        return x


class ECALayer3(nn.Module):
    def __init__(self, channel, expansion):
        super(ECALayer3, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Linear(channel, channel, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(channel, channel, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(channel, channel * expansion, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        # embed()
        x = self.conv_du(x)
        return x


class ECALayer4(nn.Module):
    def __init__(self, channel, expansion):
        super(ECALayer4, self).__init__()
        self.expansion = expansion
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Linear(channel, channel // 4, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(channel // 4, channel, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        # embed()
        x = self.conv_du(x)
        return torch.repeat_interleave(x, repeats=self.expansion, dim=1)


class ECALayer5(nn.Module):
    def __init__(self, channel, expansion):
        super(ECALayer5, self).__init__()
        self.expansion = expansion
        # global average pooling: feature --> point
        pool_dim = int(np.ceil(np.sqrt(expansion)))
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_dim)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Linear(channel * pool_dim ** 2, channel, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(channel, channel * expansion, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        # embed()
        x = self.conv_du(x)
        return x


class ECALayer6(nn.Module):
    def __init__(self, channel, expansion):
        super(ECALayer6, self).__init__()
        self.expansion = expansion
        # global average pooling: feature --> point
        pool_dim = int(np.floor(np.sqrt(expansion)))
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_dim)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Linear(channel * pool_dim ** 2, channel, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel * expansion, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        # embed()
        x = self.conv_du(x)
        return x


class NCALayer1(nn.Module):
    def __init__(self, channel, out_channels, expansion):
        super(NCALayer1, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Linear(channel, channel * expansion, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(channel * expansion, out_channels, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        # embed()
        x = self.conv_du(x)
        return x


class NCALayer2(nn.Module):
    def __init__(self, channel, out_channels, expansion):
        super(NCALayer2, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Linear(channel, channel // 4, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(channel // 4, out_channels, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        # embed()
        x = self.conv_du(x)
        return x


class ECAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1,
                 expansion=4, expansion_batchnorm=True, attention_type='E1'):
        super(ECAConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride=stride,
        self.bias = bias,
        self.groups= groups
        self.expansion_batchnorm = expansion_batchnorm

        self.inner_channels = int(in_channels * expansion)
        conv1 = []
        conv1.append(default_conv(in_channels, self.inner_channels, kernel_size, stride, bias, groups))
        if expansion_batchnorm:
            conv1.append(nn.BatchNorm2d(self.inner_channels))
        if attention_type == 'E1':
            ca = ECALayer1(in_channels, expansion=expansion)
        elif attention_type == 'E2':
            ca = ECALayer2(in_channels, expansion=expansion)
        elif attention_type == 'E3':
            ca = ECALayer3(in_channels, expansion=expansion)
        elif attention_type == 'E4':
            ca = ECALayer4(in_channels, expansion=expansion)
        elif attention_type == 'E5':
            ca = ECALayer5(in_channels, expansion=expansion)
        elif attention_type == 'E6':
            ca = ECALayer6(in_channels, expansion=expansion)
        else:
            raise NotImplementedError('Attention type {} is not implemented.'.format(attention_type))
        conv2 = default_conv(self.inner_channels, out_channels, 1, bias=bias)
        self.conv1 = nn.Sequential(*conv1)
        self.ca = ca
        self.conv2 = conv2

    def forward(self, x):
        # if self.training:
        y = self.conv1(x)
        ca = self.ca(x).unsqueeze(2).unsqueeze(3)
        y = self.conv2(y * ca)
        # else:
        #     weight1 = self.conv1._modules['0'].weight.data
        #     if self.expansion_batchnorm:
        #         bn_weight, bn_bias, bn_mean, bn_var, _ = self.conv1._modules['1'].state_dict().values()
        #         bn_std = bn_var.clone().data.add_(1e-10).pow_(-0.5)
        #         weight = weight1.mul(bn_std.view(-1, 1, 1, 1).expand_as(weight1)).mul(bn_weight.data.view(-1, 1, 1, 1).expand_as(weight1))
        #         bias = -bn_mean.data.mul(bn_std).mul(bn_weight.data) + bn_bias.data
        #     else:
        #         weight = weight1
        #         bias = torch.zeros(self.inner_channels).cuda()
        #     # embed()
        #     weight2 = self.ca(x).view(1, -1, 1, 1) * self.conv2.weight.data
        #     weight = torch.mm(weight.view(self.inner_channels, -1).t(), weight2.squeeze().t()).t()\
        #         .reshape(self.out_channels, self.in_channels, 3, 3)
        #     bias = torch.mm(bias.unsqueeze(0), weight2.squeeze().t()).squeeze()
        #     # embed()
        #     y = F.conv2d(x, weight, bias, stride=self.stride, padding=self.kernel_size // 2, groups=self.groups)
        return y


class NCAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1,
                 expansion=4, expansion_batchnorm=True, attention_type='N1'):
        super(NCAConv, self).__init__()
        conv1 = default_conv(in_channels, out_channels, kernel_size, stride, bias, groups)
        if attention_type == 'N1':
            ca = NCALayer1(in_channels, out_channels,  expansion=expansion)
        elif attention_type == 'N2':
            ca = NCALayer2(in_channels, out_channels, expansion=expansion)
        else:
            raise NotImplementedError('Attention type {} is not implemented.'.format(attention_type))
        self.conv1 = conv1
        self.ca = ca

    def forward(self, x):
        return self.ca(x).unsqueeze(2).unsqueeze(3) * self.conv1(x)



def add_feature_map_storage_handle(modules, save_dir, store_input=False, store_output=False, layer_select=-1):
    # modules = find_conv(net, criterion)
    for l, m in enumerate(modules):
        if layer_select == -1 or layer_select == l:
            m.__count_layer__ = l
            m.__save_dir__ = os.path.join(save_dir, '{}_{}'.format(m.__class__.__name__, m.__count_layer__))
            if not os.path.exists(m.__save_dir__):
                os.makedirs(m.__save_dir__)
            # print(m.__save_dir__, m.__count_layer__)
            if store_output and store_input:
                handle = m.register_forward_hook(feature_map_storage_hook)
                m.__storage_handle__ = handle
            elif store_output and not store_input:
                handle = m.register_forward_hook(output_feature_map_storage_hook)
                m.__storage_handle__ = handle
            elif not store_output and store_input:
                handle = m.register_forward_hook(input_feature_map_storage_hook)
                m.__storage_handle__ = handle
            else:
                raise NotImplementedError('Feature maps need to be stored for calling the function ' + add_feature_map_storage_handle.__name__)


def remove_feature_map_storage_handle(modules, store_input=False, store_output=False):
    # modules = find_conv(net, criterion)
    for m in modules:
        if store_output or store_input:
            m.__storage_handle__.remove()
            del m.__storage_handle__
            del m.__save_dir__
            del m.__count_layer__


def reset_feature_map_storage_handle(modules, save_dir, store_input=False, store_output=False, layer_select=-1):
    remove_feature_map_storage_handle(modules, store_input, store_output)
    add_feature_map_storage_handle(modules, save_dir, store_input, store_output, layer_select=layer_select)


def output_feature_map_storage_hook(module, input ,output):
    global count_data
    features = {'output': output}
    torch.save(features, os.path.join(module.__save_dir__, 'Batch{}_Device{}.pt'.format(count_data, torch.cuda.current_device())))
    print('{} {}, Data Batch {}, Device {}'.format(module.__class__.__name__, module.__count_layer__, count_data, torch.cuda.current_device()))


def input_feature_map_storage_hook(module, input, output):
    global count_data
    features = {'input': input[0]}
    torch.save(features, os.path.join(module.__save_dir__, 'Batch{}_Device{}.pt'.format(count_data, torch.cuda.current_device())))
    print('{} {}, Data Batch {}, Device {}'.format(module.__class__.__name__, module.__count_layer__, count_data, torch.cuda.current_device()))


def feature_map_storage_hook(module, input, output):
    global count_data
    features = {'input': input[0], 'output': output}
    torch.save(features, os.path.join(module.__save_dir__, 'Batch{}_Device{}.pt'.format(count_data, torch.cuda.current_device())))
    print('{} {}, Data Batch {}, Device {}'.format(module.__class__.__name__, module.__count_layer__, count_data, torch.cuda.current_device()))
    # from IPython import embed; embed()
    # print(torch.cuda.current_device())


def activate_dconv2d_feature_map_storage(modules, save_dir, store_input=False, store_middle=False, store_output=False,
                                         layer_select=-1):
    for l, m in enumerate(modules):
        if layer_select == -1 or layer_select == l:
            m.__save_dir__ = os.path.join(save_dir, '{}_{}'.format(m.__class__.__name__, l))
            if not os.path.exists(m.__save_dir__):
                os.makedirs(m.__save_dir__)
            m.__count_layer__ = l
            m.__store_input__ = store_input
            m.__store_output__ = store_output
            m.__store_middle__ = store_middle
            print(m.__store_input__, m.__store_output__, m.__store_middle__)


def deactivate_dconv2d_feature_map_storage(modules, layer_select=-1):
    for l, m in enumerate(modules):
        if layer_select == -1 or layer_select == l:
            del m.__save_dir__, m.__count_layer__
            del m.__store_input__
            del m.__store_output__
            del m.__store_middle__


class DConv2d(nn.Module):

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=1, bias=None):
        super(DConv2d, self).__init__()
        self.stride = stride

    def __repr__(self):
        s = '{}(weight_size={}, '.format(self.__class__.__name__, self.weight.shape)
        bias_shape = None if self.bias is None else self.bias.shape
        if hasattr(self, 'projection'):
            s += 'projection_size={}, bias_size={}, '.format(self.projection.shape,bias_shape)
        else:
            s += 'bias_size={}, '.format(bias_shape)
        if hasattr(self, 'projection2'):
            s += 'projection2_size={}, '.format(self.projection2.shape)
        return s
        # if hasattr(self, 'projection2'):
        #     return '{}(weight_size={}, projection_size={}, projection2_size={})'\
        #         .format(self.__class__.__name__, self.weight.shape, self.projection.shape, self.projection2.shape)
        # elif hasattr(self, 'projection'):
        #     return '{}(weight_size={}, projection_size={})'.\
        #         format(self.__class__.__name__, self.weight.shape, self.projection.shape)
        # else:
        #     return '{}(weight_size={}'.\
        #         format(self.__class__.__name__, self.weight.shape)

    def set_params(self, input):
        self.weight = input['weight']
        self.bias = input['bias']
        if 'projection' in input:
            self.projection = input['projection']
        if 'projection2' in input:
            self.projection2 = input['projection2']
        self.padding = self.weight.shape[-1] // 2

    def forward(self, x):
        # print(x.shape, self.weight.shape)
        if hasattr(self, 'projection'):
            bias_shape = None if self.bias is None else self.bias.shape[0]
            # if self.weight.shape[0] == bias_shape:
            #     m = F.conv2d(x, weight=self.weight, bias=self.bias, padding=self.padding, stride=self.stride)
            #     y = F.conv2d(m, weight=self.projection)
            # else:
            m = F.conv2d(x, weight=self.weight, padding=self.padding, stride=self.stride)
            y = F.conv2d(m, weight=self.projection, bias=self.bias)

            if hasattr(self, '__store_input__'):
                self.feature_map_storage(x, m, y)
                # print(self.__class__.__name__, self.__store_input__, self.__store_output__, self.__store_middle__, id(self), torch.cuda.current_device(), self.__count_layer__)
        else:
            y = F.conv2d(x, weight=self.weight, bias=self.bias, padding=self.padding, stride=self.stride)

        if hasattr(self, 'projection2'):
            y = F.conv2d(x, weight=self.projection2)

        #     basis_size = self.weight.shape[1]
        #     if basis_size == x.shape[1]:
        #         x = F.conv2d(x, weight=self.weight, bias=None, padding=self.padding)
        #         x = F.conv2d(x, weight=self.projection, bias=self.bias)
        #     else:
        #         x = torch.cat([F.conv2d(xi, weight=self.weight, bias=None, padding=self.padding)
        #                        for xi in torch.split(x, basis_size, dim=1)], dim=1)
        #         x = F.conv2d(x, weight=self.projection, bias=self.bias)
        # else:
        #     x = F.conv2d(x, weight=self.weight, bias=self.bias, padding=self.padding)
        return y

    def feature_map_storage(self, x, m, y):
        features = {}
        if self.__store_input__:
            features['input'] = x
        if self.__store_middle__:
            features['middle'] = m.norm(dim=(2, 3)).mean(dim=0, keepdim=True)
        if self.__store_output__:
            features['output'] = y
        torch.save(features, os.path.join(self.__save_dir__, 'Batch{}_Device{}.pt'.format(count_data, torch.cuda.current_device())))
        print('{} {}, Data Batch {}, Device {}'.format(self.__class__.__name__, self.__count_layer__, count_data, torch.cuda.current_device()))

    def feature_map_inter_norm(self, m):
        feat = m.detach().cpu().norm(dim=(2, 3)).mean(dim=0, keepdim=True)
        if hasattr(self, '__feature_map_norm__'):
            self.__feature_map_norm__ = torch.cat((self.__feature_map_norm__, feat), dim=0)
        else:
            self.__feature_map_norm__ = feat
        print(self.__class__.__name__, self.__store_input__, self.__store_output__, self.__store_middle__, id(self))

    # def feature_map_inter_norm_2(self, m):
    #     dv = torch.cuda.current_device()
    #     tn = '__feature_map_norm_device{}__'.format(dv)
    #     feat = getattr(self, tn)
    #     feat_add = m.detach().norm(dim=(2, 3))
    #     if hasattr(self, tn):
    #         setattr(self, tn, torch.cat((feat, feat_add), dim=0))
    #     else:
    #         setattr(self, tn, feat_add)


def default_conv(in_channels, out_channels, kernel_size=3, stride=1, bias=True, groups=1, args=None):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=stride, bias=bias, groups=groups)
    return m

def nopad_conv(
        in_channels, out_channels, kernel_size, stride=1, bias=True):

    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=0, stride=stride, bias=bias)

def default_linear(in_channels, out_channels, bias=True):
    return nn.Linear(in_channels, out_channels, bias=bias)

def default_norm(in_channels):
    return nn.BatchNorm2d(in_channels)

def default_act():
    return nn.ReLU(False)

def init_vgg(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            c_out, _, kh, kw = m.weight.size()
            n = kh * kw * c_out
            m.weight.data.normal_(0, math.sqrt(2 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

def init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            _, c_in, kh, kw = m.weight.size()
            n = kh * kw * c_in
            m.weight.data.normal_(0, math.sqrt(2 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            _, c_in = m.weight.size()
            m.weight.data.normal_(0, math.sqrt(1 / c_in))
            m.bias.data.zero_()


# class BasicBlock(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, conv3x3=default_conv, args=None):
#         modules = [conv3x3(in_channels, out_channels, kernel_size, stride=stride, bias=bias, args=args),
#                    nn.BatchNorm2d(out_channels),
#                    nn.ReLU(inplace=True)]
#         super(BasicBlock, self).__init__(*modules)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, conv3x3=default_conv, args=None):
        super(BasicBlock, self).__init__()
        modules = [conv3x3(in_channels, out_channels, kernel_size, stride=stride, bias=bias, args=args),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU()]
        self.layer = nn.Sequential(*modules)
    def forward(self, x):
        return self.layer(x)


# def weight_mask(in_channels, out_channels, weight_type):
#     mask = torch.Tensor()
#     weight = torch.Tensor()
#     if weight_type.find('diagnal') >= 0:
#         print('diagnal')
#         mask1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
#         mask1 = mask1.repeat(out_channels, in_channels, 1, 1)
#         mask2 = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32)
#         mask2 = mask2.repeat(out_channels, in_channels, 1, 1)
#         weight1 = init.xavier_normal_(torch.Tensor(out_channels, in_channels, 3, 3))
#         weight2 = init.xavier_normal_(torch.Tensor(out_channels, in_channels, 3, 3))
#         mask = torch.cat([mask1, mask2, mask], dim=0)
#         weight = torch.cat([weight1, weight2, weight], dim=0)
#
#     if weight_type.find('ellipse') >= 0:
#         print('ellipse')
#         mask1 = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
#         mask1 = mask1.repeat(out_channels, in_channels, 1, 1)
#         mask2 = torch.tensor([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=torch.float32)
#         mask2 = mask2.repeat(out_channels, in_channels, 1, 1)
#         weight1 = init.xavier_normal_(torch.Tensor(out_channels, in_channels, 3, 3))
#         weight2 = init.xavier_normal_(torch.Tensor(out_channels, in_channels, 3, 3))
#         mask = torch.cat([mask1, mask2, mask], dim=0)
#         weight = torch.cat([weight1, weight2, weight], dim=0)
#
#     if weight_type.find('triangle') >= 0:
#         print('triangle')
#         mask1 = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.float32)
#         mask1 = mask1.repeat(out_channels, in_channels, 1, 1)
#         mask2 = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float32)
#         mask2 = mask2.repeat(out_channels, in_channels, 1, 1)
#         mask3 = torch.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=torch.float32)
#         mask3 = mask3.repeat(out_channels, in_channels, 1, 1)
#         mask4 = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.float32)
#         mask4 = mask4.repeat(out_channels, in_channels, 1, 1)
#         weight1 = init.xavier_normal_(torch.Tensor(out_channels, in_channels, 3, 3))
#         weight2 = init.xavier_normal_(torch.Tensor(out_channels, in_channels, 3, 3))
#         weight3 = init.xavier_normal_(torch.Tensor(out_channels, in_channels, 3, 3))
#         weight4 = init.xavier_normal_(torch.Tensor(out_channels, in_channels, 3, 3))
#         mask = torch.cat([mask1, mask2, mask3, mask4, mask], dim=0)
#         weight = torch.cat([weight1, weight2, weight3, weight4, weight], dim=0)
#
#     return mask, weight


def weight_mask(in_channels, out_channels, weight_type):
    mask = torch.Tensor()
    weight = torch.Tensor()
    if weight_type.find('diagnal') >= 0:
        print('diagnal')
        mask1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        mask1 = mask1.repeat(out_channels, in_channels, 1, 1)
        mask2 = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32)
        mask2 = mask2.repeat(out_channels, in_channels, 1, 1)
        weight1 = init.kaiming_uniform_(torch.Tensor(out_channels, in_channels, 3, 3), a=math.sqrt(5))
        weight2 = init.kaiming_uniform_(torch.Tensor(out_channels, in_channels, 3, 3), a=math.sqrt(5))
        mask = torch.cat([mask1, mask2, mask], dim=0)
        weight = torch.cat([weight1, weight2, weight], dim=0)

    if weight_type.find('ellipse') >= 0:
        print('ellipse')
        mask1 = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
        mask1 = mask1.repeat(out_channels, in_channels, 1, 1)
        mask2 = torch.tensor([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=torch.float32)
        mask2 = mask2.repeat(out_channels, in_channels, 1, 1)
        weight1 = init.kaiming_uniform_(torch.Tensor(out_channels, in_channels, 3, 3), a=math.sqrt(5))
        weight2 = init.kaiming_uniform_(torch.Tensor(out_channels, in_channels, 3, 3), a=math.sqrt(5))
        mask = torch.cat([mask1, mask2, mask], dim=0)
        weight = torch.cat([weight1, weight2, weight], dim=0)

    if weight_type.find('triangle') >= 0:
        print('triangle')
        mask1 = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.float32)
        mask1 = mask1.repeat(out_channels, in_channels, 1, 1)
        mask2 = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float32)
        mask2 = mask2.repeat(out_channels, in_channels, 1, 1)
        mask3 = torch.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=torch.float32)
        mask3 = mask3.repeat(out_channels, in_channels, 1, 1)
        mask4 = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.float32)
        mask4 = mask4.repeat(out_channels, in_channels, 1, 1)
        weight1 = init.kaiming_uniform_(torch.Tensor(out_channels, in_channels, 3, 3), a=math.sqrt(5))
        weight2 = init.kaiming_uniform_(torch.Tensor(out_channels, in_channels, 3, 3), a=math.sqrt(5))
        weight3 = init.kaiming_uniform_(torch.Tensor(out_channels, in_channels, 3, 3), a=math.sqrt(5))
        weight4 = init.kaiming_uniform_(torch.Tensor(out_channels, in_channels, 3, 3), a=math.sqrt(5))
        mask = torch.cat([mask1, mask2, mask3, mask4, mask], dim=0)
        weight = torch.cat([weight1, weight2, weight3, weight4, weight], dim=0)

    return mask, weight


class ACConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, args=None):
        super(ACConv, self).__init__()
        self.weight_type = args.weight_type
        self.square_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2,
                                     bias=bias, groups=groups)
        self.square_bn = nn.BatchNorm2d(num_features=out_channels)

        # if self.weight_type.find('skeleton') >= 0:
        self.ver_conv = nn.Conv2d(in_channels, out_channels, (3, 1), stride, padding=(1, 0), bias=False)
        self.hor_conv = nn.Conv2d(in_channels, out_channels, (1, 3), stride, padding=(0, 1), bias=False)
        self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
        self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
        if self.weight_type.find('diagnal') >= 0 or self.weight_type.find('ellipse') >= 0 \
                or self.weight_type.find('triangle') >= 0:
            self.stride = stride
            self.out_channels = out_channels
            mask, weight = weight_mask(in_channels, out_channels, self.weight_type)
            self.register_buffer('mask', mask)
            self.register_parameter('weight', nn.Parameter(weight))
            self.addtional_bn = nn.BatchNorm2d(num_features=mask.shape[0])
            self.register_parameter('coefficient', nn.Parameter(torch.ones(1, mask.shape[0] + out_channels * 3, 1, 1)))

    def forward(self, x):

        square_outputs = self.square_bn(self.square_conv(x))
        # if self.weight_type.find('skeleton') >= 0:
        vertical_outputs = self.ver_bn(self.ver_conv(x))
        horizontal_outputs = self.hor_bn(self.hor_conv(x))
        if self.weight_type.find('diagnal') >= 0 or self.weight_type.find('ellipse') >= 0 \
                or self.weight_type.find('triangle') >= 0:
            additional_outputs = F.conv2d(x, self.mask * self.weight, stride=self.stride, padding=1)
            additional_outputs = self.addtional_bn(additional_outputs)
            outputs = torch.cat([square_outputs, vertical_outputs, horizontal_outputs, additional_outputs], dim=1)
            outputs = outputs * self.coefficient
            # embed()
            outputs = sum(torch.split(outputs, int(self.out_channels), dim=1))
            return outputs
        # print(horizontal_outputs.size())
        return square_outputs + vertical_outputs + horizontal_outputs


class ACSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, args=None):
        super(ACSConv, self).__init__()
        self.weight_type = args.weight_type
        self.square_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2,
                                     bias=bias, groups=groups)
        self.square_bn = nn.BatchNorm2d(num_features=out_channels)

        if self.weight_type.find('skeleton') >= 0:
            self.ver_conv = nn.Conv2d(in_channels, out_channels, (3, 1), stride, padding=(1, 0), bias=False)
            self.hor_conv = nn.Conv2d(in_channels, out_channels, (1, 3), stride, padding=(0, 1), bias=False)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
        elif self.weight_type.find('diagnal') >= 0 or self.weight_type.find('ellipse') >= 0 \
                or self.weight_type.find('triangle') >= 0:
            self.stride = stride
            self.out_channels = out_channels
            mask, weight = weight_mask(in_channels, out_channels, self.weight_type)
            self.register_buffer('mask', mask)
            self.register_parameter('weight', nn.Parameter(weight))
            self.kernel_bn = nn.BatchNorm2d(num_features=mask.shape[0])

    def forward(self, x):

        square_outputs = self.square_bn(self.square_conv(x))
        if self.weight_type.find('skeleton') >= 0:
            vertical_outputs = self.ver_bn(self.ver_conv(x))
            horizontal_outputs = self.hor_bn(self.hor_conv(x))
            return square_outputs + vertical_outputs + horizontal_outputs

        elif self.weight_type.find('diagnal') >= 0 or self.weight_type.find('ellipse') >= 0 \
                or self.weight_type.find('triangle') >= 0:
            kernel_outputs = F.conv2d(x, self.mask * self.weight, stride=self.stride, padding=1)
            kernel_outputs = self.kernel_bn(kernel_outputs)
            outputs = torch.cat([square_outputs, kernel_outputs], dim=1)
            return sum(torch.split(outputs, int(self.out_channels), dim=1))


class EConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, args=None):
        super(EConv, self).__init__()
        expansion = args.expansion
        m = [nn.Conv2d(in_channels, out_channels * expansion, kernel_size, padding=kernel_size // 2, stride=stride, bias=bias, groups=groups),
             nn.BatchNorm2d(out_channels * expansion),
             nn.Conv2d(out_channels * expansion, out_channels, 1, bias=bias)]
        self.m = nn.Sequential(*m)

    def forward(self, x):
        return self.m(x)


class SuperConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, args=None):
        super(SuperConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups
        self.z_dim = self.in_channels * 4
        self.hyper_batchnorm = args.hyper_batchnorm == 'Yes'
        self.z = nn.Parameter(torch.fmod(torch.randn((self.z_dim)).cuda(), 2))
        self.W1 = nn.Parameter(torch.fmod(torch.randn((self.in_channels, self.z_dim, self.z_dim)).cuda(), 2))
        self.B1 = nn.Parameter(torch.fmod(torch.randn((self.in_channels, self.z_dim)).cuda(), 2))
        self.W2 = nn.Parameter(torch.fmod(torch.randn((self.out_channels * self.kernel_size ** 2, self.z_dim)).cuda(), 2))
        self.B2 = nn.Parameter(torch.fmod(torch.randn((self.out_channels * self.kernel_size ** 2)).cuda(), 2))
        if self.hyper_batchnorm:
            self.bn1 = nn.BatchNorm1d(self.z_dim)
            self.bn2 = nn.BatchNorm1d(self.out_channels * self.kernel_size ** 2)
        # if bias:
        #     self.bias = torch.zeros(out_channels).cuda()
        # else:
        self.bias = None

    def forward(self, x):
        a = torch.matmul(self.W1, self.z) + self.B1
        if self.hyper_batchnorm: a = self.bn1(a)
        K = torch.matmul(self.W2, a.unsqueeze(-1)).squeeze() + self.B2
        if self.hyper_batchnorm: K = self.bn2(K)
        weight = K.reshape(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size).permute(1, 0, 2, 3)
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)


class HyperConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, args=None, param=None):
        super(HyperConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups
        self.W1, self.B1, self.W2, self.B2 = param['W1'].clone(), param['B1'].clone(), param['W2'].clone(), param['B2'].clone()
        N_z = 64
        B = 16
        self.B = B
        L = in_channels * out_channels // B ** 2
        self.z = nn.Parameter(torch.fmod(torch.randn((L, 1, N_z, 1)).cuda(), 2))
        # self.z = nn.Parameter(init.kaiming_uniform_(torch.Tensor(L, 1, N_z, 1).cuda(), a=math.sqrt(5)))
        if bias:
            self.bias = torch.zeros(out_channels).cuda()
        else:
            self.bias = None

    def forward(self, x):
        a = torch.matmul(self.W1, self.z) + self.B1.unsqueeze(-1)
        K = torch.matmul(self.W2, a).squeeze(dim=-1) + self.B2
        weight = K.reshape(self.out_channels // self.B, self.in_channels, -1).permute(1, 0, 2).reshape(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size).permute(1, 0, 2, 3)
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)


class HyperConvS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, args=None, param=None):
        super(HyperConvS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups
        self.hypertype = param['args'].hypertype
        self.W1, self.B1, self.W2, self.B2 = param['W1'].clone(), param['B1'].clone(), param['W2'].clone(), param['B2'].clone()
        N_z = 64
        B = 16
        self.B = B
        self.N_z = N_z
        L = in_channels * out_channels // B ** 2
        self.z = nn.ParameterList()
        for o in range(out_channels // B):
            for i in range(in_channels // B):
                self.z.append(nn.Parameter(torch.fmod(torch.randn((N_z)).cuda(), 2)))
        # self.z = nn.Parameter(init.kaiming_uniform_(torch.Tensor(L, 1, N_z, 1).cuda(), a=math.sqrt(5)))
        # if bias:
        #     self.bias = torch.zeros(out_channels).cuda()
        # else:
        self.bias = None

    def hyper(self, z):
        if self.hypertype == 'separate':
            a = torch.matmul(self.W1, z) + self.B1
            K = torch.matmul(self.W2, a.unsqueeze(-1)).squeeze() + self.B2
            weight = K.reshape(self.B, self.B, 3, 3).permute(1, 0, 2, 3)
        elif self.hypertype == 'separate-transpose':
            a = torch.matmul(z, self.W1) + self.B1
            K = torch.matmul(a.reshape(self.B, self.N_z), self.W2) + self.B2
            weight = K.reshape(self.B, self.B, 3, 3)
        else:
            raise NotImplementedError('Error in hyper of HyperConvS')
        return weight

    def aggregate(self):
        ww = []
        for o in range(self.out_channels // self.B):
            w = []
            for i in range(self.in_channels // self.B):
                w.append(self.hyper(self.z[o * self.in_channels // self.B + i]))
            ww.append(torch.cat(w, dim=1))
        return torch.cat(ww, dim=0)

    def forward(self, x):
        weight = self.aggregate()
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding, groups=self.groups)


class HyperNetwork(nn.Module):

    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = nn.Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size*self.f_size)).cuda(),2))
        self.b1 = nn.Parameter(torch.fmod(torch.randn((self.out_size*self.f_size*self.f_size)).cuda(),2))

        self.w2 = nn.Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.z_dim)).cuda(),2))
        self.b2 = nn.Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)).cuda(),2))

    def forward(self, z):

        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel

class Embedding(nn.Module):
    def __init__(self, z_num, z_dim):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim
        h, k = self.z_num

        for i in range(h):
            for j in range(k):
                self.z_list.append(nn.Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2)))

    def forward(self, hyper_net):
        ww = []
        h, k = self.z_num
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(self.z_list[i*k + j]))
            ww.append(torch.cat(w, dim=1))
        return torch.cat(ww, dim=0)