'''
Group Sparsity: The Hinge Between Filter Pruning and Decomposition forNetwork Compression
This module prune ResNeXt. The architecture of the network is not changed. Don't need to add additional 1x1 convs.
Compressing group convolution is similar to depth-wise convolution.
'''
__author__ = 'Yawei Li'
import torch
import torch.nn as nn
import os
import math
import copy
import numpy as np
from easydict import EasyDict as edict
from model.resnext import Block, ResNeXt
from model_hinge.hinge_utility import get_nonzero_index, plot_figure, plot_per_layer_compression_ratio
from model.in_use.flops_counter import get_model_complexity_info
from IPython import embed


########################################################################################################################
# used to edit the modules and parameters during the PG optimization and continuing training stage.
########################################################################################################################

def get_compress_idx(module, percentage, threshold):
    conv1 = module._modules['conv1']
    conv2 = module._modules['conv2']
    conv3 = module._modules['conv3']
    groups = conv2.groups
    print(conv1.weight.data.shape)
    weight1 = conv1.weight.data.squeeze().view(groups, -1)
    weight3 = conv3.weight.data.squeeze().t().reshape(groups, -1)
    joint = torch.cat([weight1, weight3], dim=1)

    norm, pindex = get_nonzero_index(joint, dim='input', counter=1, percentage=percentage, threshold=threshold)

    def _get_compress_statistics(norm, pindex):
        remain_norm = norm[pindex]
        channels = norm.shape[0]
        remain_channels = remain_norm.shape[0]
        remain_norm = remain_norm.detach().cpu()
        stat_channel = [channels, channels - remain_channels, (channels - remain_channels) / channels]
        stat_remain_norm = [remain_norm.max(), remain_norm.mean(), remain_norm.min()]
        return edict({'stat_channel': stat_channel, 'stat_remain_norm': stat_remain_norm,
                      'remain_norm': remain_norm, 'pindex': pindex})

    return [_get_compress_statistics(norm, pindex)]


# ========
# Compress module parameters
# ========
def compress_module_param(percentage, threshold, **kwargs):
    module = kwargs['module']

    conv1 = module._modules['conv1']
    batchnorm1 = module._modules['bn1']
    conv2 = module._modules['conv2']
    batchnorm2 = module._modules['bn2']
    conv3 = module._modules['conv3']

    groups = conv2.groups
    gs = conv2.in_channels // groups

    ws1 = conv1.weight.data.shape
    weight1 = conv1.weight.data.squeeze().view(groups, gs * conv1.in_channels)
    bn_weight1 = batchnorm1.weight.data
    bn_bias1 = batchnorm1.bias.data
    bn_mean1 = batchnorm1.running_mean.data
    bn_var1 = batchnorm1.running_var.data

    ws2 = conv2.weight.data.shape
    weight2 = conv2.weight.data.view(groups, ws2[1] * ws2[2] * ws2[3] * gs) #do not need transpose here
    bn_weight2 = batchnorm2.weight.data
    bn_bias2 = batchnorm2.bias.data
    bn_mean2 = batchnorm2.running_mean.data
    bn_var2 = batchnorm2.running_var.data

    ws3 = conv3.weight.data.shape
    weight3 = conv3.weight.data.squeeze().t().reshape(groups, gs * conv3.out_channels)

    if 'load_original_param' in kwargs and kwargs['load_original_param']: # need to pay special attention here
        weight1_teach = kwargs['module_teacher']._modules['conv1'].weight.data.squeeze()\
            .view(conv1.groups, conv1.in_channels // conv1.groups * conv1.out_channels)
        weight3_teach = kwargs['module_teacher']._modules['conv3'].weight.data.squeeze()\
            .t().view(conv2.groups, conv2.in_channels // conv2.groups * conv2.out_channels)
        joint = torch.cat([weight1_teach, weight3_teach], dim=1)
    else:
        joint = torch.cat([weight1, weight3], dim=1)

    _, pindex = get_nonzero_index(joint, dim='input', counter=1, percentage=percentage, threshold=threshold)
    # with print_array_on_one_line():
    #     print('Index of Projection1: {}'.format(pindex1.detach().cpu().numpy()))
    #     print('Index of Projection2: {}'.format(pindex2.detach().cpu().numpy()))
    pl = len(pindex)
    conv1.weight = nn.Parameter(torch.index_select(weight1, dim=0, index=pindex).view(pl * gs, ws1[1], 1, 1))
    conv1.out_channels = pl * gs

    batchnorm1.weight = nn.Parameter(torch.index_select(bn_weight1.view(groups, gs), dim=0, index=pindex).view(-1))
    batchnorm1.bias = nn.Parameter(torch.index_select(bn_bias1.view(groups, gs), dim=0, index=pindex).view(-1))
    batchnorm1.running_mean = torch.index_select(bn_mean1.view(groups, gs), dim=0, index=pindex).view(-1)
    batchnorm1.running_var = torch.index_select(bn_var1.view(groups, gs), dim=0, index=pindex).view(-1)
    batchnorm1.num_features = pl * gs

    conv2.weight = nn.Parameter(torch.index_select(weight2, dim=0, index=pindex).view(pl * gs, gs, ws2[2], ws2[3]))
    conv2.out_channels, conv2.in_channels, conv2.groups = pl * gs, pl * gs, pl
    batchnorm2.weight = nn.Parameter(torch.index_select(bn_weight2.view(groups, gs), dim=0, index=pindex).view(-1))
    batchnorm2.bias = nn.Parameter(torch.index_select(bn_bias2.view(groups, gs), dim=0, index=pindex).view(-1))
    batchnorm2.running_mean = torch.index_select(bn_mean2.view(groups, gs), dim=0, index=pindex).view(-1)
    batchnorm2.running_var = torch.index_select(bn_var2.view(groups, gs), dim=0, index=pindex).view(-1)
    batchnorm2.num_features = pl * gs

    conv3.weight = nn.Parameter(torch.index_select(weight3, dim=0, index=pindex).view(ws3[0], pl * gs, 1, 1))
    conv3.in_channels = pl * gs


def make_model(inputs):
    """
    base_module is the compressed module
    """
    class Hinge(ResNeXt):

        def __init__(self, args):
            self.args = args[0]
            self.ckp = args[1]
            super(Hinge, self).__init__(self.args)

            # traning or loading phase of network compression
            if self.args.model.lower().find('hinge') >= 0 and not self.args.test_only:
                self.load(strict=True)
                if self.args.layer_balancing:
                    # need to calculate the layer-balancing regularizer during both training and loading phase.
                    self.layer_reg = self.calc_regularization()
                    for l, reg in enumerate(self.layer_reg):
                        print('Block {:<2}: regularization {:<2.4f}'.format(l + 1, reg))

            if self.args.data_train.find('CIFAR') >= 0:
                self.input_dim = (3, 32, 32)
            elif self.args.data_train.find('Tiny_ImageNet') >= 0:
                self.input_dim = (3, 64, 64)
            else:
                self.input_dim = (3, 224, 224)
            self.flops, self.params = get_model_complexity_info(self, self.input_dim, print_per_layer_stat=False)

        def find_modules(self):
            return [m for m in self.modules() if isinstance(m, Block)]

        def sparse_param(self, module):
            groups = module._modules['conv2'].groups
            param1 = module.state_dict(keep_vars=True)['conv1.weight']
            param2 = module.state_dict(keep_vars=True)['conv3.weight']
            param1_new = param1.data.squeeze().view(groups, -1)
            param2_new = param2.data.squeeze().t().reshape(groups, -1)
            return param1, param2, torch.cat([param1_new, param2_new], dim=1)

        def calc_sparsity_solution(self, param, reg, sparsity_reg):
            eps = 1e-8
            n = torch.norm(param, p=2, dim=1)
            if sparsity_reg == 'l1':
                scale = torch.max(1 - reg / (n + eps), torch.zeros_like(n, device=n.device))
            elif sparsity_reg == 'l1-2':
                w = torch.max(n - reg, torch.zeros_like(n, device=n.device))
                w_norm = torch.norm(w, p=2)
                scale = (1 + reg / w_norm) * torch.max(1 - reg / (n + eps), torch.zeros_like(n, device=n.device))
            elif sparsity_reg == 'l1d2':
                threshold = 54 ** (1 / 3) / 4 * reg ** (2 / 3)
                idx = torch.nonzero((n > threshold).to(torch.float32), as_tuple=True)
                phi = torch.acos(reg / 8 * (n[idx[0]] / 3) ** (-3 / 2))
                scale = torch.zeros_like(n, device=n.device)
                scale[idx[0]] = 2 / 3 * (1 + torch.cos(2 / 3 * (-phi + math.pi)))
            elif sparsity_reg == 'logsum':
                c1 = n - eps
                c2 = c1 ** 2 - 4 * (reg - n * eps)
                scale = torch.zeros_like(n, device=n.device)
                idx = torch.nonzero((c2 > 0).to(torch.float32), as_tuple=True)
                scale[idx[0]] = (c1[idx[0]] + torch.sqrt(c2[idx[0]])) / 2 / n[idx[0]]
                if torch.isnan(torch.sum(scale)):
                    embed()
            else:
                raise NotImplementedError('Regularization method {} not implemented.'.format(sparsity_reg))
            return scale

        def proximal_operator(self, lr, batch, regularization):
            modules = self.find_modules()
            for l, module in enumerate(modules):
                if batch == 100 and l == 0:
                    print('Regularization is {}'.format(regularization))
                groups = module._modules['conv2'].groups
                param1, param2, param = self.sparse_param(module)
                ws1 = param1.shape
                ws3 = param2.shape
                if self.args.layer_balancing:
                    reg = regularization * self.layer_reg[l] * lr
                else:
                    reg = regularization * lr

                scale = self.calc_sparsity_solution(param, reg, self.args.sparsity_regularizer)
                scale1 = scale.repeat(int(np.prod(np.array(ws1)) / groups), 1).t()
                scale2 = scale.repeat(int(np.prod(np.array(ws3)) / groups), 1).t()
                param1.data = torch.mul(scale1, param1.squeeze().view(groups, -1)).view(ws1)
                param2.data = torch.mul(scale2, param2.squeeze().t().reshape(groups, -1)).view(ws3[1], ws3[0]).t().unsqueeze(2).unsqueeze(3)

        def compute_loss(self, batch, current_epoch, converging):
            """
            loss = ||Y - Yc||^2 + lambda * (||A_1||_{2,1} + ||A_2 ^T||_{2,1})
            """
            modules = self.find_modules()
            lambda_factor = self.args.regularization_factor
            q = self.args.q
            loss_proj = []
            for l, module in enumerate(modules):
                _, _, projection = self.sparse_param(module)
                loss_proj.append(torch.sum(torch.sum(projection ** 2, dim=1) ** (q / 2)) ** (1 / q))
                # save the norm distribution during the optimization phase
                if not converging and (batch + 1) % 100 == 0:
                    path = os.path.join(self.args.dir_save, self.args.save, 'Filter{}_norm_distribution'.format(l+1))
                    if not os.path.exists(path):
                        os.makedirs(path)
                    filename = os.path.join(path, 'Epoch{}_Batch{}'.format(current_epoch, batch + 1))
                    plot_figure(projection, l+1, filename)
            lossp = sum(loss_proj)
            if self.args.optimizer == 'SGD':
                lossp *= lambda_factor
            return lossp

        def calc_regularization(self):
            layer_reg = []
            modules = self.find_modules()
            for l, module in enumerate(modules):
                _, _, projection = self.sparse_param(module)
                layer_reg.append(torch.norm(projection, p=2, dim=1).mean().detach().cpu().numpy())
            return layer_reg

        def compress(self, **kwargs):
            if 'load_original_param' in kwargs and kwargs['load_original_param']:
                print('Start Pruning, Parameters from the original network.')
                model_teacher = kwargs['model_teacher']
                # exchange the state dictionary of the current and the teacher network
                state_current = copy.deepcopy(self.state_dict())
                state_teacher = copy.deepcopy(model_teacher.state_dict())
                model_teacher.get_model().load_state_dict(state_current, strict=True)
                self.load_state_dict(state_teacher, strict=True)

                # get the modules
                modules_teacher = [m for m in model_teacher.get_model().modules() if isinstance(m, Block)]
                modules = self.find_modules()

                # compress each module
                for l in range(len(modules)):
                    kwargs_module = {'load_original_param': kwargs['load_original_param'], 'module': modules[l],
                                     'module_teacher': modules_teacher[l]}
                    compress_module_param(self.args.remain_percentage, self.args.threshold, **kwargs_module)

            else:
                print('Start Pruning, Parameters from the trained sparsity-regularized model.')
                modules = self.find_modules()
                for l in range(len(modules)):
                    compress_module_param(self.args.remain_percentage, self.args.threshold, **{'module': modules[l]})
            self.set_channels()

        def print_compress_info(self, epoch_continue):
            info_compress = edict(
                {'flops_original': self.flops / 10. ** 9, 'flops_remaining': self.flops_compress / 10. ** 9,
                 'flops_ratio': self.flops_compress / self.flops * 100,
                 'params_original': self.params / 10. ** 3, 'params_remaining': self.params_compress / 10. ** 3,
                 'params_ratio': self.params_compress / self.params * 100,
                 'epoch_continue': epoch_continue})

            self.ckp.write_log('\nThe information about the compressed network is as follows:')

            ratio_per_layer = [[]]
            for l, m in enumerate(self.find_modules()):
                info_compress['info_layer{}'.format(l)] = \
                    get_compress_idx(m, self.args.remain_percentage, self.args.threshold)
                for i, info_layer in enumerate(info_compress['info_layer{}'.format(l)]):
                    self.ckp.write_log(
                        'Block{:<2} Layer{}: channels -> orginal={:<2}, compress={:<2}, ratio={:2.4f};'
                        ' norm -> max={:2.6f}, mean={:2.6f}, min={:2.6f}'
                        .format(l, i + 1, *info_layer.stat_channel, *info_layer.stat_remain_norm))
                    ratio_per_layer[i].append(info_layer.stat_channel[-1])

            self.ckp.write_log('Hinge searching epochs: {:<3}\n'
                               'Final nullifying threshold {:2.8f}\n'
                               'FLOPs original: {:.4f} [G]\nFLOPs remaining: {:.4f} [G]\nFLOPs ratio: {:2.2f}\n'
                               'Params original: {:.2f} [k]\nParams remaining: {:.2f} [k]\nParams ratio: {:2.2f}\n'
                               .format(epoch_continue, self.args.threshold,
                                       info_compress.flops_original, info_compress.flops_remaining,
                                       info_compress.flops_ratio,
                                       info_compress.params_original, info_compress.params_remaining,
                                       info_compress.params_ratio))

            filename = os.path.join(self.args.dir_save, self.args.save, 'per_layer_compression_ratio.png')
            plot_per_layer_compression_ratio(ratio_per_layer, filename)
            torch.save(info_compress, os.path.join(self.args.dir_save, self.args.save, 'compression_information.pt'))

        def set_channels(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    ws = m.weight.shape
                    # embed()
                    m.out_channels = ws[0]
                    if ws[2] == 1:
                        m.in_channels = ws[1]
                    else:
                        if ws[1] != 3:
                            m.in_channels = ws[0]
                            m.groups = ws[0] // ws[1]
                elif isinstance(m, nn.BatchNorm2d):
                    m.num_features = m.weight.size()[0]
        #
        # def set_channels(self):
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv2d):
        #             ws = m.weight.shape
        #             m.out_channels = ws[0]
        #             if m.groups != 1:
        #                 m.in_channels = ws[0]
        #                 m.groups = ws[0] // ws[1]
        #             else:
        #                 m.in_channels = ws[1]
        #         elif isinstance(m, nn.BatchNorm2d):
        #             m.num_features = m.weight.size()[0]

        def load_state_dict(self, state_dict, strict=True):
            if strict:
                # used to load the model parameters during training
                super(Hinge, self).load_state_dict(state_dict, strict)
            else:
                # used to load the model parameters during test
                own_state = self.state_dict(keep_vars=True)
                for name, param in state_dict.items():
                    if name in own_state:
                        if isinstance(param, nn.Parameter):
                            param = param.data
                        if param.size() != own_state[name].size():
                            own_state[name].data = param
                        else:
                            own_state[name].data.copy_(param)
                self.set_channels()

        def load(self, strict=True):
            if self.args.load:
                state_dict = torch.load(self.args.teacher)
            elif not self.args.load and self.args.pretrained:
                state_dict = torch.load(self.args.pretrained)
            else:
                raise NotImplementedError('Do not need to load {} in this mode'.format(self.__class__.__name__))

            self.load_state_dict(state_dict, strict=strict)

    return Hinge(inputs)
