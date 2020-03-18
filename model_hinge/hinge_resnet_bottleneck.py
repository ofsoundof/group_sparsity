"""
Group Sparsity: The Hinge Between Filter Pruning and Decomposition forNetwork Compression
This module implement the proposed Hinge method to ResNet164.
"""
__author__ = 'Yawei Li'
import torch
import torch.nn as nn
import os
import math
import copy
from easydict import EasyDict as edict
from model_hinge.hinge_utility import get_nonzero_index, plot_figure, plot_per_layer_compression_ratio
from model.in_use.flops_counter import get_model_complexity_info
from model.resnet164 import ResNet164
from model.resnet50 import ResNet50
from torchvision.models.resnet import Bottleneck
from IPython import embed


########################################################################################################################
# used to edit the modules and parameters during the PG optimization and continuing training stage.
########################################################################################################################

def get_compress_idx(module, percentage, threshold, p1_p2_same_ratio):
    weight1 = module._modules['conv1'].weight.data.squeeze().t()
    weight3 = module._modules['conv3'].weight.data.squeeze().t()

    norm1, pindex1 = get_nonzero_index(weight1, dim='output', counter=1, percentage=percentage, threshold=threshold)
    fix_channel = len(pindex1) if p1_p2_same_ratio else 0
    norm3, pindex3 = get_nonzero_index(weight3, dim='intput', counter=1, percentage=percentage, threshold=threshold, fix_channel=fix_channel)

    def _get_compress_statistics(norm, pindex):
        remain_norm = norm[pindex]
        channels = norm.shape[0]
        remain_channels = remain_norm.shape[0]
        remain_norm = remain_norm.detach().cpu()
        stat_channel = [channels, channels - remain_channels, (channels - remain_channels) / channels]
        stat_remain_norm = [remain_norm.max(), remain_norm.mean(), remain_norm.min()]
        return edict({'stat_channel': stat_channel, 'stat_remain_norm': stat_remain_norm,
                      'remain_norm': remain_norm, 'pindex': pindex})
    return [_get_compress_statistics(norm1, pindex1), _get_compress_statistics(norm3, pindex3)]


# ========
# Compress module parameters
# ========
def compress_module_param(percentage, threshold, p1_p2_same_ratio, **kwargs):
    module = kwargs['module']
    
    conv1 = module._modules['conv1']
    batchnorm1 = module._modules['bn1']
    conv2 = module._modules['conv2']
    batchnorm2 = module._modules['bn2']
    conv3 = module._modules['conv3']

    weight1 = conv1.weight.data.squeeze().t()
    bn_weight1 = batchnorm1.weight.data
    bn_bias1 = batchnorm1.bias.data
    bn_mean1 = batchnorm1.running_mean.data
    bn_var1 = batchnorm1.running_var.data

    ws2 = conv2.weight.data.shape
    weight2 = conv2.weight.data.view(ws2[0], ws2[1] * ws2[2] * ws2[3]).t()
    bn_weight2 = batchnorm2.weight.data
    bn_bias2 = batchnorm2.bias.data
    bn_mean2 = batchnorm2.running_mean.data
    bn_var2 = batchnorm2.running_var.data

    weight3 = conv3.weight.data.squeeze().t()

    if 'load_original_param' in kwargs and kwargs['load_original_param']: # need to pay special attention here
        weight1_teach = kwargs['module_teacher']._modules['conv1'].weight.data.squeeze().t()
        weight3_teach = kwargs['module_teacher']._modules['conv3'].weight.data.squeeze()
    else:
        weight1_teach = weight1 #TODO: whether to use copy.copy here?
        weight3_teach = weight3
    _, pindex1 = get_nonzero_index(weight1_teach, dim='output', counter=1, percentage=percentage, threshold=threshold)
    fix_channel = len(pindex1) if p1_p2_same_ratio else 0
    _, pindex3 = get_nonzero_index(weight3_teach, dim='intput', counter=1, percentage=percentage, threshold=threshold, fix_channel=fix_channel)
    # with print_array_on_one_line():
    #     print('Index of Projection1: {}'.format(pindex1.detach().cpu().numpy()))
    #     print('Index of Projection2: {}'.format(pindex2.detach().cpu().numpy()))
    pl1, pl3 = len(pindex1), len(pindex3)
    # conv1
    conv1.weight = nn.Parameter(torch.index_select(weight1, dim=1, index=pindex1).t().view(pl1, -1, 1, 1))
    conv1.out_channels = pl1
    # batchnorm1
    batchnorm1.weight = nn.Parameter(torch.index_select(bn_weight1, dim=0, index=pindex1))
    batchnorm1.bias = nn.Parameter(torch.index_select(bn_bias1, dim=0, index=pindex1))
    batchnorm1.running_mean = torch.index_select(bn_mean1, dim=0, index=pindex1)
    batchnorm1.running_var = torch.index_select(bn_var1, dim=0, index=pindex1)
    batchnorm1.num_features = pl1
    # conv2
    index = torch.repeat_interleave(pindex1, ws2[2] * ws2[3]) * ws2[2] * ws2[3] + \
            torch.tensor(range(0, ws2[2] * ws2[3])).repeat(pindex1.shape[0]).cuda()
    weight2 = torch.index_select(weight2, dim=0, index=index)
    weight2 = torch.index_select(weight2, dim=1, index=pindex3)
    conv2.weight = nn.Parameter(weight2.view(pl3, pl1, 3, 3))
    conv2.out_channels, conv2.in_channels = pl3, pl1
    # batchnorm2
    batchnorm2.weight = nn.Parameter(torch.index_select(bn_weight2, dim=0, index=pindex3))
    batchnorm2.bias = nn.Parameter(torch.index_select(bn_bias2, dim=0, index=pindex3))
    batchnorm2.running_mean = torch.index_select(bn_mean2, dim=0, index=pindex3)
    batchnorm2.running_var = torch.index_select(bn_var2, dim=0, index=pindex3)
    batchnorm2.num_features = pl3
    # conv3
    conv3.weight = nn.Parameter(torch.index_select(weight3, dim=0, index=pindex3).view(-1, pl3, 1, 1))
    conv3.in_channels = pl3



def make_model(args, ckp):
    """
    base_module is the compressed module
    """
    if args.depth == 164:
        base = ResNet164
    elif args.depth == 50:
        base = ResNet50
    else:
        raise NotImplementedError('ResNet{}-Bottleneck is not implemented.'.format(args.depth))

    class Hinge(base):

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
                        print('ResBlock{:<2}, 1x1 conv1 / 1x1 conv3: {:<2.4f} / {:<2.4f}'.format(l + 1, reg[0], reg[1]))

            if self.args.data_train.find('CIFAR') >= 0:
                self.input_dim = (3, 32, 32)
            elif self.args.data_train.find('Tiny_ImageNet') >= 0:
                self.input_dim = (3, 64, 64)
            else:
                self.input_dim = (3, 224, 224)
            self.flops, self.params = get_model_complexity_info(self, self.input_dim, print_per_layer_stat=False)

            # if self.args.model.lower().find('resnet') >= 0:
            self.register_buffer('running_grad_ratio', None)

        def find_modules(self):
            return [m for m in self.modules() if isinstance(m, Bottleneck)]

        def sparse_param(self, module):
            param1 = module.state_dict(keep_vars=True)['conv1.weight']
            param2 = module.state_dict(keep_vars=True)['conv3.weight']
            return param1, param2

        def calc_regularization(self):
            layer_reg = []
            for m in self.find_modules():
                projection1, projection2 = self.sparse_param(m)
                mean1 = torch.norm(projection1.squeeze().t(), p=2, dim=0).mean().detach().cpu().numpy()
                mean2 = torch.norm(projection2.squeeze().t(), p=2, dim=1).mean().detach().cpu().numpy()
                layer_reg.append([mean1, mean2])
            return layer_reg

        def calc_sparsity_solution(self, param, dim, reg, sparsity_reg):
            eps = 1e-8
            n = torch.norm(param.squeeze().t(), p=2, dim=dim)
            # if torch.isnan(n[0]):
            #     embed()
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
                # if phi.shape[0] == 0:
                # if torch.isnan(phi[0]):
                #     embed()
                scale = torch.zeros_like(n, device=n.device)
                scale[idx[0]] = 2 / 3 * (1 + torch.cos(2 / 3 * (-phi + math.pi)))

                # lambda_factor = 3 / 4 * reg ** (2 / 3)
                # phi = lambda_factor / 8 * (n / 3) ** (-3 / 2)
                # scale = (n > reg).to(torch.float32) * 2 / 3 * (1 + torch.cos(2 / 3 * (math.pi - phi)))
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
                # scale2 = torch.max(1 - threshold / (n2 + eps), torch.zeros_like(n2, device=n2.device))
            if dim == 0:
                scale = scale.repeat(param.shape[1], 1)
            else:
                scale = scale.repeat(param.shape[0], 1).t()
            return scale

        def p1_p2_gradient(self, modules):
            p1_grad = []
            p2_grad = []
            for l, m in enumerate(modules):
                projection1, projection2 = self.sparse_param(m)
                p1_grad.append(torch.sum(torch.norm(projection1.grad.squeeze().t(), dim=0)))
                p2_grad.append(torch.sum(torch.norm(projection2.grad.squeeze().t(), dim=1)))
            return sum(p1_grad), sum(p2_grad)

        def update_grad_ratio(self):
            modules = self.find_modules()
            p1_grad, p2_grad = self.p1_p2_gradient(modules)
            grad_ratio = p1_grad / p2_grad
            if self.running_grad_ratio is None:
                self.running_grad_ratio = grad_ratio
            else:
                momentum = 0.9
                self.running_grad_ratio = self.running_grad_ratio * (1 - momentum) + grad_ratio * momentum
                device = self.running_grad_ratio.device
                self.running_grad_ratio = min(max(self.running_grad_ratio, torch.tensor(5/9, device=device)), torch.tensor(9/5, device=device))

        def proximal_operator(self, lr, batch, regularization):
            modules = self.find_modules()
            for l in range(len(modules)):
                param1, param2 = self.sparse_param(modules[l])
                if batch == 100 and l == 0:
                    print('Regularization is {}'.format(regularization))

                if self.args.layer_balancing:
                    reg1 = regularization * lr * self.layer_reg[l][0]
                    reg2 = regularization * lr * self.layer_reg[l][1]
                else:
                    reg1 = regularization * lr
                    reg2 = regularization * lr

                scale1 = self.calc_sparsity_solution(param1, 0, reg1, self.args.sparsity_regularizer)
                scale2 = self.calc_sparsity_solution(param2, 1, reg2, self.args.sparsity_regularizer)

                # if batch % 100 == 0 and l == 0:
                #     with print_array_on_one_line():
                #         print('Scale1: \n{}'.format(scale1.detach().cpu().numpy()))
                #         print('Scale2: \n{}'.format(scale2.detach().cpu().numpy()))

                param1.data = torch.mul(scale1, param1.squeeze().t()).t().view(param1.shape)
                param2.data = torch.mul(scale2, param2.squeeze().t()).t().view(param2.shape)

        def compute_loss(self, batch, current_epoch, converging):
            """
            loss = ||Y - Yc||^2 + lambda * (||A_1||_{2,1} + ||A_2 ^T||_{2,1})
            """
            lambda_factor = self.args.regularization_factor
            q = self.args.q
            loss_proj1 = []
            loss_proj2 = []
            for l, m in enumerate(self.find_modules()):
                projection1, projection2 = self.sparse_param(m)
                loss_proj1.append(torch.sum(torch.sum(projection1.squeeze().t() ** 2, dim=0) ** (q / 2)) ** (1 / q))
                loss_proj2.append(torch.sum(torch.sum(projection2.squeeze().t() ** 2, dim=1) ** (q / 2)) ** (1 / q))
                # save the norm distribution during the optimization phase
                if not converging and (batch + 1) % 300 == 0:
                    path = os.path.join(self.args.dir_save, self.args.save, 'ResBlock{}_norm_distribution'.format(l))
                    if not os.path.exists(path):
                        os.makedirs(path)
                    filename1 = os.path.join(path, 'P1_Epoch{}_Batch{}'.format(current_epoch, batch + 1))
                    filename2 = os.path.join(path, 'P2_Epoch{}_Batch{}'.format(current_epoch, batch + 1))
                    plot_figure(projection1.squeeze().t(), l, filename1)
                    plot_figure(projection2.squeeze(), l, filename2)
            lossp1 = sum(loss_proj1) #/ len(loss_proj1)
            lossp2 = sum(loss_proj2) #/ len(loss_proj2)
            if self.args.optimizer == 'SGD':
                lossp1 *= lambda_factor
                lossp2 *= lambda_factor
            return lossp1, lossp2

        def compress(self, **kwargs):
            if 'load_original_param' in kwargs and kwargs['load_original_param']:
                print('Start Compression, Parameters from the original network.')
                model_teacher = kwargs['model_teacher']
                # exchange the state dictionary of the current and the teacher network
                state_current = copy.deepcopy(self.state_dict())
                state_teacher = copy.deepcopy(model_teacher.state_dict())
                model_teacher.get_model().load_state_dict(state_current, strict=True)
                self.load_state_dict(state_teacher, strict=True)

                # get the modules
                modules_teacher = [m for m in model_teacher.get_model().modules() if isinstance(m, Bottleneck)]
                modules = self.find_modules()

                # compress each module
                for l in range(len(modules)):
                    kwargs_module = {'load_original_param': kwargs['load_original_param'], 'module': modules[l],
                                     'module_teacher': modules_teacher[l]}
                    compress_module_param(self.args.remain_percentage, self.args.threshold, self.args.p1_p2_same_ratio, **kwargs_module)

            else:
                print('Start Compression, Parameters from the trained sparsity-regularized model.')
                modules = self.find_modules()
                for l in range(len(modules)):
                    compress_module_param(self.args.remain_percentage, self.args.threshold, self.args.p1_p2_same_ratio, **{'module': modules[l]})

        def print_compress_info(self, epoch_continue):
            info_compress = edict(
                {'flops_original': self.flops / 10. ** 9, 'flops_remaining': self.flops_compress / 10. ** 9,
                 'flops_ratio': self.flops_compress / self.flops * 100,
                 'params_original': self.params / 10. ** 3, 'params_remaining': self.params_compress / 10. ** 3,
                 'params_ratio': self.params_compress / self.params * 100,
                 'epoch_continue': epoch_continue})

            self.ckp.write_log('\nThe information about the compressed network is as follows:')

            ratio_per_layer = [[], []]
            for l, m in enumerate(self.find_modules()):
                info_compress['info_layer{}'.format(l)] = \
                    get_compress_idx(m, self.args.remain_percentage, self.args.threshold, self.args.p1_p2_same_ratio)
                for i, info_layer in enumerate(info_compress['info_layer{}'.format(l)]):
                    self.ckp.write_log(
                        'ResBlock{:<2} Layer{}: channels -> orginal={:<2}, compress={:<2}, ratio={:2.4f};'
                        ' norm -> max={:2.6f}, mean={:2.6f}, min={:2.6f}'
                        .format(l, 2 * i + 1, *info_layer.stat_channel, *info_layer.stat_remain_norm))
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
                    m.out_channels = m.weight.size()[0]
                    m.in_channels = m.weight.size()[1]
                elif isinstance(m, nn.BatchNorm2d):
                    m.num_features = m.weight.size()[0]

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
            elif not self.args.load and self.args.pretrain:
                state_dict = torch.load(self.args.pretrain)
            else:
                state_dict = None
                # raise NotImplementedError('Do not need to load {} in this mode'.format(self.__class__.__name__))
            if state_dict is not None:
                self.load_state_dict(state_dict, strict=strict)

    return Hinge((args, ckp))

