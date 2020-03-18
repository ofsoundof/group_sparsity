"""
Group Sparsity: The Hinge Between Filter Pruning and Decomposition forNetwork Compression
This module implement the proposed Hinge method to Wide ResNet.
"""
__author__ = 'Yawei Li'
import torch
import torch.nn as nn
import os
import math
from easydict import EasyDict as edict
from importlib import import_module
from model.wide_resnet import wide_basic
from model_hinge.hinge_utility import init_weight_proj, get_nonzero_index, plot_figure, plot_per_layer_compression_ratio
from model.in_use.flops_counter import get_model_complexity_info
from IPython import embed


########################################################################################################################
# used to edit the modules and parameters during the optimization and fine-tuning step
########################################################################################################################

def get_compress_idx(module, percentage, threshold, p1_p2_same_ratio):
    conv12 = module._modules['conv1']._modules['1']
    conv22 = module._modules['conv2']._modules['1']
    projection1 = conv12.weight.data.squeeze().t()
    projection2 = conv22.weight.data.squeeze().t()
    norm1, pindex1 = get_nonzero_index(projection1, dim='output', counter=1, percentage=percentage, threshold=threshold)
    fix_channel = len(pindex1) if p1_p2_same_ratio else 0
    norm2, pindex2 = get_nonzero_index(projection2, dim='input', counter=1, percentage=percentage, threshold=threshold,
                                       fix_channel=fix_channel)
    def _get_compress_statistics(norm, pindex):
        remain_norm = norm[pindex]
        channels = norm.shape[0]
        remain_channels = remain_norm.shape[0]
        remain_norm = remain_norm.detach().cpu()
        stat_channel = [channels, channels - remain_channels, (channels - remain_channels) / channels]
        stat_remain_norm = [remain_norm.max(), remain_norm.mean(), remain_norm.min()]
        return edict({'stat_channel': stat_channel, 'stat_remain_norm': stat_remain_norm,
                      'remain_norm': remain_norm, 'pindex': pindex})

    return [_get_compress_statistics(norm1, pindex1), _get_compress_statistics(norm2, pindex2)]


# ========
# Modify submodules
# ========
def modify_submodules(module):
    # The bias of conv1 and conv2 of Wide ResNet is not None. So need to deal with that.
    conv1 = [module._modules['conv1'], nn.Conv2d(10, 10, 1, bias=True)]
    module._modules['conv1'] = nn.Sequential(*conv1)
    conv2 = [module._modules['conv2'], nn.Conv2d(10, 10, 1, bias=True)]
    module._modules['conv2'] = nn.Sequential(*conv2)
    # module.optimization = True #Do not need the optimization flag here.


# ========
# Set submodule parameters
# ========
def set_module_param(module, params):

    ws1 = params.weight1.size()
    ps1 = params.projection1.size()
    ws2 = params.weight2.size()
    ps2 = params.projection2.size()

    conv11 = module._modules['conv1']._modules['0']
    conv12 = module._modules['conv1']._modules['1']
    conv21 = module._modules['conv2']._modules['0']
    conv22 = module._modules['conv2']._modules['1']

    # set conv11
    conv11.in_channels = ws1[1]
    conv11.out_channels = ws1[0]
    conv11.weight.data = params.weight1.data
    conv11.bias = None

    # set conv12
    conv12.in_channels = ps1[1]
    conv12.out_channels = ps1[0]
    conv12.weight.data = params.projection1.data
    conv12.bias.data = params.bias1.data
    # Note that the bias term is added to the second conv
    # Do not need to set batchnorm1, activation, and batchnorm2.

    # set conv11
    conv21.in_channels = ws2[1]
    conv21.out_channels = ws2[0]
    conv21.weight.data = params.weight2.data
    conv21.bias = None

    # set conv12
    conv22.in_channels = ps2[1]
    conv22.out_channels = ps2[0]
    conv22.weight.data = params.projection2.data
    conv22.bias.data = params.bias2.data


# ========
# Compress module parameters
# ========
def compress_module_param(module, percentage, threshold, p1_p2_same_ratio):
    conv11 = module._modules['conv1']._modules['0']
    conv12 = module._modules['conv1']._modules['1']
    conv21 = module._modules['conv2']._modules['0']
    conv22 = module._modules['conv2']._modules['1']
    batchnorm2 = module._modules['bn2']
    # embed()

    ws1 = conv11.weight.shape
    #weight1 = conv11.weight.data.view(ws1[0], -1).t()
    projection1 = conv12.weight.data.squeeze().t()
    bias1 = conv12.bias.data

    bn_weight2 = batchnorm2.weight.data
    bn_bias2 = batchnorm2.bias.data
    bn_mean2 = batchnorm2.running_mean.data
    bn_var2 = batchnorm2.running_var.data

    ws2 = conv21.weight.shape
    weight2 = conv21.weight.data.view(ws2[0], -1).t()
    projection2 = conv22.weight.data.squeeze().t()
    #bias2 = conv22.bias.data


    _, pindex1 = get_nonzero_index(projection1, dim='output', counter=1, percentage=percentage, threshold=threshold)
    fix_channel = len(pindex1) if p1_p2_same_ratio else 0
    _, pindex2 = get_nonzero_index(projection2, dim='input', counter=1, percentage=percentage, threshold=threshold,
                                   fix_channel=fix_channel)

    # conv11 don't need to be changed.
    # compress conv12: projection1, bias1
    projection1 = torch.index_select(projection1, dim=1, index=pindex1)
    conv12.weight = nn.Parameter(projection1.t().view(pindex1.shape[0], ws1[0], 1, 1)) #TODO: check this one.
    conv12.bias = nn.Parameter(torch.index_select(bias1, dim=0, index=pindex1))
    conv12.out_channels = conv12.weight.size()[0]
    # compress batchnorm2
    batchnorm2.weight = nn.Parameter(torch.index_select(bn_weight2, dim=0, index=pindex1))
    batchnorm2.bias = nn.Parameter(torch.index_select(bn_bias2, dim=0, index=pindex1))
    batchnorm2.running_mean = torch.index_select(bn_mean2, dim=0, index=pindex1)
    batchnorm2.running_var = torch.index_select(bn_var2, dim=0, index=pindex1)
    batchnorm2.num_features = len(batchnorm2.weight)
    # compress conv21
    index = torch.repeat_interleave(pindex1, ws2[2] * ws2[3]) * ws2[2] * ws2[3] \
            + torch.tensor(range(0, ws2[2] * ws2[3])).repeat(pindex1.shape[0]).cuda()
    weight2 = torch.index_select(weight2, dim=0, index=index)
    weight2 = torch.index_select(weight2, dim=1, index=pindex2)
    conv21.weight = nn.Parameter(weight2.t().view(pindex2.shape[0], pindex1.shape[0], ws2[2], ws2[3]))
    conv21.out_channels, conv21.in_channels = conv21.weight.size()[:2]
    # compress conv22
    projection2 = torch.index_select(projection2, dim=0, index=pindex2)
    conv22.weight = nn.Parameter(projection2.t().view(-1, pindex2.shape[0], 1, 1))
    conv22.in_channels = conv22.weight.size()[1]
    # bias2 don't need to be changed.


def modify_network(net_current):
    args = net_current.args
    modules = []
    for module_cur in net_current.modules():
        if isinstance(module_cur, wide_basic):
            modules.append(module_cur)
    for module_cur in modules:

        # get initialization values for the ResBlock to be compressed
        weight1 = module_cur.state_dict()['conv1.weight']
        bias1 = module_cur.state_dict()['conv1.bias']
        weight2 = module_cur.state_dict()['conv2.weight']
        bias2 = module_cur.state_dict()['conv2.bias']
        if args.init_method.find('disturbance') >= 0:
            weight1, projection1 = init_weight_proj(weight1, init_method=args.init_method, d=0, s=0.05)
            weight2, projection2 = init_weight_proj(weight2, init_method=args.init_method, d=1, s=0.05)
        else:
            weight1, projection1 = init_weight_proj(weight1, init_method=args.init_method)
            weight2, projection2 = init_weight_proj(weight2, init_method=args.init_method)
        # modify submodules in the ResBlock
        modify_submodules(module_cur)
        # set ResBlock module params
        params = edict({'weight1': weight1, 'projection1': projection1, 'bias1': bias1,
                        'weight2': weight2, 'projection2': projection2, 'bias2': bias2})
        set_module_param(module_cur, params)


def make_model(args, ckp):
    """
    base_module is the compressed module
    """
    current_module = import_module('model.' + args.base.lower())
    class Hinge(getattr(current_module, args.base)):

        def __init__(self, args):
            self.args = args[0]
            self.ckp = args[1]
            super(Hinge, self).__init__(self.args)

            # traning phase of network compression
            if self.args.model.lower().find('hinge') >= 0 and not self.args.test_only and not self.args.load:
                self.load(self.args, strict=True)

            if self.args.data_train.find('CIFAR') >= 0:
                self.input_dim = (3, 32, 32)
            elif self.args.data_train.find('Tiny_ImageNet') >= 0:
                self.input_dim = (3, 64, 64)
            else:
                self.input_dim = (3, 224, 224)
            self.flops, self.params = get_model_complexity_info(self, self.input_dim, print_per_layer_stat=False)
            
            self.register_buffer('running_grad_ratio', None)

            modify_network(self)

        def find_modules(self):
            return [m for m in self.modules() if isinstance(m, wide_basic)]

        def sparse_param(self, module):
            param1 = module.state_dict(keep_vars=True)['conv1.1.weight']
            param2 = module.state_dict(keep_vars=True)['conv2.1.weight']
            return param1, param2

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
            # a = 0
            for l, m in enumerate(modules):
                projection1, projection2 = self.sparse_param(m)
                p1_grad.append(torch.sum(torch.norm(projection1.grad.squeeze().t(), dim=0)))
                p2_grad.append(torch.sum(torch.norm(projection2.grad.squeeze().t(), dim=1)))
                # a += projection1.grad.squeeze().t().shape[1]
            # print(a)
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
                reg = regularization * lr
                # n1 = torch.norm(param1.squeeze().t(), p=2, dim=0)
                # n2 = torch.norm(param2.squeeze().t(), p=2, dim=1)
                # scale1 = torch.max(1 - reg / (n1 + eps), torch.zeros_like(n1, device=n1.device))
                # scale2 = torch.max(1 - reg / (n2 + eps), torch.zeros_like(n2, device=n2.device))
                # scale1 = scale1.repeat(param1.shape[1], 1)
                # scale2 = scale2.repeat(param2.shape[0], 1).t()
                scale1 = self.calc_sparsity_solution(param1, 0, reg, self.args.sparsity_regularizer)
                scale2 = self.calc_sparsity_solution(param2, 1, reg, self.args.sparsity_regularizer)

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
                # if batch % 100 == 0 and i == 0:
                #     with print_array_on_one_line():
                #         print('Norm1: \n{}'.format((torch.sum(projection1.squeeze().t() ** 2, dim=0) ** (q / 2)).detach().cpu().numpy()))
                #         print('Norm2: \n{}'.format((torch.sum(projection2.squeeze().t() ** 2, dim=1) ** (q / 2)).detach().cpu().numpy()))
                loss_proj1.append(torch.sum(torch.sum(projection1.squeeze().t() ** 2, dim=0) ** (q / 2)) ** (1 / q))
                loss_proj2.append(torch.sum(torch.sum(projection2.squeeze().t() ** 2, dim=1) ** (q / 2)) ** (1 / q))

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
            # loss_proj1 = torch.sum((torch.sum(projection1 ** 2, dim=0) ** q))
            # loss_proj2 = torch.sum((torch.sum(projection2 ** 2, dim=1) ** q))
            if self.args.optimizer == 'SGD':
                lossp1 *= lambda_factor
                lossp2 *= lambda_factor
            return lossp1, lossp2

        def compress(self, **kwargs):
            for module_cur in self.find_modules():
                compress_module_param(module_cur, self.args.remain_percentage, self.args.threshold, self.args.p1_p2_same_ratio)

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
                        'Wide ResBlock{:<2} Layer{}: channels -> orginal={:<2}, compress={:<2}, ratio={:2.4f};'
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

        def merge_conv(self):
            # used after finetuning to save the merged model
            for m in self.find_modules():
                conv = m._modules['conv1']
                if isinstance(conv, nn.Sequential):
                    ws = conv._modules['0'].weight.size()
                    ps = conv._modules['1'].weight.size()
                    weight = conv._modules['0'].weight.data.view(ws[0], -1).t()
                    projection = conv._modules['1'].weight.data.squeeze().t()
                    # need to deal with the bias
                    bias = conv._modules['1'].bias.data
                    weight = torch.mm(weight, projection).t().view(ps[0], ws[1], ws[2], ws[3])
                    m._modules['conv1'] = conv._modules['0']
                    m._modules['conv1'].weight = nn.Parameter(weight)
                    m._modules['conv1'].bias = nn.Parameter(bias)
                    m._modules['conv1'].out_channels = ps[0]

        def split_conv(self, state_dict):
            for m in self.find_modules():
                m._modules['conv1'].bias = None
                conv1 = [m._modules['conv1'], nn.Conv2d(10, 10, 1, bias=True)]
                m._modules['conv1'] = nn.Sequential(*conv1)
            self.load_state_dict(state_dict, strict=False)

        def set_channels(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.out_channels, m.in_channels = m.weight.size()[:2]
                elif isinstance(m, nn.BatchNorm2d):
                    m.num_features = m.weight.size()[0]
                # else:
                #     raise NotImplementedError('Channel setting is not implemented for {}'.format(m.__class__.__name__))

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

    return Hinge((args, ckp))