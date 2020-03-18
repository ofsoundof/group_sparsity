"""
Group Sparsity: The Hinge Between Filter Pruning and Decomposition forNetwork Compression
This module implement the proposed Hinge method to VGG.
"""
__author__ = 'Yawei Li'
import torch
import torch.nn as nn
import os
import math
from easydict import EasyDict as edict
from importlib import import_module
from model.common import BasicBlock
from model_hinge.hinge_utility import init_weight_proj, get_nonzero_index, plot_figure, plot_per_layer_compression_ratio
from model.in_use.flops_counter import get_model_complexity_info
from IPython import embed


########################################################################################################################
# used to edit the modules and parameters during the PG optimization and continuing training stage.
########################################################################################################################

def get_compress_idx(module, percentage, threshold):
    conv12 = module[0][1]
    projection1 = conv12.weight.data.squeeze().t()
    # decomposition
    norm1, pindex1 = get_nonzero_index(projection1, dim='input', counter=1, percentage=percentage, threshold=threshold)
    # pruning
    norm2, pindex2 = get_nonzero_index(projection1, dim='output', counter=1, percentage=percentage, threshold=threshold)
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
    conv1 = [module[0], nn.Conv2d(10, 10, 1, bias=False)]
    module[0] = nn.Sequential(*conv1)
    module.optimization = True


# ========
# Set submodule parameters
# ========
def set_module_param(module, params):
    ws1 = params.weight1.size()
    ps1 = params.projection1.size()

    conv11 = module[0][0]
    conv12 = module[0][1]

    # set conv11
    conv11.in_channels = ws1[1]
    conv11.out_channels = ws1[0]
    conv11.weight.data = params.weight1.data
    conv11.bias = None

    # set conv12
    conv12.in_channels = ps1[1]
    conv12.out_channels = ps1[0]
    conv12.weight.data = params.projection1.data
    if params.bias1 is not None:
        conv12.bias = nn.Parameter(params.bias1)
    # Note that the bias term is added to the second conv
    # Do not need to set batchnorm1, activation, and batchnorm2.
    # embed()


# ========
# Compress module parameters
# ========
def compress_module_param(module, percentage, threshold, index_pre=None, i=0):
    conv11 = module[0][0]
    conv12 = module[0][1]
    batchnorm1 = module[1]

    ws1 = conv11.weight.shape
    weight1 = conv11.weight.data.view(ws1[0], -1).t()
    projection1 = conv12.weight.data.squeeze().t()
    bias1 = conv12.bias.data if conv12.bias is not None else None
    bn_weight1 = batchnorm1.weight.data
    bn_bias1 = batchnorm1.bias.data
    bn_mean1 = batchnorm1.running_mean.data
    bn_var1 = batchnorm1.running_var.data

    pindex1 = get_nonzero_index(projection1, dim='input', counter=1, percentage=percentage, threshold=threshold)[1]
    pindex2 = get_nonzero_index(projection1, dim='output', counter=1, percentage=percentage, threshold=threshold)[1]

    # conv11
    if index_pre is not None:
        index = torch.repeat_interleave(index_pre, ws1[2] * ws1[3]) * ws1[2] * ws1[3] \
                + torch.tensor(range(0, ws1[2] * ws1[3])).repeat(index_pre.shape[0]).cuda()
        weight1 = torch.index_select(weight1, dim=0, index=index)
    weight1 = torch.index_select(weight1, dim=1, index=pindex1)
    conv11.weight = nn.Parameter(weight1.t().view(pindex1.shape[0], -1, ws1[2], ws1[3]))
    conv11.out_channels, conv11.in_channels = conv11.weight.size()[:2]

    # conv12: projection1, bias1
    projection1 = torch.index_select(projection1, dim=0, index=pindex1)
    if i < 11:
        projection1 = torch.index_select(projection1, dim=1, index=pindex2)
        if bias1 is not None:
            conv12.bias = nn.Parameter(torch.index_select(bias1, dim=0, index=pindex2))

        # compress batchnorm1
        batchnorm1.weight = nn.Parameter(torch.index_select(bn_weight1, dim=0, index=pindex2))
        batchnorm1.bias = nn.Parameter(torch.index_select(bn_bias1, dim=0, index=pindex2))
        batchnorm1.running_mean = torch.index_select(bn_mean1, dim=0, index=pindex2)
        batchnorm1.running_var = torch.index_select(bn_var1, dim=0, index=pindex2)
        batchnorm1.num_features = len(batchnorm1.weight)

    conv12.weight = nn.Parameter(projection1.t().view(-1, pindex1.shape[0], 1, 1)) #TODO: check this one.
    conv12.out_channels, conv12.in_channels = conv12.weight.size()[:2]


def modify_network(net_current):
    args = net_current.args
    modules = []
    for module_cur in net_current.modules():
        if isinstance(module_cur, BasicBlock):
            modules.append(module_cur)
    # skip the first BasicBlock that deals with the input images
    for module_cur in modules[1:]:

        # get initialization values for the Block to be compressed
        weight1 = module_cur.state_dict()['0.weight']
        bias1 = module_cur.state_dict()['0.bias']
        if args.init_method.find('disturbance') >= 0:
            weight1, projection1 = init_weight_proj(weight1, init_method=args.init_method, d=0, s=0.05)
        else:
            weight1, projection1 = init_weight_proj(weight1, init_method=args.init_method)
        # modify submodules in the ResBlock
        modify_submodules(module_cur)
        # set ResBlock module params
        params = edict({'weight1': weight1, 'projection1': projection1, 'bias1': bias1})
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

            modify_network(self)

        def find_modules(self):
            return [m for m in self.modules() if isinstance(m, BasicBlock)][1:]

        def sparse_param(self, module):
            param1 = module.state_dict(keep_vars=True)['0.1.weight']
            return param1

        def calc_regularization(self):
            layer_reg = []
            modules = self.find_modules()
            for l, module in enumerate(modules):
                projection1 = self.sparse_param(module)
                layer_reg.append(torch.norm(projection1, p=2, dim=1).mean().detach().cpu().numpy() +
                                 torch.norm(projection1, p=2, dim=0).mean().detach().cpu().numpy())
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

        def proximal_operator(self, lr, batch, regularization):
            modules = self.find_modules()
            for l in range(len(modules)):
                param1 = self.sparse_param(modules[l])
                if batch == 100 and l == 0:
                    print('Regularization is {}'.format(regularization))
                reg = regularization * lr

                scale1 = self.calc_sparsity_solution(param1.data, 1, reg, self.args.sparsity_regularizer)
                param1.data = torch.mul(scale1, param1.data.squeeze().t()).t().view(param1.shape)
                scale2 = self.calc_sparsity_solution(param1.data, 0, reg, self.args.sparsity_regularizer)
                param1.data = torch.mul(scale2, param1.data.squeeze().t()).t().view(param1.shape)

        def compute_loss(self, batch, current_epoch, converging):
            """
            loss = ||Y - Yc||^2 + lambda * (||A_1||_{2,1} + ||A_2 ^T||_{2,1})
            """
            modules = self.find_modules()
            lambda_factor = self.args.regularization_factor
            q = self.args.q
            loss_proj11 = []
            loss_proj12 = []
            for l, m in enumerate(modules):
                projection1 = self.sparse_param(m)
                # if batch % 100 == 0 and i == 0:
                #     with print_array_on_one_line():
                #         print('Norm1: \n{}'.format((torch.sum(projection1.squeeze().t() ** 2, dim=0) ** (q / 2)).detach().cpu().numpy()))
                #         print('Norm2: \n{}'.format((torch.sum(projection2.squeeze().t() ** 2, dim=1) ** (q / 2)).detach().cpu().numpy()))
                loss_proj11.append(torch.sum(torch.sum(projection1.squeeze().t() ** 2, dim=1) ** (q / 2)) ** (1 / q))
                loss_proj12.append(torch.sum(torch.sum(projection1.squeeze().t() ** 2, dim=0) ** (q / 2)) ** (1 / q))
                # embed()
                if not converging and (batch + 1) % 200 == 0:
                    path = os.path.join(self.args.dir_save, self.args.save, 'ResBlock{}_norm_distribution'.format(l))
                    if not os.path.exists(path):
                        os.makedirs(path)
                    filename1 = os.path.join(path, 'Row_Epoch{}_Batch{}'.format(current_epoch, batch + 1))
                    filename2 = os.path.join(path, 'Column_Epoch{}_Batch{}'.format(current_epoch, batch + 1))
                    plot_figure(projection1.squeeze().t(), l, filename1)
                    plot_figure(projection1.squeeze(), l, filename2)
            # print(loss_proj11[6], loss_proj12[6])
            lossp1 = sum(loss_proj11) #/ len(loss_proj11)
            lossp2 = sum(loss_proj12) #/ len(loss_proj12)

            # loss_proj1 = torch.sum((torch.sum(projection1 ** 2, dim=0) ** q))
            # loss_proj2 = torch.sum((torch.sum(projection2 ** 2, dim=1) ** q))
            if self.args.optimizer == 'SGD':
                lossp1 *= lambda_factor
                lossp2 *= lambda_factor
            return lossp1, lossp2

        def index_pre(self, percentage, threshold):
            index = []
            for module_cur in self.find_modules():
                conv12 = module_cur[0][1]
                projection1 = conv12.weight.data.squeeze().t()
                index.append(get_nonzero_index(projection1, dim='output', counter=1, percentage=percentage, threshold=threshold)[1])
            return index

        def compress(self, **kwargs):
            index = [None] + self.index_pre(self.args.remain_percentage, self.args.threshold)
            for i, module_cur in enumerate(self.find_modules()):
                compress_module_param(module_cur, self.args.remain_percentage, self.args.threshold, index[i], i)

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
                    get_compress_idx(m, self.args.remain_percentage, self.args.threshold)
                for i, info_layer in enumerate(info_compress['info_layer{}'.format(l)]):
                    self.ckp.write_log(
                        'ResBlock{:<2} Layer{}: channels -> orginal={:<2}, compress={:<2}, ratio={:2.4f};'
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
                    m.out_channels, m.in_channels = m.weight.size()[:2]
                elif isinstance(m, nn.BatchNorm2d):
                    m.num_features = m.weight.size()[0]
            # linear_layer = self.classifier[0]
            # weight = linear_layer.weight.data
            # linear_layer.in_features = weight.shape[1]
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
