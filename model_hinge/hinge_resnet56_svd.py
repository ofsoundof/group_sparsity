"""
Group Sparsity: The Hinge Between Filter Pruning and Decomposition forNetwork Compression
This module implement the proposed Hinge method to ResNet20 and ResNet56.
"""
__author__ = 'Yawei Li'
import torch
import torch.nn as nn
import os
import math
from easydict import EasyDict as edict
from importlib import import_module
from model.resnet import ResBlock
from model_hinge.hinge_utility import init_weight_proj, get_nonzero_index, plot_figure, plot_per_layer_compression_ratio
from model.in_use.flops_counter import get_model_complexity_info
from model_hinge.hinge_resnet56 import modify_submodules, set_module_param
from IPython import embed


########################################################################################################################
# used to edit the modules and parameters during the PG optimization and continuing training stage.
########################################################################################################################

def get_compress_idx(module, percentage, threshold, p1_p2_same_ratio):
    body = module._modules['body']
    conv12 = body._modules['0']._modules['1']
    conv22 = body._modules['3']._modules['1']
    projection1 = conv12.weight.data.squeeze().t()
    projection2 = conv22.weight.data.squeeze().t()
    norm1, pindex1 = get_nonzero_index(projection1, dim='input', counter=1, percentage=percentage, threshold=threshold)
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
# Compress module parameters
# ========
def compress_module_param(module, percentage, threshold, p1_p2_same_ratio):
    body = module._modules['body']
    conv11 = body._modules['0']._modules['0']
    conv12 = body._modules['0']._modules['1']
    conv21 = body._modules['3']._modules['0']
    conv22 = body._modules['3']._modules['1']

    ws1 = conv11.weight.shape
    weight1 = conv11.weight.data.view(ws1[0], -1).t()
    projection1 = conv12.weight.data.squeeze().t()

    ws2 = conv21.weight.shape
    weight2 = conv21.weight.data.view(ws2[0], -1).t()
    projection2 = conv22.weight.data.squeeze().t()

    _, pindex1 = get_nonzero_index(projection1, dim='input', counter=1, percentage=percentage, threshold=threshold)
    fix_channel = len(pindex1) if p1_p2_same_ratio else 0
    _, pindex2 = get_nonzero_index(projection2, dim='input', counter=1, percentage=percentage, threshold=threshold,
                                   fix_channel=fix_channel)

    # compress conv11.
    weight1 = torch.index_select(weight1, dim=1, index=pindex1)
    conv11.weight = nn.Parameter(weight1.t().view(pindex1.shape[0], ws1[1], ws1[2], ws1[3]))
    conv11.out_channels, conv11.in_channels = conv11.weight.size()[:2]
    # compress conv12: projection1, bias1
    projection1 = torch.index_select(projection1, dim=0, index=pindex1)
    conv12.weight = nn.Parameter(projection1.t().view(ws1[0], pindex1.shape[0], 1, 1))
    conv12.out_channels, conv12.in_channels = conv12.weight.size()[:2]

    # compress conv21
    weight2 = torch.index_select(weight2, dim=1, index=pindex2)
    conv21.weight = nn.Parameter(weight2.t().view(pindex2.shape[0], ws2[1], ws2[2], ws2[3]))
    conv21.out_channels, conv21.in_channels = conv21.weight.size()[:2]
    # compress conv22
    projection2 = torch.index_select(projection2, dim=0, index=pindex2)
    conv22.weight = nn.Parameter(projection2.t().view(ws2[0], pindex2.shape[0], 1, 1))
    conv22.out_channels, conv22.in_channels = conv22.weight.size()[:2]


def modify_network(net_current):
    args = net_current.args
    modules = []
    for module_cur in net_current.modules():
        if isinstance(module_cur, ResBlock):
            modules.append(module_cur)
    for module_cur in modules:

        # get initialization values for the ResBlock to be compressed
        weight1 = module_cur.state_dict()['body.0.weight']
        weight2 = module_cur.state_dict()['body.3.weight']
        if args.init_method.find('disturbance') >= 0:
            weight1, projection1 = init_weight_proj(weight1, init_method=args.init_method, d=1, s=0.05)
            weight2, projection2 = init_weight_proj(weight2, init_method=args.init_method, d=1, s=0.05)
        else:
            weight1, projection1 = init_weight_proj(weight1, init_method=args.init_method)
            weight2, projection2 = init_weight_proj(weight2, init_method=args.init_method)
        # modify submodules in the ResBlock
        modify_submodules(module_cur)
        # set ResBlock module params
        params = edict({'weight1': weight1, 'projection1': projection1, 'bias1': None,
                        'weight2': weight2, 'projection2': projection2, 'bias2': None})
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

            # traning or loading phase of network compression
            if self.args.model.lower().find('hinge') >= 0 and not self.args.test_only:
                self.load(self.args, strict=True)

            if self.args.data_train.find('CIFAR') >= 0:
                self.input_dim = (3, 32, 32)
            elif self.args.data_train.find('Tiny_ImageNet') >= 0:
                self.input_dim = (3, 64, 64)
            else:
                self.input_dim = (3, 224, 224)
            self.flops, self.params = get_model_complexity_info(self, self.input_dim, print_per_layer_stat=False)

            # if self.args.model.lower().find('resnet') >= 0:
            self.register_buffer('running_grad_ratio', None)

            modify_network(self)

            if self.args.model.lower().find('hinge') >= 0 and not self.args.test_only and self.args.layer_balancing:
                # need to calculate the layer-balancing regularizer during both training and loading phase.
                self.layer_reg = self.calc_regularization()
                for l, reg in enumerate(self.layer_reg):
                    print('ResBlock{:<2}, layer1 / layer2: {:<2.4f} / {:<2.4f}'.format(l + 1, reg[0], reg[1]))

        def find_modules(self):
            return [m for m in self.modules() if isinstance(m, ResBlock)]

        def sparse_param(self, module):
            # embed()
            param1 = module.state_dict(keep_vars=True)['body.0.1.weight']
            param2 = module.state_dict(keep_vars=True)['body.3.1.weight']
            return param1, param2

        def calc_regularization(self):
            layer_reg = []
            modules = self.find_modules()
            # embed()
            for m in modules:
                param1, param2 = self.sparse_param(m)
                mean1 = torch.norm(param1.squeeze().t(), p=2, dim=1).mean().detach().cpu().numpy()
                mean2 = torch.norm(param2.squeeze().t(), p=2, dim=1).mean().detach().cpu().numpy()
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
                p1_grad.append(torch.sum(torch.norm(projection1.grad.squeeze().t(), dim=1)))
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

                scale1 = self.calc_sparsity_solution(param1, 1, reg1, self.args.sparsity_regularizer)
                scale2 = self.calc_sparsity_solution(param2, 1, reg2, self.args.sparsity_regularizer)

                # reg = regularization * lr
                # # n1 = torch.norm(param1.squeeze().t(), p=2, dim=0)
                # # n2 = torch.norm(param2.squeeze().t(), p=2, dim=1)
                # # scale1 = torch.max(1 - reg / (n1 + eps), torch.zeros_like(n1, device=n1.device))
                # # scale2 = torch.max(1 - reg / (n2 + eps), torch.zeros_like(n2, device=n2.device))
                # # scale1 = scale1.repeat(param1.shape[1], 1)
                # # scale2 = scale2.repeat(param2.shape[0], 1).t()
                # scale2 = self.calc_sparsity_solution(param2, 1, reg, self.args.sparsity_regularizer)
                # if self.args.p1_p2_regularization == 'proximal':
                #     reg = reg * self.running_grad_ratio
                # scale1 = self.calc_sparsity_solution(param1, 1, reg, self.args.sparsity_regularizer)

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
                loss_proj1.append(torch.sum(torch.sum(projection1.squeeze().t() ** 2, dim=1) ** (q / 2)) ** (1 / q))
                loss_proj2.append(torch.sum(torch.sum(projection2.squeeze().t() ** 2, dim=1) ** (q / 2)) ** (1 / q))

                if not converging and (batch + 1) % 300 == 0:
                    path = os.path.join(self.args.dir_save, self.args.save, 'ResBlock{}_norm_distribution'.format(l))
                    if not os.path.exists(path):
                        os.makedirs(path)
                    filename1 = os.path.join(path, 'P1_Epoch{}_Batch{}'.format(current_epoch, batch + 1))
                    filename2 = os.path.join(path, 'P2_Epoch{}_Batch{}'.format(current_epoch, batch + 1))
                    plot_figure(projection1.squeeze(), l, filename1)
                    plot_figure(projection2.squeeze(), l, filename2)
            lossp1 = sum(loss_proj1) / len(loss_proj1)
            lossp2 = sum(loss_proj2) / len(loss_proj2)
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
                # elif isinstance(m, nn.BatchNorm2d):
                #     m.num_features = m.weight.size()[0]
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

