"""
Group Sparsity: The Hinge Between Filter Pruning and Decomposition forNetwork Compression
This module implement the proposed Hinge method to DenseNet.
"""
__author__ = 'Yawei Li'
import torch
import torch.nn as nn
import os
import math
from easydict import EasyDict as edict
from model.densenet import Dense, Transition, DenseNet
from model_hinge.hinge_utility import get_nonzero_index, plot_figure, plot_per_layer_compression_ratio
from model.in_use.flops_counter import get_model_complexity_info
from model_hinge.hinge_densenet import modify_network
from IPython import embed


########################################################################################################################
# used to edit the modules and parameters during the PG optimization and continuing training stage.
########################################################################################################################

# ========
# Compute pindex for each module first.
# ========
def compute_pindex(modules, percentage, threshold):
    network_pindex_decom = []
    network_norm_decom = []
    network_pindex_prune = []
    network_norm_prune = []
    for m in modules:
        if isinstance(m, Dense):
            conv2 = m._modules['body']._modules['3']
        elif isinstance(m, Transition):
            conv2 = m._modules['3']
        else:
            raise NotImplementedError('Do not need to prune the layer ' + m.__class__.__name__)
        weight = conv2.weight.squeeze().t()
        norm_decom, pindex_decom = get_nonzero_index(weight, dim='input', counter=1, percentage=percentage, threshold=threshold)
        norm_prune, pindex_prune = get_nonzero_index(weight, dim='output', counter=1, percentage=percentage, threshold=threshold)
        network_pindex_decom.append(pindex_decom)
        network_norm_decom.append(norm_decom)
        network_pindex_prune.append(pindex_prune)
        network_norm_prune.append(norm_prune)
    return network_pindex_decom, network_pindex_prune, network_norm_decom, network_norm_prune


def get_compress_idx(l, network_pindex_decom, network_norm_decom, network_pindex_prune, network_norm_prune):
    pindex_decom = network_pindex_decom[l]
    norm_decom = network_norm_decom[l]
    # norm_remain_decom = norm_decom[pindex_decom]
    pindex_prune = network_pindex_prune[l]
    norm_prune = network_norm_prune[l]
    # norm_remain_prune = norm_prune[pindex_prune]

    def _get_compress_statistics(norm, pindex):
        remain_norm = norm[pindex]
        channels = norm.shape[0]
        remain_channels = remain_norm.shape[0]
        remain_norm = remain_norm.detach().cpu()
        stat_channel = [channels, channels - remain_channels, (channels - remain_channels) / channels]
        stat_remain_norm = [remain_norm.max(), remain_norm.mean(), remain_norm.min()]
        return edict({'stat_channel': stat_channel, 'stat_remain_norm': stat_remain_norm,
                      'remain_norm': remain_norm, 'pindex': pindex})

    return [_get_compress_statistics(norm_decom, pindex_decom), _get_compress_statistics(norm_prune, pindex_prune)]


# ========
# Compress module parameters
# ========
def compress_module_param(module, l, network_pindex_decom, network_pindex_prune):
    # Bias in None. So things becomes easier.
    # get the body
    if isinstance(module, Dense):
        body = module._modules['body']
    elif isinstance(module, Transition):
        body = module
    else:
        raise NotImplementedError('Do not need to compress the layer ' + module.__class__.__name__)
    batchnorm = body._modules['0']
    conv1 = body._modules['2']
    conv2 = body._modules['3']

    # # get the parameters to be compressed
    bn_weight = batchnorm.weight.data
    bn_bias = batchnorm.bias.data
    bn_mean = batchnorm.running_mean.data
    bn_var = batchnorm.running_var.data

    ws1 = conv1.weight.data.shape
    weight1 = conv1.weight.data.view(ws1[0], ws1[1] * ws1[2] * ws1[3]).t()

    ws2 = conv2.weight.data.shape
    weight2 = conv2.weight.data.squeeze().t()

    # get the index for pruning
    if l <= 12:
        index_in = torch.tensor(range(24)).to(device=weight1.device)
        index_in = torch.cat([index_in] + network_pindex_prune[:l], dim=0)
    elif 12 < l <= 25:
        index_in = torch.cat(network_pindex_prune[12:l], dim=0)
    else:
        index_in = torch.cat(network_pindex_prune[25:l], dim=0)

    s_in = index_in.shape[0]
    index_out = network_pindex_prune[l]
    s_out = index_out.shape[0]
    index_inner = network_pindex_decom[l]
    s_inner = index_inner.shape[0]
    # print(pl_in, pl_out)

    # compress batchnorm
    batchnorm.weight = nn.Parameter(torch.index_select(bn_weight, dim=0, index=index_in))
    batchnorm.bias = nn.Parameter(torch.index_select(bn_bias, dim=0, index=index_in))
    batchnorm.running_mean = torch.index_select(bn_mean, dim=0, index=index_in)
    batchnorm.running_var = torch.index_select(bn_var, dim=0, index=index_in)
    batchnorm.num_features = s_in

    # compress conv1
    index = torch.repeat_interleave(index_in, ws1[2] * ws1[3]) * ws1[2] * ws1[3] + \
            torch.tensor(range(0, ws1[2] * ws1[3])).repeat(index_in.shape[0]).to(device=weight1.device)
    weight1 = torch.index_select(weight1, dim=0, index=index) # only need to prune the input channels of conv1
    weight1 = torch.index_select(weight1, dim=1, index=index_inner) # the output channels of conv1 is not prune. Instead prune the output channel of conv2.
    conv1.weight = nn.Parameter(weight1.t().view(s_inner, s_in, ws1[2], ws1[3]))
    conv1.out_channels, conv1.in_channels = s_inner, s_in

    # compress conv2
    weight2 = torch.index_select(weight2, dim=0, index=index_inner)
    weight2 = torch.index_select(weight2, dim=1, index=index_out)
    conv2.weight = nn.Parameter(weight2.t().view(s_out, s_inner, ws2[2], ws2[3]))
    conv2.out_channels, conv2.in_channels = s_out, s_inner


def make_model(inputs):
    """
    base_module is the compressed module
    """
    class Hinge(DenseNet):

        def __init__(self, args):
            self.args = args[0]
            self.ckp = args[1]
            super(Hinge, self).__init__(self.args)

            # traning or loading phase of network pruning
            if self.args.model.lower().find('hinge') >= 0 and not self.args.test_only:
                self.load(strict=True)

            if self.args.data_train.find('CIFAR') >= 0:
                self.input_dim = (3, 32, 32)
            elif self.args.data_train.find('Tiny_ImageNet') >= 0:
                self.input_dim = (3, 64, 64)
            else:
                self.input_dim = (3, 224, 224)
            self.flops, self.params = get_model_complexity_info(self, self.input_dim, print_per_layer_stat=False)

            modify_network(self)

            if self.args.model.lower().find('hinge') >= 0 and not self.args.test_only and self.args.layer_balancing:
                # need to calculate the layer-balancing regularizer during both training and loading phase.
                self.layer_reg = self.calc_regularization()
                for l, reg in enumerate(self.layer_reg):
                    print('DenseBlock {:<2}: {:<2.4f}'.format(l + 1, reg))

        def find_modules(self):
            return [m for m in self.modules() if isinstance(m, Dense) or isinstance(m, Transition)]

        def sparse_param(self, module):
            if isinstance(module, Dense):
                return module.state_dict(keep_vars=True)['body.3.weight']
            elif isinstance(module, Transition):
                return module.state_dict(keep_vars=True)['3.weight']
            else:
                raise NotImplementedError('Do not need to compress the layer ' + module.__class__.__name__)

        def calc_sparsity_solution(self, param, dim, reg, sparsity_reg):
            eps = 1e-8
            n = torch.norm(param.squeeze().t(), p=2, dim=dim) # Be careful here. Whether to use dim=0 or dim=1.
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
            if dim == 0:
                scale = scale.repeat(param.shape[1], 1)
            else:
                scale = scale.repeat(param.shape[0], 1).t()
            return scale

        def proximal_operator(self, lr, batch, regularization):
            modules = self.find_modules()
            for l in range(len(modules)):
                param = self.sparse_param(modules[l])

                if batch == 100 and l == 0:
                    print('Regularization is {}'.format(regularization))
                if self.args.layer_balancing:
                    reg = self.args.regularization_factor * self.layer_reg[l - 1] * lr
                else:
                    reg = self.args.regularization_factor * lr

                scale = self.calc_sparsity_solution(param.data, 1, reg, self.args.sparsity_regularizer)
                param.data = torch.mul(scale, param.data.squeeze().t()).t().unsqueeze(2).unsqueeze(3)
                scale = self.calc_sparsity_solution(param.data, 0, reg, self.args.sparsity_regularizer)
                param.data = torch.mul(scale, param.data.squeeze().t()).t().unsqueeze(2).unsqueeze(3)

        def compute_loss(self, batch, current_epoch, converging):
            """
            loss = ||Y - Yc||^2 + lambda * (||A_1||_{2,1} + ||A_2 ^T||_{2,1})
            """
            modules = self.find_modules()
            lambda_factor = self.args.regularization_factor
            q = self.args.q
            loss_proj_decom = []
            loss_proj_prune = []
            for l in range(len(modules)):
                projection = self.sparse_param(modules[l])
                loss_proj_decom.append(torch.sum(torch.sum(projection.squeeze().t() ** 2, dim=1) ** (q / 2)) ** (1 / q))
                loss_proj_prune.append(torch.sum(torch.sum(projection.squeeze().t() ** 2, dim=0) ** (q / 2)) ** (1 / q))
                # save the norm distribution during the optimization phase
                if not converging and (batch + 1) % 300 == 0:
                    path = os.path.join(self.args.dir_save, self.args.save, 'Filter{}_norm_distribution'.format(l))
                    if not os.path.exists(path):
                        os.makedirs(path)
                    filename = os.path.join(path, 'Epoch{}_Batch{}'.format(current_epoch, batch + 1))
                    plot_figure(projection.squeeze(), l, filename)
            lossp_decom = sum(loss_proj_decom)
            lossp_prune = sum(loss_proj_prune)
            if self.args.optimizer == 'SGD':
                lossp_decom *= lambda_factor
                lossp_prune *= lambda_factor
            return lossp_decom, lossp_prune

        def calc_regularization(self):
            layer_reg = []
            modules = self.find_modules()
            for l in range(len(modules)):
                projection = self.sparse_param(modules[l])
                layer_reg.append(torch.norm(projection.squeeze().t(), p=2, dim=1).mean().detach().cpu().numpy() / 2 +
                                 torch.norm(projection.squeeze().t(), p=2, dim=0).mean().detach().cpu().numpy() / 2)
            return layer_reg

        def compress_tail(self):
            # get the layers
            batchnorm_final = self._modules['features']._modules['6']
            classifier = self._modules['classifier']

            # get the pruning index
            pindex = torch.cat(self.network_pindex_prune[25:], dim=0)
            pl = pindex.shape[0]

            # get the parameters
            bn_weight = batchnorm_final.weight.data
            bn_bias = batchnorm_final.bias.data
            bn_mean = batchnorm_final.running_mean.data
            bn_var = batchnorm_final.running_var.data
            classifier_weight = classifier.weight.data.t()

            # prune the last batchnorm layer
            batchnorm_final.weight = nn.Parameter(torch.index_select(bn_weight, dim=0, index=pindex))
            batchnorm_final.bias = nn.Parameter(torch.index_select(bn_bias, dim=0, index=pindex))
            batchnorm_final.running_mean = torch.index_select(bn_mean, dim=0, index=pindex)
            batchnorm_final.running_var = torch.index_select(bn_var, dim=0, index=pindex)
            batchnorm_final.num_features = pl

            # prune the classifier
            classifier.weight = nn.Parameter(torch.index_select(classifier_weight, dim=0, index=pindex).t())
            classifier.in_features = pl

        def compress(self, **kwargs):
            modules = self.find_modules()
            print('Compute the pruning index for all of the DenseBlocks')
            self.network_pindex_decom, self.network_pindex_prune, self.network_norm_decom, self.network_norm_prune\
                = compute_pindex(modules, self.args.remain_percentage, self.args.threshold)
            print('Starting pruning.')
            for l, m in enumerate(modules):
                compress_module_param(m, l, network_pindex_decom=self.network_pindex_decom, network_pindex_prune=self.network_pindex_prune)
            self.compress_tail()

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
                    get_compress_idx(l, self.network_pindex_decom, self.network_norm_decom, self.network_pindex_prune, self.network_norm_prune)
                for i, info_layer in enumerate(info_compress['info_layer{}'.format(l)]):
                    self.ckp.write_log(
                        'DenseBlock{:<2} Layer{}: channels -> orginal={:<2}, compress={:<2}, ratio={:2.4f};'
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
            elif not self.args.load and self.args.pretrained:
                state_dict = torch.load(self.args.pretrained)
            else:
                raise NotImplementedError('Do not need to load {} in this mode'.format(self.__class__.__name__))

            self.load_state_dict(state_dict, strict=strict)

    return Hinge(inputs)
