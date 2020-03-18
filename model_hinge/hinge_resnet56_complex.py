"""
This is the module that contains the complex functions of the hinge method during the initial development stage.
The simplified version is contained in hinge_resnet56.py
"""
__author__ = 'Yawei Li'
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from easydict import EasyDict as edict
from importlib import import_module
from model.in_use import utility as dutil
from model.resnet import ResBlock
from model_hinge.hinge_utility import init_weight_proj, make_optimizer, make_scheduler
from model_hinge.hinge_utility import feature_visualize, save_tensor_image, print_array_on_one_line
from model.in_use.flops_counter import get_model_flops
from IPython import embed


########################################################################################################################
# used to edit the modules and parameters during the optimization and fine-tuning step
########################################################################################################################

# ========
# Double ReLU activation function
# ========
class DReLU(nn.Module):
    def __init__(self):
        super(DReLU, self).__init__()
        self.slope_p = nn.Parameter(torch.tensor(1.0))
        self.slope_n = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        out = self.slope_p * F.relu(x) + self.slope_n * F.relu(-x)
        return out

# ========
# Modify submodules
# ========
def modify_submodules(module):
    module._modules['body']._modules['2'] = DReLU()
    conv2 = [nn.Conv2d(10, 10, 3, bias=False), nn.Conv2d(10, 10, 1, bias=False)]
    module._modules['body']._modules['3'] = nn.Sequential(*conv2)
    # or
    # module._modules['body']._modules['5'] = nn.Conv2d(10, 10, 1)
    module.optimization = True

# ========
# Set module parameter
# ========
def set_module_param(module, params, set_type='param'):

    def _param_set_fun(p, attr, v, set_type):
        if set_type == 'tensor':
            setattr(p, attr, v)
        elif set_type == 'param':
            if v is not None:
                setattr(p, attr, nn.Parameter(v))
            else:
                setattr(p, attr, v)
        elif set_type == 'data':
            if v is not None:
                getattr(p, attr).data = v
            else:
                setattr(p, attr, v)
        else:
            raise NotImplementedError

    _param_set_fun(module, 'weight1', params.weight1.data, set_type=set_type)
    _param_set_fun(module, 'projection1', params.projection1.data, set_type = set_type)
    _param_set_fun(module, 'bias1', params.bias1.data if params.bias1 is not None else params.bias1, set_type = set_type)
    _param_set_fun(module, 'bn_weight1', params.bn_weight1.data, set_type = set_type)
    _param_set_fun(module, 'bn_bias1', params.bn_bias1.data, set_type = set_type)
    module.register_buffer('bn_mean1', params.bn_mean1)  # TODO: need to check this?
    module.register_buffer('bn_var1', params.bn_var1)

    _param_set_fun(module, 'weight2', params.weight2.data, set_type=set_type)
    _param_set_fun(module, 'projection2', params.projection2.data, set_type=set_type)
    _param_set_fun(module, 'bias2', params.bias2.data if params.bias2 is not None else params.bias2, set_type=set_type)
    _param_set_fun(module, 'bn_weight2', params.bn_weight2.data, set_type=set_type)
    _param_set_fun(module, 'bn_bias2', params.bn_bias2.data, set_type=set_type)
    module.register_buffer('bn_mean2', params.bn_mean2)  # TODO: need to check this?
    module.register_buffer('bn_var2', params.bn_var2)

    # module.weight1 = nn.Parameter(params.weight1)
    # module.projection1 = nn.Parameter(params.projection1)
    # module.bias1 = nn.Parameter(params.bias1) if params.bias1 is not None else None
    # module.bn_weight1 = nn.Parameter(params.bn_weight1)
    # module.bn_bias1 = nn.Parameter(params.bn_bias1)
    # module.register_buffer('bn_mean1', params.bn_mean1) #TODO: need to check this?
    # module.register_buffer('bn_var1', params.bn_var1)
    #
    # module.weight2 = nn.Parameter(params.weight2)
    # module.projection2 = nn.Parameter(params.projection2)
    # module.bias2 = nn.Parameter(params.bias2) if params.bias2 is not None else None
    # module.bn_weight2 = nn.Parameter(params.bn_weight2)
    # module.bn_bias2 = nn.Parameter(params.bn_bias2)
    # module.register_buffer('bn_mean2', params.bn_mean2)
    # module.register_buffer('bn_var2', params.bn_var2)

    # module.slope_p = nn.Parameter(params['slope_p'])
    # module.slope_n = nn.Parameter(params['slope_n'])


# ========
# Remove module parameter for finetuning
# ========
def del_module_param(module):
    del module.weight1, module.projection1, module.bias1, \
        module.bn_weight1, module.bn_bias1, module.bn_mean1, module.bn_var1, \
        module.weight2, module.projection2, module.bias2, \
        module.bn_weight2, module.bn_bias2, module.bn_mean2, module.bn_var2


global_step = 0


# ========
# Get module parameters
# ========
def get_module_param(module, print_flag=False):

    def _print_projection_norm(x, dim='output'):
        n = torch.norm(x.squeeze().t(), p=2, dim=0 if dim == 'output' else 1)
        # with print_array_on_one_line():
        #     print('Norm of Projection1: {}.'.format(n.detach().cpu().numpy()))

    weight1 = module.weight1.clone()
    projection1 = module.projection1.clone()
    bias1 = module.bias1.clone() if module.bias1 is not None else module.bias1
    bn_weight1 = module.bn_weight1.clone()
    bn_bias1 = module.bn_bias1.clone()
    bn_mean1 = module.bn_mean1
    bn_var1 = module.bn_var1

    weight2 = module.weight2.clone()
    projection2 = module.projection2.clone()
    bias2 = module.bias2.clone() if module.bias2 is not None else module.bias2
    bn_weight2 = module.bn_weight2.clone()
    bn_bias2 = module.bn_bias2.clone()
    bn_mean2 = module.bn_mean2
    bn_var2 = module.bn_var2

    if print_flag:
        _print_projection_norm(projection1, dim='output')
        _print_projection_norm(projection2, dim='input')

    params = edict({'weight1': weight1, 'bias1': bias1, 'projection1': projection1,
                    'bn_weight1': bn_weight1, 'bn_bias1': bn_bias1, 'bn_mean1': bn_mean1, 'bn_var1': bn_var1,
                    'weight2': weight2, 'bias2': bias2, 'projection2': projection2,
                    'bn_weight2': bn_weight2, 'bn_bias2': bn_bias2, 'bn_mean2': bn_mean2, 'bn_var2': bn_var2})
    return params

# ========
# Prune module parameters
# ========
def prune_module_param(module):

    ws1 = module.weight1.shape
    weight1 = module.weight1#.view(ws1[0], -1).t()
    projection1 = module.projection1.squeeze().t()
    bias1 = module.bias1
    bn_weight1 = module.bn_weight1
    bn_bias1 = module.bn_bias1
    bn_mean1 = module.bn_mean1
    bn_var1 = module.bn_var1

    ws2 = module.weight2.shape
    weight2 = module.weight2.view(ws1[0], -1).t()
    projection2 = module.projection2.squeeze().t()
    bias2 = module.bias2
    bn_weight2 = module.bn_weight2
    bn_bias2 = module.bn_bias2
    bn_mean2 = module.bn_mean2
    bn_var2 = module.bn_var2

    def _get_nonzero_index(x, dim='output', counter=1):
        n = torch.norm(x, p=2, dim=0 if dim == 'output' else 1)
        threshold = n.topk(int(len(n) * 0.5)).values[-1]
        f = n >= min(threshold, 5e-3) #> n.max() / 10
        # global resblock_counter, global_step
        # writer.add_histogram('ResBlock{}_index{}'.format(resblock_counter, counter), n, global_step=global_step, bins=10)
        # global_step += 1
        # print(x[1, :5].detach().cpu().numpy())
        # if counter ==1:
            # with print_array_on_one_line():
            #     print('Norm of Projection1: {}.'.format(n.detach().cpu().numpy()))
        # print(f.detach().cpu().numpy())
        return torch.nonzero(f).squeeze(dim=1)

    pindex1 = _get_nonzero_index(projection1, dim='output', counter=1)
    pindex2 = _get_nonzero_index(projection2, dim='input', counter=1)
    # with print_array_on_one_line():
    #     print('Index of Projection1: {}'.format(pindex1.detach().cpu().numpy()))
    #     print('Index of Projection2: {}'.format(pindex2.detach().cpu().numpy()))
    projection1 = torch.index_select(projection1, dim=1, index=pindex1)
    projection1 = projection1.t().view(pindex1.shape[0], ws1[0], 1, 1) #TODO: check this one.
    bn_weight1 = torch.index_select(bn_weight1, dim=0, index=pindex1)
    bn_bias1 = torch.index_select(bn_bias1, dim=0, index=pindex1)
    bn_mean1 = torch.index_select(bn_mean1, dim=0, index=pindex1)
    bn_var1 = torch.index_select(bn_var1, dim=0, index=pindex1)

    index = torch.repeat_interleave(pindex1, ws2[2] * ws2[3]) * ws2[2] * ws2[3] \
            + torch.tensor(range(0, ws2[2] * ws2[3])).repeat(pindex1.shape[0]).cuda()
    weight2 = torch.index_select(weight2, dim=0, index=index)
    weight2 = torch.index_select(weight2, dim=1, index=pindex2)
    weight2 = weight2.t().view(pindex2.shape[0], pindex1.shape[0], ws2[2], ws2[3])
    bias2 = torch.index_select(bias2, dim=0, index=pindex2) if bias2 is not None else None
    projection2 = torch.index_select(projection2, dim=0, index=pindex2)
    projection2 = projection2.t().view(ws2[1], pindex2.shape[0], 1, 1)

    params = edict({'weight1': weight1, 'bias1': bias1, 'projection1': projection1,
                    'bn_weight1': bn_weight1, 'bn_bias1': bn_bias1, 'bn_mean1': bn_mean1, 'bn_var1': bn_var1,
                    'weight2': weight2, 'bias2': bias2, 'projection2': projection2,
                    'bn_weight2': bn_weight2.clone(), 'bn_bias2': bn_bias2.clone(),
                    'bn_mean2': bn_mean2.clone(), 'bn_var2': bn_var2.clone()})
    return params


# ========
# Set submodule parameters
# ========
def set_submodule_param(module, params, set_type):

    ws = params.weight1.size()
    prs = params.projection1.size()
    weight1 = params.weight1.view(ws[0], -1).t()
    projection1 = params.projection1.squeeze().t()
    weight1 = torch.mm(weight1, projection1)
    weight1 = weight1.t().view(prs[0], ws[1], ws[2], ws[3])
    bias1 = torch.mm(params.bias1.unsqueeze(dim=0), projection1).squeeze() if params.bias1 is not None else None

    ws1 = weight1.size()
    ws2 = params.weight2.size()
    prs = params.projection2.size()

    def _param_set_fun(p, attr, v, set_type):
        """
        :param p: the input object
        :param attr: the attribute
        :param v: the value of the attribute
        :param set_type: tensor - torch.Tensor, param - nn.Parameter, data - data attribute
        """
        if set_type == 'tensor':
            setattr(p, attr, v)
        elif set_type == 'param':
            if v is not None:
                setattr(p, attr, nn.Parameter(v))
            else:
                setattr(p, attr, v)
        elif set_type == 'data':
            if v is not None:
                getattr(p, attr).data = v
            else:
                setattr(p, attr, v)
        else:
            raise NotImplementedError

    # set conv1
    body = module._modules['body']
    body._modules['0'].in_channels = ws1[1]
    body._modules['0'].out_channels = ws1[0]
    body._modules['0'].kernel_size = [ws1[2], ws1[3]]
    _param_set_fun(body._modules['0'], 'weight', weight1, set_type)
    _param_set_fun(body._modules['0'], 'bias', bias1, set_type)
    # set batchnorm1
    body._modules['1'].num_features = params.bn_weight1.size()[0]
    _param_set_fun(body._modules['1'], 'weight', params.bn_weight1, set_type)
    _param_set_fun(body._modules['1'], 'bias', params.bn_bias1, set_type)
    body._modules['1'].register_buffer('running_mean', params.bn_mean1)
    body._modules['1'].register_buffer('running_var', params.bn_var1)
    # set activation
    # body._modules['2'].slope_p = module.slope_p
    # body._modules['2'].slope_n = module.slope_n
    # set conv2
    body._modules['3']._modules['0'].in_channels = ws2[1]
    body._modules['3']._modules['0'].out_channels = ws2[0]
    body._modules['3']._modules['0'].kernel_size = [ws2[2], ws2[3]]
    body._modules['3']._modules['0'].padding = [ws2[2] // 2, ws2[2] // 2]
    _param_set_fun(body._modules['3']._modules['0'], 'weight', params.weight2, set_type)
    _param_set_fun(body._modules['3']._modules['0'], 'bias', params.bias2, set_type)
    body._modules['3']._modules['1'].in_channels = prs[1]
    body._modules['3']._modules['1'].out_channels = prs[0]
    body._modules['3']._modules['1'].kernel_size = [prs[2], prs[3]]
    _param_set_fun(body._modules['3']._modules['1'], 'weight', params.projection2, set_type)
    body._modules['3']._modules['1'].bias = None
    # set batchnorm2
    body._modules['4'].num_features = params.bn_weight2.size()[0]
    _param_set_fun(body._modules['4'], 'weight', params.bn_weight2, set_type)
    _param_set_fun(body._modules['4'], 'bias', params.bn_bias2, set_type)
    body._modules['4'].register_buffer('running_mean', params.bn_mean2)
    body._modules['4'].register_buffer('running_var', params.bn_var2)


# ========
# Delete submodule parameters before setting them
# ========
def del_submodule_param(module):
    del module._modules['body']._modules['0'].weight
    del module._modules['body']._modules['0'].bias
    del module._modules['body']._modules['1'].weight
    del module._modules['body']._modules['1'].bias
    del module._modules['body']._modules['1'].running_mean
    del module._modules['body']._modules['1'].running_var
    # del module._modules['body']._modules['2'].slope_p
    # del module._modules['body']._modules['2'].slope_n
    del module._modules['body']._modules['3']._modules['0'].weight
    del module._modules['body']._modules['3']._modules['0'].bias
    del module._modules['body']._modules['3']._modules['1'].weight
    del module._modules['body']._modules['3']._modules['1'].bias
    del module._modules['body']._modules['4'].weight
    del module._modules['body']._modules['4'].bias
    del module._modules['body']._modules['4'].running_mean
    del module._modules['body']._modules['4'].running_var


# ========
# Prepare module for optimization or finetuning
# ========
def prune(module):
    prune_procedure = module.prune_procedure
    optimization = module.optimization
    if optimization: #optimization
        if prune_procedure == 'complete':
            params = prune_module_param(module)
            set_module_param(module, params)
            params = get_module_param(module)
        elif prune_procedure == 'undergoing':
            params = prune_module_param(module)
        elif prune_procedure == 'final':
            params = get_module_param(module, print_flag=True)
        else:
            raise NotImplementedError
        set_submodule_param(module, params, set_type='tensor')
    else: #finetune
        if prune_procedure == 'complete':
            params = get_module_param(module, print_flag=True)
        elif prune_procedure == 'undergoing' or prune_procedure == 'final':
            params = prune_module_param(module)
        else:
            raise NotImplementedError
        set_submodule_param(module, params, set_type='param')
        del_module_param(module)


# ========
# module editting hook
# ========
def module_edit_hook(module, input):
    prune(module)


########################################################################################################################
# Add or remove feature map collection hook
########################################################################################################################


# ========
# Add feature map handle
# ========
def add_feature_map_handle(module, store_input=False, store_output=False):

    if store_output and store_input:
        handle = module.register_forward_hook(feature_map_hook)
        module.__storage_handle__ = handle
    elif store_output and not store_input:
        handle = module.register_forward_hook(output_feature_map_hook)
        module.__storage_handle__ = handle
    elif not store_output and store_input:
        handle = module.register_forward_hook(input_feature_map_hook)
        module.__storage_handle__ = handle
    else:
        raise NotImplementedError('Feature maps need to be stored for calling the function ' + add_feature_map_handle.__name__)


# ========
# Remove feature map handle
# ========
def remove_feature_map_handle(module, store_input=False, store_output=False):
    if store_output or store_input:
        module.__storage_handle__.remove()


# ========
# Global variable used to passing the feature maps between different functions
# ========

# used for the feature map of the whole ResBlock
feature_map = {}
# used for the intermediate feature map of the ResBlock submodules
feature_map_inter = {}


def output_feature_map_hook(module, input ,output):
    global feature_map, feature_map_inter
    features = {'output': output}
    if module.__class__.__name__ == 'ResBlock':
        feature_map = features
    else:
        feature_map_inter = features
    # print('{} {}, Data Batch {}, Device {}'.format(module.__class__.__name__, module.__count_layer__, count_data, torch.cuda.current_device()))


def input_feature_map_hook(module, input, output):
    global feature_map, feature_map_inter
    features = {'input': input[0]}
    if module.__class__.__name__ == 'ResBlock':
        feature_map = features
    else:
        feature_map_inter = features
    # print('{} {}, Data Batch {}, Device {}'.format(module.__class__.__name__, module.__count_layer__, count_data, torch.cuda.current_device()))


def feature_map_hook(module, input, output):
    global feature_map, feature_map_inter
    features = {'input': input[0], 'output': output}
    if module.__class__.__name__ == 'ResBlock':
        feature_map = features
    else:
        feature_map_inter = features
    # print('{} {}, Data Batch {}, Device {}'.format(module.__class__.__name__, module.__count_layer__, count_data, torch.cuda.current_device()))


########################################################################################################################
# used to optimize the compression configuration
########################################################################################################################

# ========
# Loss function for optimization
# ========
def compute_loss(feature_ori, feature_comp, projection1, projection2, prediction, label, lambda_factor=1.0, q=1):
    """
    loss = ||Y - Yc||^2 + lambda * (||A_1||_{2,1} + ||A_2 ^T||_{2,1})
    """
    loss_function = nn.MSELoss()
    projection1 = projection1.squeeze().t()
    projection2 = projection2.squeeze().t()

    loss_function_ce = nn.CrossEntropyLoss()
    # embed()
    loss_acc = loss_function_ce(prediction, label.cuda()) * 10

    feat_norm_comp = torch.norm(feature_comp) ** 2
    feat_norm_ori = torch.norm(feature_ori) ** 2
    loss_feat = loss_function(feature_comp, feature_ori) * 0.1
    # q = 0.5
    loss_proj1 = torch.sum(torch.sum(projection1 ** 2, dim=0) ** (q/2)) ** (1/q) * lambda_factor * 0.1
    loss_proj2 = torch.sum(torch.sum(projection2 ** 2, dim=1) ** (q/2)) ** (1/q) * lambda_factor * 0.1
    # loss_proj1 = torch.sum((torch.sum(projection1 ** 2, dim=0) ** q))
    # loss_proj2 = torch.sum((torch.sum(projection2 ** 2, dim=1) ** q))

    loss =  loss_feat + loss_proj1 + loss_proj2 + loss_acc
    with print_array_on_one_line():
        print('Current Loss {:>2.5f}: feature loss {:>2.5f}/norm(c{:>2.5f}, o{:>2.5f}), projection1 loss {:>2.5f}, projection2 loss {:>2.5f},'
              'error rate: {:>2.5f}'
          .format(loss.detach().cpu().numpy(), loss_feat.detach().cpu().numpy(),
                  feat_norm_comp.detach().cpu().numpy(), feat_norm_ori.detach().cpu().numpy(), loss_proj1.detach().cpu().numpy(),
                  loss_proj2.detach().cpu().numpy(), loss_acc.detach().cpu().numpy()))
    return loss


resblock_counter = 0


def compress_resblock(module_current, module_parent, net_current, net_parent, data, ckp, args):
    """
    module_current: the current module
    module_parent: the parent module
    net_current: the current full network
    net_parent: the parent full network
    data: used by data driven compression algorithm
    ckp: checkpoint used to write the log
    """

    for (name_cur, module_cur), (name_par, module_par) in zip(module_current._modules.items(), module_parent._modules.items()):
        if isinstance(module_cur, ResBlock):
            global resblock_counter, global_step
            global_step = 0
            resblock_counter += 1
            module_cur.prune_procedure = args.prune_procedure  # choices final, complete, and undergoing. module_cur.optimization changed in modify_submodules

            # get initialization values for the ResBlock to be compressed
            weight1, bn_weight1, bn_bias1, bn_mean1, bn_var1, _,\
                weight2, bn_weight2, bn_bias2, bn_mean2, bn_var2, _ = [p.data for k, p in module_cur.state_dict().items()]
            if args.prune_init_method.find('disturbance') >= 0:
                weight1, projection1 = init_weight_proj(weight1, init_method=args.prune_init_method, d=0)
                weight2, projection2 = init_weight_proj(weight2, init_method=args.prune_init_method, d=1)
            else:
                weight1, projection1 = init_weight_proj(weight1, init_method=args.prune_init_method)
                weight2, projection2 = init_weight_proj(weight2, init_method=args.prune_init_method)
            # modify submodules in the ResBlock
            modify_submodules(module_cur)
            # set ResBlock module params
            params = edict({'weight1': weight1, 'projection1': projection1, 'bias1': None,
                 'bn_weight1': bn_weight1, 'bn_bias1': bn_bias1, 'bn_mean1': bn_mean1, 'bn_var1': bn_var1,
                 'weight2': weight2, 'projection2': projection2, 'bias2': None,
                 'bn_weight2': bn_weight2, 'bn_bias2': bn_bias2, 'bn_mean2': bn_mean2, 'bn_var2': bn_var2})
            set_module_param(module_cur, params)
            # delete submodule parameter
            del_submodule_param(module_cur)
            # register module and parameter editting hook
            para_edit_hook_handle = module_cur.register_forward_pre_hook(module_edit_hook)
            module_cur.para_edit_hook_handle = para_edit_hook_handle

            ## optimize for the ResBlock
            args_optim = edict({'optimizer': args.prune_solver, 'momentum': 0.9, 'nesterov': False, 'betas': (0.9, 0.999),
                          'epsilon': 1e-8, 'weight_decay': args.prune_weight_decay, 'lr': args.prune_lr,
                          'decay': args.prune_decay, 'gamma': 0.1})
            # fine-tuning optimizer
            optimizer = make_optimizer(args_optim, module_cur, separate=True, scale=0.1)
            scheduler = make_scheduler(args_optim, optimizer)
            # add feature map collection hook
            add_feature_map_handle(module_cur, store_output=True)
            add_feature_map_handle(module_par, store_output=True)
            # used to collect the feature map of the intermediate layers
            # add_feature_map_handle(module_cur._modules['body']._modules['1'], store_input=True, store_output=True)
            # add_feature_map_handle(module_par._modules['body']._modules['1'], store_input=True, store_output=True)

            net_current.to(torch.device('cuda'))
            net_parent.to(torch.device('cuda'))
            global feature_map #, feature_map_inter
            # SGD optimization
            for b, (img, label) in enumerate(data):
                if b < args.prune_iteration:

                    for i in range(1):
                        # if b % 10 == 0:
                        print('\nCompress ResBlock {}, Batch {}, Iteration {}, Last lr {:2.5f}'
                              .format(resblock_counter - 1, b, scheduler.last_epoch, scheduler.get_lr()[0]))

                        optimizer.zero_grad()
                        img = img.to(torch.device('cuda'))
                        prediction = net_current(img)
                        output_comp = feature_map['output']
                        # embed()
                        # intermediate feature map
                        # output_comp_inter = feature_map_inter['output']
                        # input_comp_inter = feature_map_inter['input']
                        net_parent(img)
                        output_parent = feature_map['output'].detach()
                        # intermediate feature map
                        # output_parent_inter = feature_map_inter['output']
                        # input_parent_inter = feature_map_inter['input']
                        padding=1
                        normalize=True

                        # intermediate feature map
                        # p = os.path.join(ckp.args.dir_save, ckp.args.save, 'feature_grid_inter')
                        # if not os.path.exists(p):
                        #     os.makedirs(p)
                        # grid_comp_inter = feature_visualize(output_comp_inter, 16, 16, normalize=normalize, padding=padding)
                        # grid_parent_inter = feature_visualize(output_parent_inter, 16, 16, normalize=normalize, padding=padding)
                        # save_tensor_image(grid_comp_inter, p + '/grid_comp{}.png'.format(scheduler.last_epoch))
                        # save_tensor_image(grid_parent_inter, p + '/grid_parent{}.png'.format(scheduler.last_epoch))
                        # save_tensor_image(grid_parent_inter - grid_comp_inter, p + '/grid_sub{}.png'.format(scheduler.last_epoch))
                        # grid_comp_in = feature_visualize(input_comp_inter, 16, 16, normalize=normalize, padding=padding)
                        # grid_parent_in = feature_visualize(input_parent_inter, 16, 16, normalize=normalize, padding=padding)
                        # save_tensor_image(grid_comp_in, p + '/in_grid_comp{}.png'.format(scheduler.last_epoch))
                        # save_tensor_image(grid_parent_in, p + '/in_grid_parent{}.png'.format(scheduler.last_epoch))
                        # save_tensor_image(grid_parent_in - grid_comp_in, p + '/in_grid_sub{}.png'.format(scheduler.last_epoch))

                        p = os.path.join(ckp.args.dir_save, ckp.args.save, 'feature_grid')
                        if not os.path.exists(p):
                            os.makedirs(p)
                        grid_comp = feature_visualize(output_comp, 16, 16, normalize=normalize, padding=padding)
                        grid_parent = feature_visualize(output_parent, 16, 16, normalize=normalize, padding=padding)
                        save_tensor_image(grid_comp,  p + '/grid_comp{}.png'.format(scheduler.last_epoch))
                        save_tensor_image(grid_parent, p + '/grid_parent{}.png'.format(scheduler.last_epoch))
                        save_tensor_image(grid_parent - grid_comp, p + '/grid_sub{}.png'.format(scheduler.last_epoch))

                        # output_parent.requires_grad = False
                        loss = compute_loss(output_parent, output_comp, module_cur.projection1, module_cur.projection2,
                                            prediction, label,
                                            lambda_factor=args.prune_regularization, q=args.q)



                        if args.prune_procedure == 'complete':
                            optimizer.param_groups[0]['params'] = list(filter(lambda x: x.requires_grad, module_cur.parameters()))
                            # for p in optimizer.param_groups[0]['params']:
                            #     if p.grad is not None:
                            #         p.grad.data = torch.zeros_like(p)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()


                        # check the the change rate of projection, and the params and buffers of batchnorm
                        # global projection
                        # if b >= 1:
                        #     x = module_cur.projection1 / projection['projection1']
                        #     print('Divide projection1 max: {:>2.5f}, min {:>2.5f}'.
                        #           format(x.detach().cpu().max().numpy(), x.detach().cpu().min().numpy()))
                        #
                        #     x = module_cur.bn_weight1 / projection['bn_weight1']
                        #     print('Divide bn weight1  max: {:>2.5f}, min {:>2.5f}'.
                        #           format(x.detach().cpu().max().numpy(), x.detach().cpu().min().numpy()))
                        #
                        #     x = module_cur.bn_bias1 / projection['bn_bias1']
                        #     print('Divide bn bias1    max: {:>2.5f}, min {:>2.5f}'.
                        #           format(x.detach().cpu().max().numpy(), x.detach().cpu().min().numpy()))
                        #
                        #     x = module_cur.bn_mean1 / projection['bn_mean1']
                        #     print('Divide bn_mean1    max: {:>2.5f}, min {:>2.5f}'.
                        #           format(x.detach().cpu().max().numpy(), x.detach().cpu().min().numpy()))
                        #
                        #     x = module_cur.bn_var1 / projection['bn_var1']
                        #     print('Divide bn_var1     max: {:>2.5f}, min {:>2.5f}'.
                        #           format(x.detach().cpu().max().numpy(), x.detach().cpu().min().numpy()))
                        # projection = {'projection1': module_cur.projection1.clone(), 'projection2': module_cur.projection2.clone(),
                        #               'bn_weight1': module_cur.bn_weight1.clone(), 'bn_bias1': module_cur.bn_bias1,
                        #               'bn_mean1': module_cur.bn_mean1.clone(), 'bn_var1': module_cur.bn_var1}

                        # check the gradients of weight and projection matrix
                        print('Projection1 grad max: {:2.5f}, min: {:2.5f}, Weight1 grad max: {:2.5f}, min: {:2.5f}'.
                            format(module_cur.projection1.grad.max().detach().cpu().numpy(),
                                   module_cur.projection1.grad.min().detach().cpu().numpy(),
                                   module_cur.weight1.grad.max().detach().cpu().numpy(),
                                   module_cur.weight1.grad.min().detach().cpu().numpy()))
                        print('Projection2 grad max: {:2.5f}, min: {:2.5f}, Weight2 grad max: {:2.5f}, min: {:2.5f}'.
                            format(module_cur.projection2.grad.max().detach().cpu().numpy(),
                                   module_cur.projection1.grad.min().detach().cpu().numpy(),
                                   module_cur.weight2.grad.max().detach().cpu().numpy(),
                                   module_cur.weight2.grad.min().detach().cpu().numpy()))

                else:
                    break

            # remove parameter editting hook
            module_cur.para_edit_hook_handle.remove()

            # remove feature map collection hook
            remove_feature_map_handle(module_cur, store_input=True)
            remove_feature_map_handle(module_par, store_output=True)
            # used to collect the feature map of the intermediate layers
            # remove_feature_map_handle(module_cur._modules['body']._modules['1'], store_input=True, store_output=True)
            # remove_feature_map_handle(module_par._modules['body']._modules['1'], store_input=True, store_output=True)

            # prepare parameters for fine-tuning
            module_cur.optimization = False
            print('\nPrepare for finetuning.')
            prune(module_cur)
        elif module_cur is None:
            continue
        else:
            compress_resblock(module_cur, module_par, net_current, net_parent, data, ckp, args)

def compress_resblock_together(net_current, net_parent, data, ckp, args):
    modules = []
    for module_cur in net_current.modules():
        if isinstance(module_cur, ResBlock):
            modules.append(module_cur)
    for module_cur in modules:
        global resblock_counter
        resblock_counter += 1
        module_cur.prune_procedure = args.prune_procedure  # choices final, complete, and undergoing. module_cur.optimization changed in modify_submodules

        # get initialization values for the ResBlock to be compressed
        weight1, bn_weight1, bn_bias1, bn_mean1, bn_var1, _, \
        weight2, bn_weight2, bn_bias2, bn_mean2, bn_var2, _ = [p.data for k, p in module_cur.state_dict().items()]
        embed()
        # bn_weight1 = torch.ones(len(bn_weight1))
        # bn_bias1 = torch.zeros(len(bn_bias1))
        # bn_mean1 = torch.zeros(len(bn_mean1))
        # bn_var1 = torch.ones(len(bn_var1))
        # bn_weight2 = torch.ones(len(bn_weight2))
        # bn_bias2 = torch.zeros(len(bn_bias2))
        # bn_mean2 = torch.zeros(len(bn_mean2))
        # bn_var2 = torch.ones(len(bn_var2))
        if args.prune_init_method.find('disturbance') >= 0:
            weight1, projection1 = init_weight_proj(weight1, init_method=args.prune_init_method, d=0, s=0.05)
            weight2, projection2 = init_weight_proj(weight2, init_method=args.prune_init_method, d=1, s=0.05)
        else:
            weight1, projection1 = init_weight_proj(weight1, init_method=args.prune_init_method)
            weight2, projection2 = init_weight_proj(weight2, init_method=args.prune_init_method)
        # embed()
        # modify submodules in the ResBlock
        modify_submodules(module_cur)
        # set ResBlock module params
        params = edict({'weight1': weight1, 'projection1': projection1, 'bias1': None,
                        'bn_weight1': bn_weight1, 'bn_bias1': bn_bias1, 'bn_mean1': bn_mean1, 'bn_var1': bn_var1,
                        'weight2': weight2, 'projection2': projection2, 'bias2': None,
                        'bn_weight2': bn_weight2, 'bn_bias2': bn_bias2, 'bn_mean2': bn_mean2, 'bn_var2': bn_var2})
        set_module_param(module_cur, params)
        # delete submodule parameter
        del_submodule_param(module_cur)
        # register module and parameter editting hook
        para_edit_hook_handle = module_cur.register_forward_pre_hook(module_edit_hook)
        module_cur.para_edit_hook_handle = para_edit_hook_handle

    ## optimize for the ResBlock
    args_optim = edict({'optimizer': args.prune_solver, 'momentum': 0.9, 'nesterov': False, 'betas': (0.9, 0.999),
                        'epsilon': 1e-8, 'weight_decay': args.prune_weight_decay, 'lr': args.prune_lr,
                        'decay': args.prune_decay, 'gamma': 0.1})
    # fine-tuning optimizer
    optimizer = make_optimizer(args_optim, net_current, separate=True, scale=0.1)
    scheduler = make_scheduler(args_optim, optimizer)

    net_current.to(torch.device('cuda'))
    # net_parent.to(torch.device('cuda'))
    # global feature_map  # , feature_map_inte
    for b, (img, label) in enumerate(data):
        if b < 100:

            for i in range(1):
                # if b % 10 == 0:
                print('\nCompress All ResBlocks, Batch {}, Iteration {}, Last lr {:2.5f}'
                      .format(b, scheduler.last_epoch, scheduler.get_lr()[0]))

                optimizer.zero_grad()
                img = img.to(torch.device('cuda'))
                prediction = net_current(img)

                def compute_loss(modules, prediction, label,
                                 lambda_factor=1.0, q=1):
                    """
                    loss = ||Y - Yc||^2 + lambda * (||A_1||_{2,1} + ||A_2 ^T||_{2,1})
                    """
                    loss_function = nn.MSELoss()
                    loss_proj1 = 0
                    loss_proj2 = 0
                    for m in modules:
                        projection1 = m.projection1.squeeze().t()
                        projection2 = m.projection2.squeeze().t()
                        loss_proj1 = torch.sum(torch.sum(projection1 ** 2, dim=0) ** (q / 2)) ** (
                                    1 / q) * lambda_factor
                        loss_proj2 = torch.sum(torch.sum(projection2 ** 2, dim=1) ** (q / 2)) ** (
                                    1 / q) * lambda_factor
                    loss_proj1 /= len(modules)
                    loss_proj2 /= len(modules)
                    loss_function_ce = nn.CrossEntropyLoss()
                    # embed()
                    loss_acc = loss_function_ce(prediction, label.cuda())
                    # embed()
                    # loss_proj1 = torch.sum((torch.sum(projection1 ** 2, dim=0) ** q))
                    # loss_proj2 = torch.sum((torch.sum(projection2 ** 2, dim=1) ** q))
                    loss_proj1 = torch.tensor(0).cuda()
                    loss_proj2 = torch.tensor(0).cuda()
                    loss = loss_proj1 + loss_proj2 + loss_acc
                    # embed()
                    with print_array_on_one_line():
                        print(
                            'Current Loss {:>2.5f}: projection1 loss {:>2.5f}, projection2 loss {:>2.5f},'
                            'error rate: {:>2.5f}'
                            .format(loss.detach().cpu().numpy(),
                                    loss_proj1.detach().cpu().numpy(),
                                    loss_proj2.detach().cpu().numpy(), loss_acc.detach().cpu().numpy()))
                    return loss
                # embed()
                loss = compute_loss(modules,prediction, label,lambda_factor=args.prune_regularization, q=args.q)
                torch.save({'img': img, 'label': label, 'prediction': prediction, 'loss': loss}, 'batch.pt')

                # embed()
                loss.backward()
                optimizer.step()
                scheduler.step()
                torch.save(net_current.state_dict(), 'before1.pt')
        else:
            break

def make_model(inputs):
    """
    parent_module is the orginal module
    base_module is the compressed module
    """
    args = inputs[0]
    parent_module = import_module('model.' + args.base_p.lower())
    current_module = import_module('model.' + args.base.lower())
    class Pruning(getattr(current_module, args.base)):

        def __init__(self, args):
            self.args = args[0]
            self.ckp = args[1]
            super(Pruning, self).__init__(self.args)
            # self.load(self.args, strict=False) # load the pretrained parameters

            # prepare parent model
            parent_model = parent_module.make_model(args[0:1])
            self.num_params = dutil.param_count(self, self.args.ignore_linear)
            s = 224 if self.args.data_train == 'ImageNet' else 32
            self.flops = get_model_flops(self, (3, s, s))

            data = args[2]
            self.ckp.write_log('Start Network Compression!')
            # self.train()
            compress_resblock_together(self, parent_model, data, self.ckp, self.args)
            # self.eval()
            torch.save(self.state_dict(), 'before2.pt')
            # self.to('cpu')
            # embed()
            # self.flops_compress = get_model_flops(self, (3, s, s))
            # embed()
            # print('Flops before and after compression: {}, {}, compression ratio: {}'.
            #       format(self.flops, self.flops_compress, self.flops_compress/self.flops))
            # os.system('rm -rf ' + self.save_dir)

    return Pruning(inputs)

    # class PruningOld(getattr(base_module, args.base)):
    #
    #     def __init__(self, args):
    #         # set all of the parameters and super initialize the class.
    #         self.args = args[0]
    #         self.ckp = args[1]
    #         super(PruningOld, self).__init__(self.args, resblock=PResBlock)
    #         self.load(self.args, strict=False) # load the pretrained parameters
    #         self.save_dir = os.path.join(self.args.dir_save, self.args.save, 'features')
    #         self.use_data_parent = self.args.decomp_type == 'gsvd' or self.args.decomp_type == 'svd-mse'
    #         self.use_data_base = self.args.decomp_type == 'svd-mse'
    #         self.n_batch = 50 #50000 // self.args.batch_size
    #         self.device = torch.device('cpu' if self.args.cpu else 'cuda')
    #
    #
    #         # prepare parent model
    #         parent_model = parent_module.make_model(args[0:1])
    #         # print(parent_model)
    #         self.num_params = dutil.param_count(parent_model, self.args.ignore_linear)
    #         s = 224 if self.args.data_train == 'ImageNet' else 32
    #         self.parent_flops = get_model_flops(parent_model, (3, s, s))
    #
    #         # find the conv layers in parent model
    #         parent_layers = dutil.find_conv(parent_model, criterion=ResBlock)
    #         current_layers = dutil.find_conv(self, criterion=PResBlock)
    #
    #         for b, block in enumerate(parent_layers):
    #             weight1, bn_weight1, bn_bias1, bn_means1, bn_var1, _, \
    #                 weight2, bn_weight2, bn_bias2, bn_means2, bn_var2, _ = list(block.parameters())
    #             weight1, projection1 = weight_proj_init(weight1, init_method='svd')
    #             weight2, projection2 = weight_proj_init(weight2, init_method='svd')
    #             current_layers[b].set_params(
    #                 {'weight1': weight1, 'projection1': projection1, 'bias1': None, 'bn_weight1': bn_weight1,
    #                  'bn_bias1': bn_bias1, 'bn_means1': bn_means1, 'bn_var1': bn_var1,
    #                  'weight2': weight2, 'projection2': projection2, 'bias2': None, 'bn_weight2': bn_weight2,
    #                  'bn_bias2': bn_bias2, 'bn_means2': bn_means2, 'bn_var2': bn_var2,
    #                  'slope_p': nn.Parameter(torch.tensor(1.0)), 'slope_n': nn.Parameter(torch.tensor(0.0))})



















