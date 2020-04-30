__author__ = 'Yawei Li'
import torch
from torch.nn import init
from torchvision.utils import make_grid
from numpy import linalg
import numpy as np
import math
import imageio
import copy
import matplotlib.pyplot as plt
from model.in_use.flops_counter import get_model_complexity_info
from contextlib import contextmanager
#from IPython import embed


########################################################################################################################
# used to initialize the weight and projection parameters derived from one conv layer
########################################################################################################################

# ========
# SVD initialization, singular value is merge into the first matrix.
# ========
def init_svd(weight):
    # print('\nSVD Initialization, Singular Value Merged into the Weight Matrix')
    ws = weight.size()
    weight_out = weight.view(ws[0], -1).t()
    usv = linalg.svd(weight_out.detach().cpu().numpy().astype(np.float64))
    u, s, v = [torch.from_numpy(i) for i in usv]
    s_new = torch.zeros(torch.prod(torch.tensor(ws[1:])), ws[0]).to(torch.float64)
    s_new[:s.shape[0], :s.shape[0]] = s.diag()
    weight_out = torch.mm(u, s_new).t().view(ws).to(torch.float32)
    projection = v.t().view(ws[0], ws[0], 1, 1).to(torch.float32)
    # embed()
    return weight_out, projection


# ========
# SVD initialization, singular value is merged into the projection matrix.
# ========
def init_svd2(weight):
    # print('\nSVD Initialization, Singular Value Merged into Projection Matrix')
    ws = weight.size()
    weight_out = weight.view(ws[0], -1).t()
    usv = linalg.svd(weight_out.detach().cpu().numpy().astype(np.float64))
    u, s, v = [torch.from_numpy(i) for i in usv]
    # print(u.shape[0] < v.shape[0], u.shape[0], v.shape[0])
    if u.shape[0] < v.shape[0]:
        s_new = torch.zeros(torch.prod(torch.tensor(ws[1:])), ws[0]).to(torch.float64)
        s_new[:s.shape[0], :s.shape[0]] = s.diag()
        weight_out = u.t().view(-1, ws[1], ws[2], ws[3]).to(torch.float32)
        projection = torch.mm(s_new, v).t().view(ws[0], -1, 1, 1).to(torch.float32)
    else:
        weight_out = u[:, :ws[0]].t().view(ws).to(torch.float32)
        projection = torch.mm(s.diag(), v).t().view(ws[0], ws[0], 1, 1).to(torch.float32)
    return weight_out, projection


# def init_svd2(weight):
#     # print('\nSVD Initialization, Singular Value Merged into Projection Matrix')
#     ws = weight.size()
#     weight_out = weight.view(ws[0], -1).t()
#     usv = linalg.svd(weight_out.detach().cpu().numpy().astype(np.float64))
#     u, s, v = [torch.from_numpy(i) for i in usv]
#     weight_out = u[:, :ws[0]].t().view(ws).to(torch.float32)
#     projection = torch.mm(s.diag(), v).t().view(ws[0], ws[0], 1, 1).to(torch.float32)
#     return weight_out, projection


# ========
# Weight from the original parameters, projection as an identity matrix
# ========
def init_projection_identity(weight):
    # print('\nWeight Original, Identity Initialization for Projection')
    ws = weight.size()
    projection = torch.eye(ws[0]).view(ws[0], ws[0], 1, 1)
    return weight, projection


# ========
# Random initialization. Not preferred. Takes long time to train.
# ========
def init_total_random(weight):
    # print('\nRandom Initialization for Both Weight and Projection')
    ws = weight.size()
    weight = torch.zeros(ws)
    projection = torch.zeros(ws[0], ws[0], 1, 1)
    # init.xavier_normal_(weight)
    # init.xavier_normal_(projection)
    init.kaiming_normal_(weight, a=math.sqrt(5))
    init.kaiming_normal_(projection, a=math.sqrt(5))
    return weight, projection


# ========
# Weight from the original parameters, projection as a random matrix
# ========
def init_projection_random(weight):
    # print('\nWeight Original, Random Initialization for Projection')
    ws = weight.size()
    projection = torch.zeros(ws[0], ws[0], 1, 1)
    init.xavier_normal_(projection)
    return weight, projection


# ========
# Weight from the original parameters, projection as a disturbed identity matrix
# ========
def init_projection_disturbance_uniform(weight, d, s=0.1):
    # print('\nWeight Original, Small Disturbance (Uniform) of Identity for Projection')
    ws = weight.size()
    projection = torch.eye(ws[0]) + torch.rand([ws[0], ws[0]]) * s
    if d == 0:
        projection = projection.div_(projection.norm(2, dim=0)).t().view(ws[0], ws[0], 1, 1)
    else:
        projection = projection.t().div_(projection.t().norm(2, dim=0)).view(ws[0], ws[0], 1, 1)
    return weight, projection


# ========
# Weight from the original parameters, projection as a disturbed identity matrix
# ========
def init_projection_disturbance_normal(weight, d, s=0.1):
    # print('\nWeight Original, Small Disturbance (Normal) of Identity for Projection')
    ws = weight.size()
    projection = torch.eye(ws[0]) + torch.randn([ws[0], ws[0]]) * s
    if d == 0:
        projection = projection.div_(projection.norm(2, dim=0)).t().view(ws[0], ws[0], 1, 1)
    else:
        projection = projection.t().div_(projection.t().norm(2, dim=0)).view(ws[0], ws[0], 1, 1)
    return weight, projection


# ========
# Select the initialization method
# For whole network compression, only svd, svd2, p-identity, and t-random are valid.
# ========
def init_weight_proj(weight, init_method, d=0, s=0.1):
    if init_method == 'svd':
        weight, projection = init_svd(weight)
    elif init_method == 'svd2':
        weight, projection = init_svd2(weight)
    elif init_method == 'p-identity':
        weight, projection = init_projection_identity(weight)
    elif init_method == 't-random':
        weight, projection = init_total_random(weight)
    elif init_method == 'p-random':
        weight, projection = init_projection_random(weight)
    elif init_method == 'p-disturbance-u':
        weight, projection = init_projection_disturbance_uniform(weight, d=d, s=s)
    elif init_method == 'p-disturbance-n':
        weight, projection = init_projection_disturbance_normal(weight, d=d, s=s)
    else:
        raise NotImplementedError('Initialization method {} not implemented.'.format(init_method))
    return weight.data, projection.data


########################################################################################################################
# The network and ResBlock compression main function
########################################################################################################################
def feature_visualize(feature, row=None, column=None, normalize=False, padding=2):
    if row is not None and column is not None:
        feature = feature[:row, :column, :, :]
    sz = feature.size()
    feature = torch.reshape(feature, (sz[0] * sz[1], 1, sz[2], sz[3]))
    feature = make_grid(feature, nrow=sz[1], normalize=normalize, padding=padding)
    # range=(0, float(feature.max().detach().cpu().numpy()))
    return feature


def save_tensor_image(tensor, filename):
    if tensor.max() <= 1:
        tensor = tensor * 255
    ndarr = tensor.permute(1, 2, 0).detach().to('cpu', torch.uint8).numpy()
    imageio.imwrite(filename, ndarr)


########################################################################################################################
# set numpy print format
########################################################################################################################

@contextmanager
def print_array_on_one_line():
    oldoptions = np.get_printoptions()
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=np.inf)
    yield
    np.set_printoptions(**oldoptions)


########################################################################################################################
# select the remaining channels
########################################################################################################################

def get_nonzero_index(x, dim='output', counter=1, percentage=0.2, threshold=5e-3, fix_channel=0):
    n = torch.norm(x, p=2, dim=0 if dim == 'output' else 1)
    if fix_channel == 0:
        if n.max() == 0:
            # print('flag')
            f = torch.randperm(int(len(n) * percentage)).sort().values.to(n.device)
        else:
            t = n.topk(int(len(n) * percentage)).values[-1]
            if t == 0:
                f = n > min(t, threshold)  # > n.max() / 10
                f = torch.nonzero(f).squeeze(dim=1)
                f0 = torch.randperm(int(len(n) * percentage)).sort().values.to(n.device)
                # print(int(len(n) * percentage))
                f = torch.cat([f0, f], dim=0).unique().sort().values
            else:
                if threshold == 0:
                    f = n > min(t, threshold)  # > n.max() / 10
                else:
                    f = n >= min(t, threshold)
                f = torch.nonzero(f).squeeze(dim=1)
    else:
        if n.max() == 0:
            f = torch.randperm(fix_channel).sort().values.to(n.device)
        else:
            t = n.topk(fix_channel).values[-1]
            if t == 0:
                f = n > t
                f = torch.nonzero(f).squeeze(dim=1)
                f0 = torch.randperm(fix_channel).sort().values.to(n.device)
                # print(int(len(n) * percentage))
                f = torch.cat([f0, f], dim=0).unique().sort().values
            else:
                f = n >= t
                f = torch.nonzero(f).squeeze(dim=1)
        #print('{:<2}, {:2.8f}, {:2.8f}'.format(f.shape[0], t, n.max()))

    # global resblock_counter, global_step
    # writer.add_histogram('ResBlock{}_index{}'.format(resblock_counter, counter), n, global_step=global_step, bins=10)
    # global_step += 1
    # print(x[1, :5].detach().cpu().numpy())
    # if counter ==1:
        # with print_array_on_one_line():
        #     print('Norm of Projection1: {}.'.format(n.detach().cpu().numpy()))
    # print(f.detach().cpu().numpy())
    return n, f


########################################################################################################################
# plot the figures
########################################################################################################################

# used by compute_loss method in the hinge_** functions
def plot_figure(filter_matrix, l, filename):
    filter_matrix = torch.norm(filter_matrix, dim=0)
    axis = np.array(list(range(1, filter_matrix.shape[0]+1)))
    filter_matrix = filter_matrix.detach().cpu().numpy()
    fig = plt.figure()
    plt.title('Layer {}, Max: {:.4f}, Ave: {:.4f}, Min: {:.4f}'.
              format(l, filter_matrix.max(), filter_matrix.mean(), filter_matrix.min()))
    plt.plot(axis, filter_matrix, label='Unsorted')
    plt.plot(axis, np.sort(filter_matrix), label='Sorted')
    plt.legend()
    plt.xlabel('Filter Index')
    plt.ylabel('Filter Norm')
    plt.grid(True)
    plt.savefig(filename, dpi=50)
    # plt.show()
    plt.close(fig)


# used by main_hinge
def plot_compression_ratio(compression_ratio, filename, frequency_per_epoch=1):
    if frequency_per_epoch == 1:
        axis = np.array(list(range(1, len(compression_ratio) + 1)))
    else:
        axis = np.array(list(range(1, len(compression_ratio) + 1)), dtype=float) / frequency_per_epoch
    compression_ratio = np.array(compression_ratio)
    fig = plt.figure()
    plt.title('Network Compression Ratio')
    plt.plot(axis, compression_ratio)
    # plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close(fig)


# used by print_compress_info method in the hinge_** functions
def plot_per_layer_compression_ratio(ratio_per_layer, filename):
    channels = np.array(list(range(1, len(ratio_per_layer[0]) + 1)))
    fig = plt.figure()
    plt.title('Per-Layer Compression Ratio')
    for i, ratio in enumerate(ratio_per_layer):
        plt.plot(channels, np.array(ratio), label='P'.format(i + 1))
    plt.legend()
    plt.xlabel('Layer Index')
    plt.ylabel('Ratio')
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close(fig)


########################################################################################################################
# functions used by main_hinge and trainer. Used during the optimization.
########################################################################################################################

def calc_model_complexity(model):
    model = model.get_model()
    model.flops_compress, model.params_compress = get_model_complexity_info(model, model.input_dim,
                                                                           print_per_layer_stat=False)
    print('FLOPs ratio {:.2f} = {:.4f} [G] / {:.4f} [G]; Parameter ratio {:.2f} = {:.2f} [k] / {:.2f} [k].'
          .format(model.flops_compress / model.flops * 100, model.flops_compress / 10. ** 9, model.flops / 10. ** 9,
                  model.params_compress / model.params * 100, model.params_compress / 10. ** 3, model.params / 10. ** 3))


def calc_model_complexity_running(model, merge_flag=False):
    '''
    This function calculates the model complexity during the compressing procedure.
    It first nullifies the filters with small norms, and then calculates the model complexity.
    Note that the 3x3 and 1x1 needs to be merged if necessary.
    After that, the network parameters and structure before the calculation should be recovered.
    The merged convs from 3x3 and 1x1 convs are split again.
    '''
    state = copy.deepcopy(model.get_model().state_dict())
    model.get_model().compress()
    if merge_flag:
        # merge the 3x3 and 1x1 convs before the calculation and split the merged convs after the calculation.
        model.get_model().merge_conv()
        calc_model_complexity(model)
        # print(model)
        model.get_model().split_conv(state)
    else:
        calc_model_complexity(model)
        model.get_model().load_state_dict(state, strict=False)


def binary_search(model, target, merge_flag=False):
    """
    Binary search algorithm to determine the threshold
    :param model:
    :param target:
    :param merge_flag:
    :return:
    """
    # target = 0.70
    # threshold = model.get_model().args.threshold
    step = 0.01
    # step_min = 0.0001
    status = 1.0
    stop = 0.001
    counter = 1
    max_iter = 100

    while abs(status - target) > stop and counter <= max_iter:
        status_old = status
        # calculate flops and status
        calc_model_complexity_running(model, merge_flag)
        status = model.get_model().flops_compress / model.get_model().flops

        string = 'Iter {}: current step={:1.4f}, current threshold={:2.8f}, status={:2.4f}, flops={}.\n' \
            .format(counter, step, model.get_model().args.threshold, status, model.get_model().flops_compress)
        print(string)

        if abs(status - target) > stop:
            # calculate the next step
            flag = False if counter == 1 else (status_old >= target) == (status < target)
            if flag:
                step /= 2
            # calculate the next threshold
            if status > target:
                model.get_model().args.threshold += step
            elif status < target:
                model.get_model().args.threshold -= step
                model.get_model().args.threshold = max(model.get_model().args.threshold, 0)

            counter += 1
            # deal with the unexpected status
            if model.get_model().args.threshold < 0 or status <= 0:
                print('Status {} or threshold {} is out of range'.format(status, model.get_model().args.threshold))
                break
        else:
            print('The target compression ratio is achieved. The loop is stopped')


def reg_anneal(lossp, regularization_factor, annealing_factor, annealing_t1, annealing_t2):
    """
    Anneal the regularization factor
    :param lossp:
    :param regularization_factor:
    :param annealing_factor:
    :param annealing_t1:
    :param annealing_t2:
    :return:
    """
    if annealing_factor == 0:
        regularization = regularization_factor
    else:
        if annealing_t2 < lossp <= annealing_t1:
            regularization = regularization_factor / annealing_factor
        elif lossp <= annealing_t2:
            regularization = regularization_factor / annealing_factor ** 2
        else:
            regularization = regularization_factor
    return regularization
