from util.option import parser
from util import template

# Group
parser.add_argument('--group_size', type=int, default=16,
                    help='group size for the network of filter group approximation, ECCV 2018 paper.')

# DenseNet Basis
parser.add_argument('--n_group', type=int, default=1,
                    help='number of groups for the compression of densenet')
parser.add_argument('--k_size1', type=int, default=3,
                    help='kernel size 1')
parser.add_argument('--k_size2', type=int, default=3,
                    help='kernel size 2')
parser.add_argument('--inverse_index', action='store_true',
                    help='index the basis using inverse index')
parser.add_argument('--transition_group', type=int, default=6,
                    help='number of groups in the transition layer of DenseNet')

# ResNet Basis
parser.add_argument('--basis_size1', type=int, default=16,
                    help='basis size for the first res group in ResNet')
parser.add_argument('--basis_size2', type=int, default=32,
                    help='basis size for the second res group in ResNet')
parser.add_argument('--basis_size3', type=int, default=64,
                    help='basis size for the third res group in ResNet')
parser.add_argument('--n_basis1', type=int, default=24,
                    help='number of basis for the first res group in ResNet')
parser.add_argument('--n_basis2', type=int, default=48,
                    help='number of basis for the second res group in ResNet')
parser.add_argument('--n_basis3', type=int, default=84,
                    help='number of basis for the third res group in ResNet')

# Basis specification
parser.add_argument('--vgg_decom_type', type=str, default='all',
                    help='vgg decomposition type, valid value all, select')
parser.add_argument('--basis_size_str', type=str, default='',
                    help='basis size')
parser.add_argument('--n_basis_str', type=str, default='',
                    help='number of basis')
parser.add_argument('--basis_size', type=int, default=128,
                    help='basis size')
parser.add_argument('--n_basis', type=int, default=128,
                    help='number of basis')
parser.add_argument('--pre_train_optim', type=str, default='/home/yawli/projects/clustering-kernels/models/vgg16-89711a85.pt',
                    help='pre-trained weights directory')
parser.add_argument('--unique_basis', action='store_true',
                    help='whether to use the same basis for the two convs in the Residual Block')
parser.add_argument('--loss_norm', action='store_true',
                    help='whether to use default loss_norm')

# Decompose specification
parser.add_argument('--use_data', action='store_true',
                    help='whether to use feature map data to constrain the decomposition')
parser.add_argument('--decomp_type', type=str, default='svd', choices=['gsvd', 'svd', 'svd-mse'],
                    help='whether to use GSVD or SVD to decompose the filters.')
parser.add_argument('--gsvd_constrain', type=str, default='features', choices=['params', 'features'],
                    help='whether to use weight parameters or feature maps to constrain GSVD.')
parser.add_argument('--comp_method', type=str, default='normal', choices=['fixed-rank', 'adp-simple', 'adp-tight', 'manual'],
                    help='the method used to compress the network.')
parser.add_argument('--comp_rule', type=str, default='s-value', choices=['s-value', 'f-norm', 's-value-normalization', 'f-norm-normalization'],
                    help='the compression rule')
parser.add_argument('--comp_target', type=str, default='params', choices=['params', 'flops'],
                    help='the compression target')
parser.add_argument('--sample', type=int, default=4,
                    help='number of samples extracted from each feature map.')
parser.add_argument('--n_data_driven', type=int, default=80,
                    help='number of images used for data-driven decomposition or pruning.')
parser.add_argument('--conv_single', action='store_true',
                    help='whether to separate a conv into a conv followed by a 1x1 conv. '
                         'Flase: decompose a conv into a smaller conv and a 1x1 conv.'
                         'True: a single conv is reconstructed from the reduced parameters.')
parser.add_argument('--a_sep', action='store_true',
                    help='whether to use the matrix A resulting from mse as a single 1x1 conv. '
                         'Reference: Extreme-Network-Compression-via-Filter-Group-Approximation')
parser.add_argument('--energy_constrain', action='store_true',
                    help='whether to constrain the energy in the remained singular values.')
parser.add_argument('--ignore_linear', action='store_true',
                    help='whether to ignore the linear layer during compression.')
parser.add_argument('--include_bias', action='store_true',
                    help='whether to ignore the linear layer during compression.')
parser.add_argument('--searching_method', default='energy', choices=['energy', 'metric'],
                    help='searching method')

args = parser.parse_args()
template.set_template(args)


