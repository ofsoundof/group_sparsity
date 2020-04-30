import argparse
# from util import template

parser = argparse.ArgumentParser(description='Image Classification Options')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='disable CUDA training')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', default='/scratch_net/ofsoundof/yawli/Datasets/classification',
                    help='dataset directory')
parser.add_argument('--data_train', default='CIFAR10',
                    help='train dataset name')
parser.add_argument('--data_test', default='CIFAR10',
                    help='test dataset name')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_flip', action='store_true',
                    help='disable flip augmentation')
parser.add_argument('--crop', type=int, default=1,
                    help='enables crop evaluation')

# Log specifications
parser.add_argument('--dir_save', default='/scratch_net/ofsoundof/yawli/logs_iccv19_classification',
                    help='the directory used to save')
parser.add_argument('--save', default='test',
                    help='file name to save')
parser.add_argument('--load', default='',
                    help='file name to load')
parser.add_argument('--print_every', type=int, default=100,
                    help='print intermediate status per N batches')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--compare', type=str, default='',
                    help='experiments to compare with')
parser.add_argument('--top', type=int, default=1, choices=[1, -1],
                    help='save model for top1 or top5 error. top1: 1, top5: -1.')

# Model specifications
parser.add_argument('--model', default='DenseNet',
                    help='model name')
parser.add_argument('--template', default='',
                    help='You can set various templates in template.py')
parser.add_argument('--vgg_type', type=str, default='16',
                    help='VGG type')
parser.add_argument('--download', action='store_true',
                    help='download pre-trained model')
parser.add_argument('--base', default='',
                    help='base model')
parser.add_argument('--base_p', default='',
                    help='base model for parent')
parser.add_argument('--act', default='relu',
                    help='activation function')
parser.add_argument('--pretrain', default='',
                    help='pre-trained model directory')
# In the test_only mode of Hinge method,
# if pretrain.find('merge') >=0, then the merged model is loaded.
# Otherwise, the model before merging is loaded or there is no merging operation for the current network.
parser.add_argument('--extend', default='',
                    help='pre-trained model directory')
parser.add_argument('--depth', type=int, default=100,
                    help='number of convolution modules')
parser.add_argument('--in_channels', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--k', type=int, default=12,
                    help='DenseNet grownth rate')
parser.add_argument('--reduction', type=float, default=1,
                    help='DenseNet reduction rate')
parser.add_argument('--bottleneck', action='store_true',
                    help='ResNet/DenseNet bottleneck')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='kernel size')
parser.add_argument('--no_bias', action='store_true',
                    help='do not use bias term for conv layer')
parser.add_argument('--downsample_type', type=str, default='C',
                    help='downsample type of ResNet')
parser.add_argument('--precision', default='single',
                    help='model and data precision')
# wide resnet specification
parser.add_argument('--widen_factor', type=int, default=2,
                    help='the widening factor of wide resnet')
parser.add_argument('--dropout_rate', type=float, default=0.0,
                    help='the dropout rate of wide resnet')
# resnext specification
parser.add_argument('--cardinality', type=int, default=32,
                    help='cardinality of ResNeXt')
parser.add_argument('--bottleneck_width', type=int, default=4,
                    help='bottleneck width of ResNeXt')
# mobilenetv1, mobilenetv2, mnasnet, and shufflenetv2
parser.add_argument('--width_mult', type=float, default='1.0',
                    help='width multiplication scale for mobilenet_v1 and mobilenet_v2')
# mobilenetv3
parser.add_argument('--mode', type=str, default='large', choices=['large', 'small'],
                    help='the mode of mobilenetv3')
# regnet
parser.add_argument('--regime', type=str, default='large',
                    choices=['x_200mf', 'x_400mf', 'x_600mf', 'x_800mf', 'y_200mf', 'y_400mf', 'y_600mf', 'y_800mf'],
                    help='the flops regime of regnet')
# network FLOPs compression ratio
parser.add_argument('--ratio', type=float, default=0.2,
                    help='compression ratio')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--resume', type=int, default=-1,
                    help='load the model from the specified epoch')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training')

# Optimization specifications
parser.add_argument('--linear', type=int, default=1,
                    help='linear scaling rule')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate')
parser.add_argument('--decay', default='step-150-225',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'PG', 'APG', 'RMSprop'],
                    help='optimizer to use. SGD for image classification, PG for the Hinge method')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='enable nesterov momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM betas')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay parameter')

# Loss specifications
parser.add_argument('--loss', default='1*CE',
                    help='loss function configuration')

# Summary writer
parser.add_argument('--summary', action='store_true',
                    help='add tensorboardX summary to monitor the weights and the gradients')

# args = parser.parse_args()
# template.set_template(args)
#
# if args.epochs == 0:
#     args.epochs = 1e8
#
# if args.pretrain and args.pretrain != 'download':
#     args.n_init = 1
#     args.max_iter = 1

# args.lr_adjust_flag = args.model.lower().find('resnet') >= 0

