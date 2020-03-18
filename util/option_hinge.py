from util.option import parser
from util import template

# Prune specification
# Possible unused
parser.add_argument('--prune_procedure', default='complete', choices=['complete', 'undergoing', 'final'],
                    help='pruning procedure.')
parser.add_argument('--prune_weight_decay', type=float, default=0.001,
                    help='weight decay during pruning phase.')
parser.add_argument('--prune_decay', type=str, default='step-100-200-300',
                    help='decay step during pruning phase')
parser.add_argument('--prune_lr', type=float, default=0.1,
                    help='learning rate during pruning phase.')
parser.add_argument('--prune_iteration', type=int, default=250,
                    help='number of iterations during pruning phase.')
parser.add_argument('--prune_solver', default='SGD', choices=['SGD', 'PG'],
                    help='pruning optimization method.')
parser.add_argument('--load_original_param', action='store_true',
                    help='To make the finetuning step faster, loading the parameters from the original network during'
                         'the pruning of the network.')

# Hinge used specification
parser.add_argument('--q', type=float, default=1,
                    help='l2,q norm.')
# prune_regularizer
parser.add_argument('--sparsity_regularizer', type=str, default='l1', choices=['l0', 'l1', 'l1-2', 'l1d2', 'logsum'],
                    help='the types of sparsity regularizer applied.')
# prune_regularization
parser.add_argument('--regularization_factor', type=float, default=1e-4,
                    help='the sparsity regularization factor.')
# prune_init_method
parser.add_argument('--init_method', default='svd2',
                    choices=['svd', 'svd2', 'p-identity', 'p-random', 't-random', 'p-disturbance-u', 'p-disturbance-n'],
                    help='the initialization method of the sparsity-inducing matrix.')
# prox_frequency
parser.add_argument('--prox_freq', default=1, type=int,
                    help='the frequency of the proximal operator applied to the sparsity-inducing matrix.')
#  prune_layer_balance
parser.add_argument('--layer_balancing', action='store_true',
                    help='adjust the sparsity regularization factors for different layers in order to have a balanced compressed ResBlock.')
parser.add_argument('--p1_p2_same_ratio', action='store_true',
                    help='In addition to the gradient based lr adjustment, enabling this flag directly forces the same '
                         'compression ratio for the two sparsity-inducing matrices p1 and p2.')
parser.add_argument('--lr_adjust_method', type=str, choices=['p1', 'p2'], default='p1',
                    help="'lr-p1' -- adjust the lr by dividing the lr of p1 by gradient ratio;"
                         "'lr-p2' -- adjust the lr by multipying the lr of p2 by gradient ratio")
# prune_percentage
parser.add_argument('--remain_percentage', type=float, default=0.2,
                    help='this indicates the minimum percentage of remaining channels after the compression procedure.')
# prune_regularization_anneal
parser.add_argument('--annealing_factor', type=float, default=0,
                    help='the annealing factor applied when trying to anneal the sparsity regularization factor.')
# prune_regularization_anneal_threshold1
parser.add_argument('--annealing_t1', type=float, default=10,
                    help='the first threshold for the annealing of the sparsity regularization factor. Default = 10 for ResNet56')
# prune_regularization_anneal_threshold2
parser.add_argument('--annealing_t2', type=float, default=5,
                    help='the second threshold for the annealing of the sparsity regularization factor. Default = 5 for ResNet56')
parser.add_argument('--teacher', type=str, default='/home/yawli/projects/image_classification/models/mobilenet_v2-b0353104.pth',
                    help='pretrained teacher model used for distillation')
parser.add_argument('--distillation', action='store_true',
                    help='whether to use distillation loss.')
# parser.add_argument('--baseline', type=str, default='.',
#                     help='pretrained baseline model used for compression')
# lr_ratio
parser.add_argument('--lr_ratio', type=float, default=0.1,
                    help='the ratio between the learning rates of the other parameters and the sparsity-inducing matrices')
# lr_factor
parser.add_argument('--lr_factor', type=float, default=1.0,
                    help='the factor between the base lr in the searching stage and in the converging stage')
# prune_threshold
parser.add_argument('--threshold', type=float, default=5e-3,
                    help='the threshold used to mask out or nullifying the small elements.')
parser.add_argument('--stop_limit', type=float, default=0.05,
                    help='the stop limit of the binary searching method')

parser.add_argument('--start', type=int, default=3,
                    help='finetune start, still used for mobilenetv2.')

args = parser.parse_args()
template.set_template(args)

