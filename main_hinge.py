"""
This is the main function for Group Sparsity: The Hinge Between Filter Pruning and Decomposition forNetwork Compression.
The method is denoted as 'Hinge'.
"""
__author__ = 'Yawei Li'
import os
import torch
from tensorboardX import SummaryWriter
from util import utility
from data import Data
from model_hinge import Model
from loss import Loss
from util.trainer_hinge import Trainer
from util.option_hinge import args
from model_hinge.hinge_utility import calc_model_complexity, calc_model_complexity_running, binary_search, plot_compression_ratio
# from IPython import embed

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    """
    The program could start from 4 different 'phases' and they are used to decide how to load the model.
    Phase 1: training from scratch; load = '' 
            -> no loading
    Phase 2: testing phase; test_only, load = '', pretrain = '*/model/model_latest.pt' or '*/model/model_merge_latest.pt'
            -> loading the pretrained model
    Phase 3: loading models for PG optimization; load = '*/' -- a directory, epoch_continue = None
            -> loading from model_latest.pt
    Phase 4: loading models to continue the training. load = '*/' -- a directory, epoch_continue = a number
            -> loading from model_continue.pt
    During the loading phase (3 & 4), args.load is set to a directory. The loaded model is determined by the 'stage' of 
    the algorithm. 
    Thus, need to first determine the stage of the algorithm. 
    Then decide whether to load model_latest.pt or model_continue_latest.pt
    
    The stages of the algorithm is determined by epoch_continue. The optimization procedure in those 
    two stages are different.
    epoch_continue denotes where the proximal gradient (PG) optimization finishes and the training continues 
    until the convergence of the network.
    epoch_continue = None -> PG optimzation stage (searching stage)
    epoch_continue = a number -> continue-training stage (converging stage)
    """

    # ========
    # Step 1: Initialize the objects.
    # ========

    # Data loader, model, loss function, and trainer are first level objects.
    # They are initialized in the main function.

    # Get epoch_continue or alternatively get 'epochs_converging.pt'.
    # This is used to indicate whether the algorithm is in the stage of PG optimization.
    # compression_information.pt exists. This means that PG optimization has finished and epoch_continue is saved there.
    # compression_information.pt does not exist. This means that this is a new experiment. Thus, just set epoch_continue
    # to None to indicate this phase.
    merge_flag = args.model.lower() in ['hinge_resnet56', 'hinge_wide_resnet', 'hinge_densenet']
    info_path = os.path.join(args.dir_save, args.save, 'compression_information.pt')
    if args.load != '' and os.path.exists(info_path):
        # converging stage
        info = torch.load(info_path)
        epoch_continue = info['epoch_continue']
    else:
        # searching stage
        epoch_continue = None
    converging = False if epoch_continue is None else True

    # data loader, model, teacher model, loss function, and trainer
    loader = Data(args)
    # The model is automatically loaded depending on the phase. Phase 2, 3, and 4.
    # Phase 1: do not load in the initialization of Model, but load in the initialization of hinge.
    # Phase 2: depending on args.pretrain.find('merge') >= 0
    # Phase 3: load from '*/model_lastest.pt'
    # Phase 4: load from '*/model_converging_latest.pt'
    # There is no need to reload afterwards.
    my_model = Model(args, checkpoint, converging=converging)
    model_teacher = Model(args, checkpoint, teacher=True) if args.distillation or args.load_original_param else None
    loss = Loss(args, checkpoint)
    writer = SummaryWriter(os.path.join(args.dir_save, args.save), comment='searching') if args.summary else None
    t = Trainer(args, loader, my_model, loss, checkpoint, writer, converging, model_teacher)

    # ========
    # Step 2:  In the PG-optimization (searching) stage
    # ========

    # Judge which stage the algorithm is in, namely the searching stage or the converging stage.
    # If already in the converging stage, need to skip the searching phase in the beginning.
    if not converging and not args.test_only:
        #TODO: Range normalization (or calibration, balancing?)
        current_ratio, ratio_log = 1.0, []
        # Searching
        while current_ratio - args.ratio > args.stop_limit: # either change the stop condition here or change the nullifying threshold
            t.train()
            t.test()
            calc_model_complexity_running(my_model, merge_flag)
            current_ratio = my_model.get_model().flops_compress / my_model.get_model().flops
            ratio_log.append(current_ratio)
            plot_compression_ratio(ratio_log, os.path.join(args.dir_save, args.save, 'compression_ratio.png'))

        # Binary searching for the optimal nullifying threshold that achieves the target FLOPs compression ratio
        binary_search(my_model, args.ratio, merge_flag)

        # Print more information about the compressed network: i. compression ratio, ii. the index of the compressed channel.
        epoch_continue = t.scheduler.last_epoch + 1
        my_model.get_model().print_compress_info(epoch_continue)

        # Compressing the network and loading the network parameters from the compression procedure or the pre-trained network.
        kwargs = {'load_original_param': args.load_original_param}
        if args.load_original_param:
            kwargs['model_teacher'] = t.model_teacher
        my_model.get_model().compress(**kwargs)

    # ========
    # Step 3: Reinitialize the optimizer and the lr_scheduler
    # ========

    t.reset_after_optimization(epoch_continue)

    # ========
    # Step 4: In the converging stage. All of the 4 Phases 1, 2, 3, 4 need to pass this step.
    # ========

    while not t.terminate():
        # if t.scheduler.last_epoch == 0 and not args.test_only:
        #     t.test()
        # if t.scheduler.last_epoch + 1 == 2:
        t.train()
        t.test()

    if merge_flag:
        my_model.get_model().merge_conv()
        if not args.test_only:
            target = my_model.get_model().state_dict()
            torch.save(target, os.path.join(checkpoint.dir, 'model/model_merge_latest.pt'))
    calc_model_complexity(my_model)
    checkpoint.done()

'''
Question: what if all of the columns or rows are zeros?
    Should avoid such cases but need to deal with them
        Solution: if all of the columns or rows are zeros, that means the regularization is so strong,
        which should not be the case. In this case, the program should stop and give a warning that the regularization
        should be reduced.
        Next: with this solved, the current_ratio should reduce slowly to the predefined.

Range normalization (or calibration, balancing?)

current_ratio = 1.0

while current_ratio - args.ratio > 0.02:
    continue the epoch:
        train()
        test()
    prune the network
    merge the convs if necessary (resnet and others but not mobilenet)
    compute the current compression ratio: current_ratio
    #if current_ratio - args.ratio > 0.02:
    split the convs if necessary
    reload the original state_dict of the model

binary search: need to match the compression ratio very closely

prune the network for finetuning according to the searched threshold: note that here the convs needs not to be merged.
    provide more information about the pruned network here.
        compression ratio,
        pruned channels

reset the optimizer
reset the lr scheduler, need to use step() to get the correct learning rate for finetuning.

while not t.terminate():
    train()
    test()


Question 1: use one while loop or two while loops? one for pruning optimization, the other one for finetuning.
    I decided to use two loops.

Question 2: During finetuning, should we count from the first epoch?
    No. If counting from the first epoch, then how should the loss and error log should be dealt with?
    I want to show the loss and error log during pruning and finetuning in the same figure.

Question 3: is it still necesary to have the finetune epoch flag in the beginning.
    Yes. There are several reasons:
    Need to show the loss, training and test error during pruning and finetuning in the same figure.
    Used to distinguish the pruning phase and finetuning phase in trainer

Question 4: whether to use different batch sizes for pruning and finetuning?
    Maybe needed for ImageNet classification to make the pruning procedure faster.

Question 5: How to train the network during pruning optimization phase?
    keep the original parameter almost fixed?
        problem: the norm of the filters reduces with the same speed.
    or change it a lot?
        problem: the accuracy reduces a lot. More epochs is needed to recover the original accuracy.
'''

'''
The epoch_start_finetune tag

It is used in a somewhat loop manner.
1. At the begining of the algorithm, need to check whether the program should be in finetuning phase.
    This is achieved by examining whether the Prune_info.pt file exists in the current save folder.
2. If already in finetuning phase, need to skip the pruning procedure. Go directly to the finetuning procedure.
    If still in pruning phase, continue until the end of the program.
    Need special dealing with the optimizer and lr_scheduler during loading mode?
3. During training, the epoch_start_finetune tag is used to choose the different bells and whistles in the two different phase.
4. After pruning, the epoch_start_finetune tag is saved in Prune_info.pt.


'''

'''
whether need to add args.epoch_start_finetune?
need to consider the load mode for pruning procedure and finetuning procedure.
'''

'''
Pruning: optimizer -- args.load
         lr_scheduler -- args.load
         thus, args.load is used to judge whether in load mode
Finetuning: normal mode: args.load = ''
                optimizer -- unloaded
                lr_scheduler -- step is not used
            load mode: argw.load = 'load_dir'
                optimizer -- don't need load, but is loaded
                lr_scheduler -- step is used as usual
            Conclusion: in finetune mode, optimizer should not be loaded and lr_scheduler should be stepped.
'''













