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
    Phase 1: training phase, need to load the model.  load = '', pretrain = '*.pt' 
             -> loading the pretrained orginal model
             -> not loading the optimizer
    Phase 2: testing phase; test_only.                load = '', pretrain = '*/model/model_latest.pt' or '*/model/model_merge_latest.pt'
             -> loading the pretrained compressed model
             -> not loading the optimizer
    Phase 3: loading models for PG optimization.      load = '*/' -- a directory, pretrain = '*.pt', epoch_continue = None
             -> loading from model_latest.pt
             -> loading optimizer.pt
    Phase 4: loading models to continue the training. load = '*/' -- a directory, pretrain = '*.pt', epoch_continue = a number
             -> loading from model_converging_latest.pt
             -> loading optimizer_converging.pt
    During the loading phase (3 & 4), args.load is set to a directory. The loaded model is determined by the 'stage' of 
            the algorithm. 
    Thus, need to first determine the stage of the algorithm. 
    Then decide whether to load model_latest.pt or model_continue_latest.pt
    
    The algorithm has two stages, i.e, proximal gradient (PG) optimization (searching stage) and continue-training 
            (converging stage). 
    The stage is determined by epoch_continue. The variable epoch_continue denotes the epoch where the PG optimization
            finishes and the training continues until the convergence of the network.
    i.  epoch_continue = None -> PG optimzation stage (searching stage)
    ii. epoch_continue = a number -> continue-training stage (converging stage)
    
    Initial epoch_continue:
        Phase 1, 2, &3 -> epoch_continue = None, converging = False
        PHase 4 -> epoch_continue = a number, converging = True
    """

    # ==================================================================================================================
    # Step 1: Initialize the objects.
    # ==================================================================================================================

    # Data loader, model, loss function, and trainer are first level objects. They are initialized in the main function.

    # Get epoch_continue or alternatively get 'epochs_converging.pt'.
    # This is used to indicate whether the algorithm is in the stage of PG optimization.
    # i.  compression_information.pt exists. This means that PG optimization has finished and epoch_continue is saved there.
    # ii. compression_information.pt does not exist. This means that this is a new experiment. Thus, just set epoch_continue
    #     to None to indicate this phase.
    merge_flag = args.model.lower() in ['hinge_resnet56', 'hinge_wide_resnet', 'hinge_densenet']
    info_path = os.path.join(checkpoint.dir, 'compression_information.pt')
    if args.load != '' and os.path.exists(info_path):
        # converging stage
        info = torch.load(info_path)
        epoch_continue = info['epoch_continue']
    else:
        # searching stage
        epoch_continue = None
    # Judge which stage the algorithm is in, namely the searching stage or the converging stage.
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
    writer = SummaryWriter(checkpoint.dir, comment='searching') if args.summary else None
    t = Trainer(args, loader, my_model, loss, checkpoint, writer, converging, model_teacher)

    # ==================================================================================================================
    # Step 2:  In the PG-optimization (searching) stage
    # ==================================================================================================================

    # If already in the converging stage, need to skip the searching phase in the beginning.
    if not converging and not args.test_only:
        current_ratio, ratio_log = 1.0, []
        # Searching
        # t.test()
        # t.model = t.model_teacher
        # t.test()
        while current_ratio - args.ratio > args.stop_limit and not t.terminate(): # either change the stop condition here or change the nullifying threshold
            t.train()
            t.test()
            calc_model_complexity_running(my_model, merge_flag)
            current_ratio = my_model.get_model().flops_compress / my_model.get_model().flops
            ratio_log.append(current_ratio)
            plot_compression_ratio(ratio_log, os.path.join(checkpoint.dir, 'compression_ratio.png'))

        if args.summary:
            t.writer.close()
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

    # ==================================================================================================================
    # Step 3: Reinitialize the optimizer and the lr_scheduler
    # ==================================================================================================================

    t.reset_after_optimization(epoch_continue)

    # ==================================================================================================================
    # Step 4: In the converging stage. All of the 4 Phases 1, 2, 3, 4 need to pass this step.
    # ==================================================================================================================

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
    if args.summary:
        t.writer.close()
    print(my_model)
    checkpoint.done()

