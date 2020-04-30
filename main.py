"""
The main function to train a pure network without any modification.
"""
import torch
from util import utility
from data import Data
from model import Model
from loss import Loss
from util.trainer_clean import Trainer
from util.option_basis import args
from tensorboardX import SummaryWriter
import os



torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:

    # model
    # if args.decomp_type == 'gsvd' or args.decomp_type == 'svd-mse' or args.comp_rule.find('f-norm') >= 0 \
    #         or args.model.lower().find('prune_resnet56') >= 0:
    #     my_model = Model(args, checkpoint, loader.loader_train)
    # else:
    my_model = Model(args, checkpoint)
    # data loader
    loader = Data(args)
    # loss function
    loss = Loss(args, checkpoint)
    # writer
    writer = SummaryWriter(os.path.join(args.dir_save, args.save), comment='optimization') if args.summary else None
    # trainer
    t = Trainer(args, loader, my_model, loss, checkpoint, writer)
    # print('Mem 1 {:2.4f}'.format(torch.cuda.max_memory_allocated()/1024.0**3))
    while not t.terminate():
        # if t.scheduler.last_epoch == 0 and not args.test_only:
        #     t.test()
        # if t.scheduler.last_epoch + 1 == 2:
        t.train()
        t.test()

    checkpoint.done()

