import os
import time
import datetime
import matplotlib
matplotlib.use('Agg')
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import misc.warm_multi_step_lr as misc_wms
import misc.custom_sgd as misc_cs
from misc.apg import APG
#from IPython import embed
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join(args.dir_save, args.save)
            if args.reset:
                os.system('rm -rf ' + self.dir)
        else:
            self.dir = os.path.join(args.dir_save, args.load)
            if not os.path.exists(self.dir):
                args.load = ''

        os.makedirs(os.path.join(self.dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(self.dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.dir, 'features'), exist_ok=True)
        os.makedirs(os.path.join(self.dir, 'per_layer_compression_ratio'), exist_ok=True)
        # prepare log file
        log_dir = os.path.join(self.dir, 'log.txt')
        open_type = 'a' if os.path.exists(log_dir) else 'w'
        self.log_file = open(log_dir, open_type)
        # print config information
        with open(os.path.join(self.dir, 'config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.compare = []
        if len(args.compare) > 0:
            if args.compare == 'same':
                no_postfix = '_'.join(args.save.split('_')[:-1])
                for d in os.listdir('../experiment'):
                    if d.find(no_postfix) >= 0 and d != args.save:
                        self.compare.append(d)
            else:
                self.compare = args.compare.split('+')

        self.write_log('Batch size: {} = {} x {}'.format(
            args.batch_size, args.linear, args.batch_size // args.linear
        ))

    def save(self, trainer, epoch, converging=False, is_best=False):
        trainer.model.save(self.dir, epoch, converging=converging, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir)

        if not converging:
            torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))
            torch.save(epoch, os.path.join(self.dir, 'epochs.pt'))
        else:
            torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer_converging.pt'))
            torch.save(epoch, os.path.join(self.dir, 'epochs_converging.pt'))

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot(self, epoch):
        pass

    def save_results(self, epoch, m):
        # This is used by clustering convolutional kernels. Not used by me.
        m = m.get_model()
        if hasattr(m, 'split'):
            from model import clustering
            for i in set(v for v in m.split['map'].values()):
                centroids = getattr(m, 'centroids_{}'.format(i), None)
                if centroids is not None:
                    clustering.save_kernels(
                        centroids,
                        '{}/results/iter{:0>3}_c{:0>2}.png'.format(
                            self.dir, epoch, i
                        )
                    )


def make_optimizer(args, target, ckp=None, lr=None):
    if args.optimizer.find('SGD') >= 0:
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum, 'nesterov': args.nesterov}
    elif args.optimizer == 'APG':
        optimizer_function = APG
        kwargs = {'momentum': args.momentum, 'nesterov': args.nesterov,
                  'regularization': args.prune_regularization, 'prox_frequency': args.prox_frequency}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {'betas': args.betas,
            'eps': args.epsilon}
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}
    elif args.optimizer == 'CustomSGD':
        optimizer_function = misc_cs.CustomSGD
        kwargs = {'momentum': args.momentum, 'nesterov': args.nesterov}
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optimizer))

    kwargs['lr'] = args.lr if lr is None else lr
    kwargs['weight_decay'] = args.weight_decay

    trainable = filter(lambda x: x.requires_grad, target.parameters())
    optimizer = optimizer_function(trainable, **kwargs)

    if args.load != '' and ckp is not None:
        print('Loading the optimizer from the checkpoint...')
        optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
    return optimizer


def make_scheduler(args, target):
    '''
    :param args:
    :param target:
    :param resume:
    :param last_epoch:
    :param prune: True -> prune scheduler
                  False -> finetune scheduler
    :return:
    '''
    # embed()
    if args.decay.find('step') >= 0:
        milestones = list(map(lambda x: int(x), args.decay.split('-')[1:]))
        kwargs = {'milestones': milestones, 'gamma': args.gamma}
        if args.decay.find('warm') >= 0:
            scheduler_function = misc_wms.WarmMultiStepLR
            kwargs['scale'] = args.linear
        else:
            scheduler_function = lrs.MultiStepLR
    elif args.decay.find('cosine') >= 0:
        # kwargs['start'] = args.start
        # scheduler_function = misc_wms.CosineMultiStepLR
        kwargs = {'T_max': args.epochs}
        scheduler_function = lrs.CosineAnnealingLR
    if args.load != '':
        last_epoch = torch.load(os.path.join(args.dir_save, args.save, 'epochs.pt'))
    else:
        last_epoch = -1
    kwargs['last_epoch'] = last_epoch
    scheduler = scheduler_function(target, **kwargs)
    # else:
    #     raise NotImplementedError('Scheduler is not implemented')
    return scheduler


def make_optimizer_hinge(args, target, ckp=None, converging=False, lr_adjust_flag=False, lr=None):
    """
    :param args: the arguments
    :param target: the target model
    :param ckp: the checkpoint
    :param converging: indicate whether in the searching stage or in the converging stage
                        If converging is True, do not need to set lr_adjust_flag
    :param lr_adjust_flag: indicate whether to adjust the learning rate for the two convs within one block, e.g. ResBlock
    :param lr:
    :return:
    """

    if args.optimizer == 'SGD' or args.optimizer == 'PG':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum, 'nesterov': args.nesterov}
    elif args.optimizer == 'APG':
        optimizer_function = APG
        kwargs = {'momentum': args.momentum, 'nesterov': args.nesterov,
                  'regularization': args.prune_regularization, 'prox_frequency': args.prox_frequency}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {'betas': args.betas, 'eps': args.epsilon}
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optimizer))
    kwargs['weight_decay'] = args.weight_decay
    kwargs['lr'] = args.lr if lr is None else lr
    # lr is the base learning rate for the projection parameters during the searching stage.
    # lr * lr_ratio is the base learning rate for the other parameters during the searching stage.
    # lr * lr_factor is the base learning rate for all of the parameters during the converging stage

    trainable = list(filter(lambda x: x.requires_grad, target.parameters()))
    if converging:
        kwargs['lr'] = kwargs['lr'] * args.lr_factor
        optimizer = optimizer_function(trainable, **kwargs)
    else:
        # lr_adjust depends on how many independent sparsity-inducing matrices exist in the current block.
        # 1. ResNet: there are two sparsity-inducing matrices in both the basic and the bottleneck block.
        # 2. MobileNet and ResNeXt: the two sparsity-inducing matrices are combined and regularized together.
        # 3. DenseNet and VGG: there is only one sparsity-inducing matrix in one block.
        #    It is possible to have both P1 and P2 which indicates that both pruning and decomposition are used.
        #    But P1 and P2 are just from the rows and columns of the same sparsity-inducing matrix, respectively.
        if not lr_adjust_flag:
            if args.model.lower().find('mobilenet') >= 0:
                trainable_proj = [x for x in trainable
                                  if x.dim() == 4 and x.size()[-1] == 1 and x.size()[0] > 16 and x.size()[0] < 1280]
                trainable_finetune = [x for x in trainable
                                      if not (x.dim() == 4 and x.size()[-1] == 1 and x.size()[0] > 16 and x.size()[0] < 1280)]
            else:
                trainable_proj = [x for x in trainable if x.dim() == 4 and x.size()[-1] == 1]
                trainable_finetune = [x for x in trainable if x.dim() != 4 or x.size()[-1] != 1]
            optimizer = optimizer_function([{'params': trainable_finetune, 'lr': kwargs['lr'] * args.lr_ratio},
                                            {'params': trainable_proj}], **kwargs)
        else:
            trainable_finetune = [x for x in trainable if x.dim() != 4 or x.size()[-1] != 1]
            trainable_proj1 = []
            trainable_proj2 = []
            for m in target.get_model().find_modules():
                proj1, proj2 = target.get_model().sparse_param(m)
                trainable_proj1.append(proj1)
                trainable_proj2.append(proj2)
            optimizer = optimizer_function([{'params': trainable_finetune, 'lr': kwargs['lr'] * args.lr_ratio},
                                            {'params': trainable_proj1},
                                            {'params': trainable_proj2}], **kwargs)

    if args.load != '' and ckp is not None:
        if not converging:
            print('Loading the optimizer in the searching stage from the checkpoint...')
            optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
        else:
            print('Loading the optimizer in the converging stage from the checkpoint...')
            optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer_converging.pt')))

    return optimizer


def make_scheduler_hinge(args, target, converging=False, lr_adjust_flag=False):
    if args.decay.find('step') >= 0:
        milestones = list(map(lambda x: int(x), args.decay.split('-')[1:]))
        kwargs = {'milestones': milestones, 'gamma': args.gamma}
        if args.decay.find('warm') >= 0:
            scheduler_function = misc_wms.WarmMultiStepLR
            kwargs['scale'] = args.linear
        elif args.decay.find('cosine') >= 0:
            kwargs['start'] = args.start
            scheduler_function = misc_wms.CosineMultiStepLR
        elif args.decay.find('hinge') >= 0:
            scheduler_function = misc_wms.HingeMultiStepLR
            kwargs['lr_adjust_flag'] = not converging and lr_adjust_flag # in the searching phase when compressing ResNet20, 56, 164 and Wide ResNet.
            kwargs['lr_adjust_method'] = args.lr_adjust_method
        else:
            scheduler_function = lrs.MultiStepLR
        if args.load != '':
            if not converging:
                last_epoch = torch.load(os.path.join(args.dir_save, args.save, 'epochs.pt'))
            else:
                last_epoch = torch.load(os.path.join(args.dir_save, args.save, 'epochs_converging.pt'))
        else:
            last_epoch = -1
        kwargs['last_epoch'] = last_epoch
        scheduler = scheduler_function(target, **kwargs)
    else:
        raise NotImplementedError('Scheduler is not implemented')

    return scheduler

