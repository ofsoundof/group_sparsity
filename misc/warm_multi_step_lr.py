import math
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler
#from IPython import embed
# MultiStep learning rate scheduler with warm restart
class WarmMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, scale=1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                'Milestones should be a list of increasing integers. Got {}',
                milestones
            )

        self.milestones = milestones
        self.gamma = gamma
        self.scale = scale

        self.warmup_epochs = 5
        self.gradual = (self.scale - 1) / self.warmup_epochs
        super(WarmMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (1 + self.last_epoch * self.gradual) / self.scale
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]

class FinetuneMultiStepLR(_LRScheduler):

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, factor=1, start=3, grad_ratio_method='p2'):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        self.factor = factor
        self.prune = True
        self.p1_p2_regularization = ''
        self.running_grad_ratio = 1
        self.grad_ratio_method = grad_ratio_method
        super(FinetuneMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        x = bisect_right(self.milestones, self.last_epoch)
        if self.prune:
            if self.p1_p2_regularization == '' or self.p1_p2_regularization == 'proximal':
                return [base_lr * self.gamma ** x for base_lr in self.base_lrs]
            else:
                if self.grad_ratio_method == 'p1':
                    ratio = 1 if self.running_grad_ratio ==0 else self.running_grad_ratio
                    lr = [base_lr * self.gamma ** x / (ratio ** (1.35)) for base_lr in self.base_lrs[:-1]]
                    lr.append(self.base_lrs[-1] * self.gamma ** x)
                else:
                    lr = [base_lr * self.gamma ** x for base_lr in self.base_lrs[:-1]]
                    lr.append(self.base_lrs[-1] * self.gamma ** x * (self.running_grad_ratio ** (1.35)))
                return lr
        else:
            self.base_lrs = [self.base_lrs[1], self.base_lrs[1]]
            return [self.factor * base_lr * self.gamma ** x for base_lr in self.base_lrs]

class HingeMultiStepLR(_LRScheduler):

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, lr_adjust_flag=False, lr_adjust_method='p1'):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        self.lr_adjust_flag = lr_adjust_flag
        self.lr_adjust_method = lr_adjust_method
        self.running_grad_ratio = 1
        super(HingeMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        x = bisect_right(self.milestones, self.last_epoch)
        if self.lr_adjust_flag:
            # In the searching stage of ResNet20, 56, 164 and Wide ResNet, need to determine how the learning rate
            # is adjusted for p1 and p2
            ratio = 1 if self.running_grad_ratio == 0 else self.running_grad_ratio
            # TODO: why running_grad_ratio could be 0.
            if self.lr_adjust_method == 'p1':
                lr = [base_lr * self.gamma ** x / (ratio ** (1.35)) for base_lr in self.base_lrs[:-1]]
                lr.append(self.base_lrs[-1] * self.gamma ** x)
            else: # self.lr_adjust_method = 'p2'
                lr = [base_lr * self.gamma ** x for base_lr in self.base_lrs[:-1]]
                lr.append(self.base_lrs[-1] * self.gamma ** x * (ratio ** (1.35)))
        else:
            # In converging stage, the base learning rates of all of the parameter groups are the same.
            lr = [base_lr * self.gamma ** x for base_lr in self.base_lrs]
        return lr

class CosineMultiStepLR(_LRScheduler):

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, start=3):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        self.prune = True
        super(CosineMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        x = bisect_right(self.milestones, self.last_epoch)
        # print(x, x < self.start, self.factor)
        if self.prune:
            return [base_lr * self.gamma ** x for base_lr in self.base_lrs]
        else:
            self.base_lrs = [0.05, 0.05]
            return [base_lr * (1 + math.cos(math.pi * (self.last_epoch + 72) / 150)) / 2
                    for base_lr in self.base_lrs]

