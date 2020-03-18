from importlib import import_module

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

class Data:
    def __init__(self, args):
        kwargs = {
            'num_workers': args.n_threads,
            'pin_memory': True
        }
        if args.cpu: kwargs['pin_memory'] = False

        module = import_module('data.' + args.data_train.lower())
        self.loader_train, self.loader_test = module.get_loader(args, kwargs)
