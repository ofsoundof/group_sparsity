import os
import warnings
from importlib import import_module

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

def get_loader(args, kwargs):
    warnings.filterwarnings('ignore')
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    loader_train = None

    if not args.test_only:
        transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)]

        if not args.no_flip:
            transform_list.remove(transform_list[1])
        
        transform_train = transforms.Compose(transform_list)

        loader_train = DataLoader(
            datasets.ImageFolder(
                root=os.path.join(args.dir_data, 'train'),
                transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs
        )


    transform_list = [transforms.Resize(256)]
    batch_test = 128

    if args.crop > 1:
        def _fused(pil):
            tensor = transforms.ToTensor()(pil)
            normalized = transforms.Normalize(norm_mean, norm_std)(tensor)

            return normalized

        if args.crop == 5:
            transform_list.append(transforms.FiveCrop(224))
            batch_test //= 5
        elif args.crop == 10:
            transform_list.append(transforms.TenCrop(224))
            batch_test //= 10

        transform_list.append(transforms.Lambda(
            lambda crops: torch.stack([_fused(crop) for crop in crops])))
    else:
        transform_list.append(transforms.CenterCrop(224))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(norm_mean, norm_std))

    transform_test = transforms.Compose(transform_list)

    loader_test = DataLoader(
        datasets.ImageFolder(
            root=os.path.join(args.dir_data, 'val'),
            transform=transform_test),
        batch_size=batch_test, shuffle=False, **kwargs
    )

    return loader_train, loader_test
