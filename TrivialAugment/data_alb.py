import bisect
import logging
import os
import random
from collections import Counter
from copy import deepcopy


import torchvision
from PIL import Image

from torch.utils.data import SubsetRandomSampler, Sampler
from torch.utils.data.dataset import ConcatDataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C
from types import MethodType

from TrivialAugment.augmentations_alb import *
from TrivialAugment.common import get_logger, copy_and_replace_transform, stratified_split, denormalize
from TrivialAugment.imagenet import ImageNet

#from TrivialAugment.augmentations import Lighting
from aug_lib_alb import UniAugmentWeighted

import albumentations as A
from albumentations.pytorch import ToTensorV2


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    From https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/18
    """
    if cls == ConcatDataset:
        def __getitem__(self, index):
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
            if dataset_idx == 0:
                sample_idx = index
            else:
                sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
            data, target = self.datasets[dataset_idx][sample_idx]
            return data, target, index
    else:
        def __getitem__(self, index):
            data, target = cls.__getitem__(self, index)
            return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def dataset_with_transform_stats(cls, y="targets"):
    if y=="targets":
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            # some datasets do not return a PIL Image
            # and transforms assume a PIL Image as input
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            if self.transform is not None:
                for t in self.transform.transforms:
                    if isinstance(t, UniAugmentWeighted):
                        img, op_num, level = t(img)
                    else:
                        img = t(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target, torch.tensor(op_num), torch.tensor(level)
    elif y=="labels":
        def __getitem__(self, index):
            img, target = self.data[index], int(self.labels[index])
            # some datasets do not return a PIL Image
            # and transforms assume a PIL Image as input
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.transpose(img, (1, 2, 0)))

            if self.transform is not None:
                for t in self.transform.transforms:
                    if isinstance(t, UniAugmentWeighted):
                        img, op_num, level = t(img)
                    else:
                        img = t(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target, torch.tensor(op_num), torch.tensor(level)
    elif y=="targets-only":
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            # some datasets do not return a PIL Image
            # and transforms assume a PIL Image as input
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            if self.transform is not None:
                for t in self.transform.transforms:
                    if isinstance(t, UniAugmentWeighted):
                        img, _, _ = t(img)
                    else:
                        img = t(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target
    else:
        raise NotImplementedError
    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def batched_transforms(cls, y="targets", n_aug=1):
    if y=="targets":
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            # some datasets do not return a PIL Image
            # and transforms assume a PIL Image as input
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            
            if self.transform is not None:
                aug_img = []
                for n in range(self.n_aug):
                    for t in self.transform.transforms:
                        if isinstance(t, UniAugmentWeighted):
                            img = t(img)
                        else:
                            img = t(img)
                    aug_img.append(img)
                if isinstance(img, torch.Tensor):
                    aug_img = torch.stack(aug_image)

            if self.target_transform is not None:
                target = self.target_transform(target)
            return aug_img, target

    elif y=="labels":
        def __getitem__(self, index):
            img, target = self.data[index], int(self.labels[index])
            # some datasets do not return a PIL Image
            # and transforms assume a PIL Image as input
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.transpose(img, (1, 2, 0)))

            if self.transform is not None:
                aug_img = []
                for n in range(self.n_aug):
                    for t in self.transform.transforms:
                        if isinstance(t, UniAugmentWeighted):
                            img = t(img)
                        else:
                            img = t(img)
                    aug_img.append(img)
                if isinstance(img, torch.Tensor):
                    aug_img = torch.stack(aug_image)

            if self.target_transform is not None:
                target = self.target_transform(target)
            return aug_img, target

    elif y=="targets-only":
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            # some datasets do not return a PIL Image
            # and transforms assume a PIL Image as input
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            if self.transform is not None:
                aug_img = []
                for n in range(self.n_aug):
                    for t in self.transform.transforms:
                        if isinstance(t, UniAugmentWeighted):
                            img = t(img)
                        else:
                            img = t(img)
                    aug_img.append(img)
                if isinstance(img, torch.Tensor):
                    aug_img = torch.stack(aug_image)

            if self.target_transform is not None:
                target = self.target_transform(target)
            return aug_img, target

    else:
        raise NotImplementedError
    
    return type(cls.__name__, (cls,), {
            '__getitem__': __getitem__,
            'n_aug':n_aug })

logger = get_logger('TrivialAugment')
logger.setLevel(logging.INFO)
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) # these are for CIFAR 10, not for cifar100 actaully. They are pretty similar, though.
# mean für cifar 100: tensor([0.5071, 0.4866, 0.4409])


def get_dataloaders(dataset, batch, dataroot, split=0.0, split_idx=0, num_workers=0):
    dataset_info = {}
    pre_transform_train = A.Compose([])
    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        dataset_info['mean'] = _CIFAR_MEAN
        dataset_info['std'] = _CIFAR_STD
        dataset_info['img_dims'] = (3,32,32)
        dataset_info['num_labels'] = 100 if '100' in dataset and 'ten' not in dataset else 10
    elif 'pre_transform_cifar' in dataset:
        pre_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        dataset_info['mean'] = _CIFAR_MEAN
        dataset_info['std'] = _CIFAR_STD
        dataset_info['img_dims'] = (3, 32, 32)
        dataset_info['num_labels'] = 100 if '100' in dataset and 'ten' not in dataset else 10
    elif 'svhn' in dataset:
        svhn_mean = [0.4379, 0.4440, 0.4729]
        svhn_std = [0.1980, 0.2010, 0.1970]
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])
        dataset_info['mean'] = svhn_mean
        dataset_info['std'] = svhn_std
        dataset_info['img_dims'] = (3, 32, 32)
        dataset_info['num_labels'] = 10
    elif 'imagenet' in dataset:
        print("IMNET DEFAULT LOADERS")
        transform_train = A.Compose([
            A.RandomResizedCrop(224,224, scale=(0.08, 1.0)),
            A.HorizontalFlip(),
            A.HueSaturationValue(hue_shift_limit=int(0.4*360), sat_shift_limit=int(0.4*360), val_shift_limit=int(0.4*360), always_apply=True),
            A.augmentations.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
        ])

        transform_test = A.Compose([
            A.augmentations.geometric.resize.Resize(256, 256, always_apply=True),
            A.augmentations.crops.transforms.CenterCrop(224,224),
            A.augmentations.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
            ToTensorV2(),
        ])
        dataset_info['mean'] = [0.485, 0.456, 0.406]
        dataset_info['std'] = [0.229, 0.224, 0.225]
        dataset_info['img_dims'] = (3,224,244)
        dataset_info['num_labels'] = 1000
    elif 'smallwidth_imagenet' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((224,224), scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset_info['mean'] = [0.485, 0.456, 0.406]
        dataset_info['std'] = [0.229, 0.224, 0.225]
        dataset_info['img_dims'] = (3,224,224)
        dataset_info['num_labels'] = 1000
    elif 'ohl_pipeline_imagenet' in dataset:
        pre_transform_train = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1.,1.,1.])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1.,1.,1.])
        ])
        dataset_info['mean'] = [0.485, 0.456, 0.406]
        dataset_info['std'] = [1.,1.,1.]
        dataset_info['img_dims'] = (3,224,224)
        dataset_info['num_labels'] = 1000
    else:
        raise ValueError('dataset=%s' % dataset)

    logger.debug('augmentation: %s' % C.get()['aug'])
    if C.get()['aug'] in ['randaugment', 'primaldual']:
        assert not C.get()['randaug'].get('corrected_sample_space') and not C.get()['randaug'].get('google_augmentations')
        transform_train.transforms.insert(0, get_randaugment(n=C.get()['randaug']['N'], m=C.get()['randaug']['M'],
                                                             weights=C.get()['randaug'].get('weights',None), bs=C.get()['batch']))
    elif C.get()['aug'] in ['default', 'inception', 'inception320']:
        pass
    else:
        raise ValueError('not found augmentations. %s' % C.get()['aug'])

    transform_train.transforms.insert(0, pre_transform_train)

    if C.get()['cutout'] > 0:
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))

    if 'preprocessor' in C.get():
        if 'imagenet' in dataset:
            print("Only using cropping/centering transforms on dataset, since preprocessor active.")
            transform_train = A.Compose([
                A.RandomResizedCrop(224,224, scale=(0.08, 1.0)),
                ToTensorV2(),
            ])

            transform_test = A.Compose([
                A.augmentations.geometric.resize.Resize(256, 256, always_apply=True),
                A.augmentations.crops.transforms.CenterCrop(224,224),
                ToTensorV2(),
            ])
        else:
            print("Not using any transforms in dataset, since preprocessor is active.")
            transform_train = ToTensorV2()#PILImageToHWCByteTensor()
            transform_test =  ToTensorV2()#PILImageToHWCByteTensor()

    if dataset in ('cifar10', 'pre_transform_cifar10'):
        total_trainset = batched_transforms(torchvision.datasets.CIFAR10, y = "targets-only", n_aug=C.get()['n_aug'])(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset in ('cifar100', 'pre_transform_cifar100'):
        total_trainset = batched_transforms(torchvision.datasets.CIFAR100, y = "targets-only", n_aug=C.get()['n_aug'])(root=dataroot, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=dataroot, train=False, download=True, transform=transform_test)
    elif dataset == 'svhncore':
        total_trainset = batched_transforms(torchvision.datasets.SVHN, y="labels", n_aug=C.get()['n_aug'])(root=dataroot, split='train', download=True,
                                                   transform=transform_train)
        testset = torchvision.datasets.SVHN(root=dataroot, split='test', download=True, transform=transform_test)
    elif dataset == 'svhn':
       raise NotImplementedError
    elif dataset in ('imagenet', 'ohl_pipeline_imagenet', 'smallwidth_imagenet'):
        # Ignore archive only means to not to try to extract the files again, because they already are and the zip files
        # are not there no more
        total_trainset = ImageNet(root=os.path.join(dataroot, 'train'), transform=[transform_train, transform_test])
        testset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'val'), transform=transform_test)
        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    train_sampler = None
    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])

    test_sampler = None
    test_train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=train_sampler is None, num_workers=num_workers, pin_memory=True,
        sampler=train_sampler, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True,
        sampler=valid_sampler, drop_last=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True,
        drop_last=False, sampler=test_sampler
    )
    # We use this 'hacky' solution s.t. we do not need to keep the dataset twice in memory.
    #test_total_trainset = copy_and_replace_transform(total_trainset, transform_test)
    #clean_trainloader = torch.utils.data.DataLoader(
        #test_total_trainset, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True,
        #drop_last=False, sampler=train_sampler
    #)
    #clean_trainloader.denorm = lambda x: denormalize(x, dataset_info['mean'], dataset_info['std'])
    return train_sampler, trainloader, validloader, testloader, None, dataset_info

class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)
   
