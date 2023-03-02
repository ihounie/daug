from torchvision.datasets.imagenet import *
import cv2
from joblib import Parallel, delayed, parallel_backend
import numpy as np
import os
from shutil import copytree
from time import time

def imread(file_path):
# Read an image with OpenCV
    image = cv2.imread(file_path)
    
    # By default OpenCV uses BGR color space for color images,
    # so we need to convert the image to RGB color space.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.
    Copied from torchvision, besides warning below.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

        WARN::
        This is the same ImageNet class as in torchvision.datasets.imagenet, but it has the `ignore_archive` argument.
        This allows us to only copy the unzipped files before training.
    """

    def __init__(self, root,transform=None,n_aug=2, albument=False, load_mem=True):
        if load_mem:
            mem_path = '/dev/shm/'+root.split("/")[-1]
            if not os.path.exists(mem_path):
                copytree("/home/chiche/imnet-data/"+root.split("/")[-1], "/dev/shm/"+root.split("/")[-1])
            root = mem_path
        self.albument = albument
        if self.albument:
            super().__init__(root, transform=transform, loader=imread)
        else:
            super().__init__(root, transform=transform)
        self.n_aug = n_aug
        #self.transform[0] = torch.jit.script(self.transform[0])
        #self.transform[1] = torch.jit.script(self.transform[1])
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        #tinit = time()
        img = self.loader(path)
        #print(f"Imload {time()-tinit}")
        if self.target_transform is not None:
            target = self.target_transform(target)
        #tinit = time()
        if self.n_aug>0:
            if self.albument:
                if self.transform is not None:
                    aug_img = [self.transform[1](image=img)['image'][None,:,:,:]]
                    for n in range(self.n_aug):
                        aug_img.append(self.transform[0](image=img)['image'][None,:,:,:])
                    aug_img = torch.cat(aug_img)
            else:
                if self.transform is not None:
                    aug_img = [self.transform[1](img)]
                    for n in range(self.n_aug):
                        aug_img.append(self.transform[0](img))
                    aug_img = torch.stack(aug_img)
        else:
            if self.albument:
                if self.transform is not None:
                    aug_img = self.transform(image=img)['image']
            else:
                aug_img = self.transform(img)
        #print(f" Transf {time()-tinit}")
        return aug_img, target

'''
class ImageNetMem(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.
    Copied from torchvision, besides warning below.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

        WARN::
        This is the same ImageNet class as in torchvision.datasets.imagenet, but it has the `ignore_archive` argument.
        This allows us to only copy the unzipped files before training.
    """

    def __init__(self, root,transform= None,n_aug=3, mem_frac = 0.5, albument=False):
        self.albument = albument
        if self.albument:
            super().__init__(root, transform=transform, loader=imread)
        else:
            super().__init__(root, transform=transform)
        self.n_aug = n_aug
        self.mem_frac = 0.1
        self.num_mem_samples = int(len(self.samples)*self.mem_frac)
        self.num_reloads=0
        self.__preload()

    def __preload_sample(self, sample, index):
        path, _ = sample
        img = np.asarray(self.loader(path), dtype=np.uint16)
        self.memdata[index] = img
        return

    def __preload(self):
        one_path, _ = self.samples[0]
        one_image = np.asarray(self.loader(one_path))
        self.memdata = self.num_mem_samples*[None,]
        with parallel_backend('threading', n_jobs=48):
            Parallel(require='sharedmem',verbose=10)(delayed(self.__preload_sample)(self.samples[i], i) for i in range(self.num_mem_samples))
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if index == self.num_mem_samples*self.num_reload:
            self.__preload()
            self.num_reload += 1
        img, target = self.memdata[index]
        img = fromarray(img)
        img = self.loader(path)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.albument:
            if self.transform is not None:
                aug_img = [self.transform[1](image=img)['image']]
                for n in range(self.n_aug):
                    aug_img.append(self.transform[0](image=img)['image'])
                aug_img = torch.cat(aug_img)
        else:
            if self.transform is not None:
                aug_img = [self.transform[1](img)]
                for n in range(self.n_aug):
                    aug_img.append(self.transform[0](img))
                aug_img = torch.stack(aug_img)

        return aug_img, target
'''