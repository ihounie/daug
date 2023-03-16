from torchvision.datasets import ImageFolder
import os
from shutil import copytree
from aug_lib import UniAugmentWeighted
import torch

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

    def __init__(self, root, transform=None, split='train',frac_samples=0.02, download=None, load_mem=True, with_indexes=False, with_stats=False):
        if load_mem:
            mem_path = '/dev/shm/'+split
            if not os.path.exists(mem_path):
                copytree(root+split, "/dev/shm/"+split)
            root = mem_path
        super(ImageNet, self).__init__(root,transform=transform)
        self.root = root
        self.split=split
        self.with_indexes = with_indexes
        self.with_stats = with_stats
        #print("classes", self.classes)
        self.class_to_idx = {clss: idx
                             for idx, clss in enumerate(self.classes)}
        self.frac_samples = frac_samples
        self.preprocess()

    def preprocess(self):
        self.samples = []
        directory = os.path.expanduser(self.root)
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                num_samples = int(len(fnames)*self.frac_samples)
                fnames = sorted(fnames)
                for f in range(num_samples):
                    path = os.path.join(root, fnames[f])
                    item = path, class_index
                    self.samples.append(item)
        print("samples", len(self.samples))
    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.target_transform is not None:
                target = self.target_transform(target)
        if self.transform is not None:
            for t in self.transform.transforms:
                if isinstance(t, UniAugmentWeighted):
                    img, op_num, level = t(img)
                else:
                    img = t(img)
        if self.with_stats:
            return img, target, torch.tensor(op_num), torch.tensor(level)
        elif self.with_indexes:
            return img, target, index
        else:
            return img, target
    