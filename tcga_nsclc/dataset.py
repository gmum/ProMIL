import csv
import os.path

import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor


class NSCLCPreprocessedBagsCross(data_utils.Dataset):
    def __init__(self, path, train=True, test=False, push=False, shuffle_bag=False, data_augmentation=False,
                 loc_info=False, folds=10, fold_id=1, random_state=3, all_labels=False, max_bag=20000):
        self.path = path
        self.train = train
        self.test = test
        self.folds = folds
        self.fold_id = fold_id
        self.random_state = random_state
        self.push = push
        self.all_labels = all_labels
        self.shuffle_bag = shuffle_bag
        self.data_augmentation = data_augmentation
        self.location_info = loc_info
        self.r = np.random.RandomState(random_state)
        self.max_bag = max_bag
        self.labels = {}
        self.files = os.listdir(self.path)
        self.files = [f for f in self.files if '.pth' in f]

    @classmethod
    def load_raw_image(cls, path):
        return to_tensor(pil_loader(path))

    class LazyLoader:
        def __init__(self, path, dir, indices):
            self.path = path
            self.dir = dir
            self.indices = indices

        def __getitem__(self, item):
            return NSCLCPreprocessedBagsCross.load_raw_image(
                os.path.join(self.path, self.dir, 'patch.{}.jpg'.format(int(self.indices[item]))))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        dir = self.files[index]
        try:
            bag = torch.load(os.path.join(self.path, dir))
        except:
            print(dir)
            raise

        if bag.shape[0] > self.max_bag:
            if self.train:
                rng = np.random
            else:
                rng = np.random.default_rng(3)
            indices = rng.permutation(bag.shape[0])[:self.max_bag]
            bag = bag[indices].detach().clone()
        else:
            indices = np.arange(0, bag.shape[0])
        if 'LUAD' in dir:
            label = 0
        elif 'LUSC' in dir:
            label = 1
        else:
            raise NotImplemented

        if self.push:
            return self.LazyLoader(self.path, dir, indices), bag, label
        else:
            return bag, label


if __name__ == '__main__':
    ds = NSCLCPreprocessedBagsCross(path="../data/NSCLCN_patches", train=False, all_labels=True, fold_id=1,
                                       folds=10, random_state=3, push=False)
    print(len(ds))
