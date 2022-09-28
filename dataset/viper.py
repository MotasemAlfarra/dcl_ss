import glob
import os

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed
from .utils import Subset

BASE_PATH = '/export/share/datasets/VIPER/'
class VIPER(data.Dataset):

    def __init__(self, root='./', train=True, transform=None, domain_transform=None):
        # root = os.path.expanduser(root)
        train_folder = os.path.join(BASE_PATH, 'train')
        val_folder = os.path.join(BASE_PATH, 'val')

        self.images = [  # Add train cities
            (
                path,
                path.replace('img', 'annotations').replace('.jpg', '.png'),
                -1
            ) for path in sorted(glob.glob(os.path.join(train_folder, "img/*/*.jpg")))
        ]
        self.images += [  # Add val cities
            (
                path,
                path.replace('img', 'annotations').replace('.jpg', '.png'),
                -1
            ) for path in sorted(glob.glob(os.path.join(val_folder, "img/*/*.jpg")))
        ]
        self.images = [self.images[i] for i in range(0, len(self.images), 10)]# This is to skip frames
        self.transform = transform
        self.domain_transform = domain_transform

    def __getitem__(self, index, get_domain=False):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if get_domain:
            domain = self.images[index][2]
            if self.domain_transform is not None:
                domain = self.domain_transform(domain)
            return domain

        try:
            img = Image.open(self.images[index][0]).convert('RGB')
            target = Image.open(self.images[index][1])
            # target = transform_mask(np.asarray(target))
            # target = Image.fromarray(target)
        except Exception as e:
            raise Exception(f"Index: {index}, len: {len(self)}, message: {str(e)}")

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class VIPER_Incremental(data.Dataset):
    """Labels correspond to domains not classes in this case."""
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=range(19),
        idxs_path=None,
        masking=True,
        overlap=True,
        **kwargs
    ):

        full_data = VIPER(root, train)
        # print('length', len(full_data))
        # if idxs_path is not None and os.path.exists(idxs_path):
        #     idxs = np.load(idxs_path).tolist()
        # else:
        #     print("I am here")
        idxs = filter_images(full_data, labels)
        if idxs_path is not None and distributed.get_rank() == 0:
            np.save(idxs_path, np.array(idxs, dtype=int))

        rnd = np.random.RandomState(1)
        rnd.shuffle(idxs)
        train_len = int(0.8 * len(idxs))
        if train:
            idxs = idxs[:train_len]
            print(f"{len(idxs)} images for train")
        else:
            idxs = idxs[train_len:]
            print(f"{len(idxs)} images for val")
        # import pdb
        # pdb.set_trace()
        target_transform = tv.transforms.Lambda(
            lambda t: t.
            apply_(lambda x: x) # the labels in BDD100K are already mapped! Do nothing then :)
        )
        # make the subset of the dataset
        self.dataset = Subset(full_data, idxs, transform, target_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)



def filter_images(dataset, labels):
    # This just returns the set of all indices
    idxs = []

    print(f"Filtering images...")
    for i in range(len(dataset)):
        idxs.append(i)
        # domain_id = dataset.__getitem__(i, get_domain=True)  # taking domain id
        # if domain_id in labels:
        #     idxs.append(i)
        # if i % 1000 == 0:
        #     print(f"\t{i}/{len(dataset)} ...")
    return idxs