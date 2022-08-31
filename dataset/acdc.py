import glob
import os

import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

from .utils import Subset


# Converting the id to the train_id. Many objects have a train id at
# 255 (unknown / ignored).
# See there for more information:
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
id_to_trainid = {
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,   # road
    8: 1,   # sidewalk
    9: 255,
    10: 255,
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    14: 255,
    15: 255,
    16: 255,
    17: 5,  # pole
    18: 255,
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    29: 255,
    30: 255,
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
    -1: 255
}


weather_conditions = {
    "fog": 0, "night": 1, "rain": 2, "snow": 3
}


class ACDC(data.Dataset):

    def __init__(self, root, train=True, transform=None, domain_transform=None):
        root = os.path.expanduser(root)
        annotation_folder = os.path.join(root, 'gt')
        image_folder = os.path.join(root, 'rgb_anon')

        self.images = [  # Add train cities
            (
                path,
                os.path.join(
                    annotation_folder,
                    path.split("/")[-4],
                    path.split("/")[-3],
                    path.split("/")[-2],
                    path.split("/")[-1][:-12] + "gt_labelIds.png"
                ),
                weather_conditions[path.split("/")[-4]]
            ) for path in sorted(glob.glob(os.path.join(image_folder, "*/train/*/*.png")))
        ]
        self.images += [  # Add validation cities
            (
                path,
                os.path.join(
                    annotation_folder,
                    path.split("/")[-4],
                    path.split("/")[-3],
                    path.split("/")[-2],
                    path.split("/")[-1][:-12] + "gt_labelIds.png"
                ),
                weather_conditions[path.split("/")[-4]]
            ) for path in sorted(glob.glob(os.path.join(image_folder, "*/val/*/*.png")))
        ]

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
        except Exception as e:
            raise Exception(f"Index: {index}, len: {len(self)}, message: {str(e)}")

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class ACDC_Incremental(data.Dataset):
    """Labels correspond to domains not classes in this case."""
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=range(21),
        idxs_path=None,
        masking=True,
        overlap=True,
        **kwargs
    ):

        full_data = ACDC(root, train)
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
            apply_(lambda x: id_to_trainid.get(x, 255))
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