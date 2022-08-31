import glob
import os
import cv2 as cv

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
# def labels_to_cityscapes_palette(image):
#     """
#     Convert an image containing CARLA semantic segmentation labels to
#     Cityscapes palette.
#     """
#     classes = {
#         0: [0, 0, 0],         # None
#         1: [70, 70, 70],      # Buildings
#         2: [190, 153, 153],   # Fences
#         3: [72, 0, 90],       # Other
#         4: [220, 20, 60],     # Pedestrians
#         5: [153, 153, 153],   # Poles
#         6: [157, 234, 50],    # RoadLines
#         7: [128, 64, 128],    # Roads
#         8: [244, 35, 232],    # Sidewalks
#         9: [107, 142, 35],    # Vegetation
#         10: [0, 0, 255],      # Vehicles
#         11: [102, 102, 156],  # Walls
#         12: [220, 220, 0]     # TrafficSigns
#     }
#     array = labels_to_array(image)
#     result = numpy.zeros((array.shape[0], array.shape[1], 3))
#     for key, value in classes.items():
#         result[numpy.where(array == key)] = value
#     return result

# weather_conditions = {
#     "fog": 0, "night": 1, "rain": 2, "snow": 3
# }

# Change to Town3 for the other town.
BASE_PATH = "/export/share/datasets/Carla_Dynamic/Town10_720p_V1/car_view/data/*/"
# out_recording_rgb_cam9_1280_720/*/*.jpg
class CARLA(data.Dataset):

    def __init__(self, root='', train=True, transform=None, domain_transform=None):
        # root = os.path.expanduser(root)
        rgb_paths = os.path.join(BASE_PATH, "*seg_cam9*")
        # seg_paths = os.path.join(BASE_PATH, "*seg_cam9*")

        self.images = [

        ]

        self.images = [  # Add train cities
            (
                path.replace('seg', 'rgb').replace('tif', 'jpg'), path
                # path, path.replace('rgb', 'seg').replace('jpg', 'tif')
                ,
                1 # weather_conditions[path.split("/")[-4]]
            ) for path in sorted(glob.glob(os.path.join(rgb_paths, "*/*.tif")))
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
            target = cv.imread(self.images[index][1])
            if img is None or target is None:
                print(index)
                print(self.images[index])
                print("Skip this batch")
            target = Image.fromarray(target[:, :, 2])
        except Exception as e:
            raise Exception(f"Index: {index}, len: {len(self)}, message: {str(e)}")

        if self.transform is not None:
            # print("I am here")
            # print(self.transform)
            # print(img.shape, target.shape)
            # img, target = self.transform(img), self.transform(target)
            # print("Shariq was here", self.transform)
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.images)


class CARLA_Incremental(data.Dataset):
    """Labels correspond to domains not classes in this case."""
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=range(23),
        idxs_path=None,
        masking=True,
        overlap=True,
        **kwargs
    ):

        full_data = CARLA(root, train)
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

        target_transform = tv.transforms.Lambda(
            lambda t: t.
            apply_(lambda x: id_to_trainid.get(x, 255))
        )
        # make the subset of the dataset
        self.dataset = Subset(full_data, idxs, transform, target_transform)
        # print("I am here")
        # print(transform)
        # print(target_transform)
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