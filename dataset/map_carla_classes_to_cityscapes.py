# mapping from carla classes to cityscapes
carla_classes_to_cityscapes = {
    0: 255,  # 'unlabeled': 0,
    1: 2,  # 'building': 1,
    2: 4,  # 'fence': 2,
    3: 255,  # 'other': 3,
    4: 11,  # 'pedestrian': 4,
    5: 5,  # 'pole': 5,
    6: 0,  # 'road line': 6,
    7: 0,  # 'road': 7,
    8: 1,  # 'sidewalk': 8,
    9: 8,  # 'vegetation': 9,
    10: 13,  # 'car': 10,
    11: 255,  # 'wall': 11,
    12: 7,  # 'traffic sign': 12,
    13: 10,  # 'sky': 13,
    14: 255,  # 'ground': 14,
    15: 255,  # 'bridge': 15,
    16: 255,  # 'rail track': 16,
    17: 255,  # 'guard rail': 17,
    18: 6,  # 'traffic light': 18,
    19: 255,  # 'static': 19,
    20: 255,  # 'dynamic': 20,
    21: 255,  # 'water': 21,
    22: 9,  # 'terrain': 22,
}


def transform_CARLA_Segmentation_mask_to_CityScapes_Segmentation_mask(segmentation_mask):
    '''
    This functions recieves CARLA segmentation mask
    and returns the segmentation map in CityScapes format
    '''
    
    for i in carla_classes_to_cityscapes.keys():
        segmentation_mask[segmentation_mask == i] = carla_classes_to_cityscapes[i]

    return segmentation_mask