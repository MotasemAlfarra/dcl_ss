# mapping from carla classes to cityscapes
# carla_classes_to_cityscapes = {
#     0: 255,  # 'unlabeled': 0,
#     1: 2,  # 'building': 1,
#     2: 4,  # 'fence': 2,
#     3: 255,  # 'other': 3,
#     4: 11,  # 'pedestrian': 4,
#     5: 5,  # 'pole': 5,
#     6: 0,  # 'road line': 6,
#     7: 0,  # 'road': 7,
#     8: 1,  # 'sidewalk': 8,
#     9: 8,  # 'vegetation': 9,
#     10: 13,  # 'car': 10,
#     11: 255,  # 'wall': 11,
#     12: 7,  # 'traffic sign': 12,
#     13: 10,  # 'sky': 13,
#     14: 255,  # 'ground': 14,
#     15: 255,  # 'bridge': 15,
#     16: 255,  # 'rail track': 16,
#     17: 255,  # 'guard rail': 17,
#     18: 6,  # 'traffic light': 18,
#     19: 255,  # 'static': 19,
#     20: 255,  # 'dynamic': 20,
#     21: 255,  # 'water': 21,
#     22: 9,  # 'terrain': 22,
# }
carla_classes_to_cityscapes = {
    0: -1,  # 'unlabeled': 0,
    1: 11,  # 'building': 1,
    2: 13,  # 'fence': 2,
    3: -1,  # 'other': 3,
    4: 24,  # 'pedestrian': 4,
    5: 17,  # 'pole': 5,
    6: 7,  # 'road line': 6,
    7: 7,  # 'road': 7,
    8: 8,  # 'sidewalk': 8,
    9: 21,  # 'vegetation': 9,
    10: 26,  # 'car': 10,
    11: -1,  # 'wall': 11,
    12: 20,  # 'traffic sign': 12,
    13: 23,  # 'sky': 13,
    14: -1,  # 'ground': 14,
    15: -1,  # 'bridge': 15,
    16: -1,  # 'rail track': 16,
    17: -1,  # 'guard rail': 17,
    18: 19,  # 'traffic light': 18,
    19: -1,  # 'static': 19,
    20: -1,  # 'dynamic': 20,
    21: -1,  # 'water': 21,
    22: 22,  # 'terrain': 22,
}
# id_to_trainid = {
#     0: 255,
#     1: 255,
#     2: 255,
#     3: 255,
#     4: 255,
#     5: 255,
#     6: 255,
#     7: 0,   # road
#     8: 1,   # sidewalk
#     9: 255,
#     10: 255,
#     11: 2,  # building
#     12: 3,  # wall
#     13: 4,  # fence
#     14: 255,
#     15: 255,
#     16: 255,
#     17: 5,  # pole
#     18: 255,
#     19: 6,  # traffic light
#     20: 7,  # traffic sign
#     21: 8,  # vegetation
#     22: 9,  # terrain
#     23: 10, # sky
#     24: 11, # person
#     25: 12, # rider
#     26: 13, # car
#     27: 14, # truck
#     28: 15, # bus
#     29: 255,
#     30: 255,
#     31: 16, # train
#     32: 17, # motorcycle
#     33: 18, # bicycle
#     -1: 255
# }

def transform_CARLA_Segmentation_mask_to_CityScapes_Segmentation_mask(segmentation_mask):
    '''
    This functions recieves CARLA segmentation mask
    and returns the segmentation map in CityScapes format
    '''
    
    for i in carla_classes_to_cityscapes.keys():
        segmentation_mask[segmentation_mask == i] = carla_classes_to_cityscapes[i]

    return segmentation_mask