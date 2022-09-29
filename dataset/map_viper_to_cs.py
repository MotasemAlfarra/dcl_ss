import pandas as pd
import numpy as np
from copy import deepcopy

col_list = ['id','classname','red','green','blue','class_eval','trainid','instance_eval','cs_id']
df = pd.read_csv("dataset/classes.csv", usecols=col_list)

# ids = df['id'].tolist()
viper_trainid = df['trainid'].tolist()
cs_classes = df['cs_id'].tolist()
mapping_dict = {viper_trainid[i]: cs_classes[i] for i in range(len(viper_trainid))}

def convert_viper_to_cs(segmentation_mask):

    # Convert the segmentation mask to the Cityscapes format
    # The mapping is defined in the classes.csv file
    
    transformed_mask = deepcopy(segmentation_mask)
    for i in mapping_dict.keys():
        transformed_mask[segmentation_mask == i] = mapping_dict[i]

    return transformed_mask