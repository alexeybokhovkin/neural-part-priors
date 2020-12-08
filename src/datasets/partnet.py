import os
import pickle
import json

import numpy as np
import torch
from torch.utils.data import Dataset

class VoxelPartnetAllShapesDataset(Dataset):

    def __init__(self, root, partnet_to_dirs_path, object_list):

        self.root = root

        with open(partnet_to_dirs_path, 'rb') as f:
            self.partnet_to_dirs = pickle.load(f)

        if isinstance(object_list, str):
            with open(os.path.join(self.root, object_list), 'r') as f:
                self.object_names = [item.rstrip() for item in f.readlines()]
        else:
            self.object_names = object_list

    def __getitem__(self, index):
        partnet_id = self.object_names[index]
        common_path = self.partnet_to_dirs[partnet_id]

        geo_fn = os.path.join(common_path+'_geo', f'{partnet_id}_full.npy')
        shape_mask = torch.FloatTensor(np.load(geo_fn))

        output = (shape_mask)

        return output

    def __len__(self):
        return len(self.object_names)