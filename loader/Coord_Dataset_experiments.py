import os
import json
import tqdm
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm

class Coord_Dataset(Dataset):
    def __init__(self, config, mode, d_path, embedding_mode):
        self.config = config
        self.mode = mode
        self.d_path = d_path
        self.embedding_mode = embedding_mode
        self.data = self.get_data()
        print('NUMBER OF FRAMES:' + str(len(self.data)))

    def get_data(self):
        if self.mode == 'TRAIN':
            with open(self.d_path, 'rb') as f:
                data = pickle.load(f)

        elif self.mode == 'VALID':
            with open(self.d_path, 'rb') as f:
                data = pickle.load(f)

        else:
            raise ValueError(f'Not Implemented {self.mode}, please choose TRAIN or VALID')

        samples = [np.stack(list(sample.values()), axis=0) for sample in data]
        data_tensor = torch.from_numpy(np.stack(samples, axis=0)).float()

        return data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # [20, 3] = [NUM_JOINTS, (X, Y, JOINT_IDX/21)]
        if self.embedding_mode != 'RwID':
            return self.data[idx][:, :-1]
        else:
            return self.data[idx]
