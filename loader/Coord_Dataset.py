import os
import json
import tqdm
import torch

from torch.utils.data import Dataset
from tqdm import tqdm

class Coord_Dataset(Dataset):
    def __init__(self, data_path):
        self.data = self.get_data(data_path=data_path)
        self.vocab = self.get_vocab()
        self.preprocessed_data = self.preprocess()
        print('NUMBER OF FRAMES:' + str(len(self.preprocessed_data)))

    def get_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def get_vocab(self):
        vocab = {}

        exercise_name = list(self.data.keys())[0]
        video_idx = list(self.data[exercise_name])[0]
        frame_idx = list(self.data[exercise_name][video_idx])[0]
        view_idx = list(self.data[exercise_name][video_idx][frame_idx])[0]

        for joint_name in self.data[exercise_name][video_idx][frame_idx][view_idx]:
            vocab.setdefault(joint_name, len(vocab))

        for exercise_name in self.data.keys():
            vocab.setdefault(exercise_name, len(vocab))

        return vocab

    def preprocess(self):
        processed_data = []
        for i, (exercise_name, exercise_name_token) in enumerate(self.vocab.items()):
            if i > 19: # 19 : the number of joints -> to use only exercise(i > 19) information
                for video_idx in self.data[exercise_name].keys():
                    for frame_idx in self.data[exercise_name][video_idx].keys():
                        for idx, view_idx in enumerate(self.data[exercise_name][video_idx][frame_idx].keys()):
                            sample = self.data[exercise_name][video_idx][frame_idx][view_idx]
                            for joint_name in sample.keys():
                                sample[joint_name] = sample[joint_name] + [self.vocab[joint_name]] + [exercise_name_token] + [int(frame_idx)] + [idx] + [int(video_idx)]
                            processed_data.append(sample) # sample = {'head' : [x,y]}, v = exercise_name
        return processed_data

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        # [20, 4] = [NUM_JOINTS, (X, Y, JOINT_IDX, EXERCISE_IDX)]
        joint_coord_info = self.preprocessed_data[idx]
        joint_coord = torch.tensor(list(joint_coord_info.values()), dtype=torch.float32)[:, :4]
        joint_indices = torch.tensor(list(joint_coord_info.values()), dtype=torch.float32)[:, 2].long()
        exercise_name = torch.tensor(list(joint_coord_info.values()), dtype=torch.int8)[:, 3][0]
        frame_idx = torch.tensor(list(joint_coord_info.values()), dtype=torch.int8)[:, 4][0]
        view_idx = torch.tensor(list(joint_coord_info.values()), dtype=torch.int8)[:, 5][0]
        video_idx = torch.tensor(list(joint_coord_info.values()), dtype=torch.int8)[:, 6][0]
        # joint_coord = torch.tensor(list(joint_coord_info.values()), dtype=torch.float32)[:, [0, 1, 3]]

        return joint_coord, joint_indices, exercise_name, frame_idx, view_idx, video_idx

    @staticmethod
    def collate_fn(batch):
        joint_coord, joint_indices, exercise_name, frame_idx, view_idx, video_idx = zip(*batch)

        X = torch.stack(joint_indices, dim=0)
        ref = torch.arange(X.size(1), dtype=X.dtype)
        valid_mask = (X == ref).all(dim=1)
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]

        joint_coord = torch.stack(joint_coord, dim=0)[valid_idx]
        joint_indices = torch.stack(joint_indices, dim=0)[valid_idx]
        exercise_name = torch.stack(exercise_name, dim=0)[valid_idx]
        frame_idx = torch.stack(frame_idx, dim=0)[valid_idx]
        view_idx = torch.stack(view_idx, dim=0)[valid_idx]
        video_idx = torch.stack(video_idx, dim=0)[valid_idx]

        return joint_coord, joint_indices, exercise_name, frame_idx, view_idx, video_idx

