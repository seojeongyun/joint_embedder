import os
import pprint

import torch
import yaml
import pickle
import random
import numpy as np

from torch import nn
from tqdm import tqdm
from pprint import pprint
from types import SimpleNamespace
from Embedder.Embedder_config import config
from collections import OrderedDict
from Embedder.new_video_loader import Video_Loader


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, SimpleNamespace):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

class Embedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #
        self.vocab = self.get_vocab()
        #
        self.in_features = self.config.IN_FEAT
        self.out_features = self.config.OUT_FEAT
        self.num_layer = self.config.NUM_LAYER
        #
        self.use_embedding = self.config.USE_EMBEDDING
        #
        self.embedding = nn.Embedding(num_embeddings=self.config.NUM_JOINTS, embedding_dim=self.config.OUT_FEAT).to(self.config.DEVICE)
        self.load_state_dict_embedding()
        #
        self.layers = nn.ModuleList(self.make_layer())
        self.load_state_dict_linear()
        #
        if self.config.ACTIV == 'GELU':
            self.atfc = nn.GELU()
        else:
            self.atfc = nn.ReLU()

    def get_vocab(self):
        with open(config.VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)

        return vocab

    def load_state_dict_embedding(self):
        if self.config.USE_EMBEDDING:
            if os.path.isfile(self.config.PRETRAINED_EMB_PATH):
                pretrained_emb_weight = torch.load(self.config.PRETRAINED_EMB_PATH)
                self.embedding.weight.data.copy_(pretrained_emb_weight['weight'])
                if config.BASIS_FREEZE:
                    self.embedding.weight.requires_grad = False

                self.embedding.to(self.config.DEVICE)
                pprint('Pretrained embedding weights loaded successfully.')

            else:
                raise ValueError("NOT EXIST PRETRAINED EMBEDDING WEIGHT PATH")

        else:
            pass

    def load_state_dict_linear(self):
        if os.path.isfile(self.config.PRETRAINED_PATH):
            pretrained_linear_weight = torch.load(self.config.PRETRAINED_PATH, map_location=self.config.DEVICE)
            #
            del pretrained_linear_weight['embedding.weight']
            #
            state_dict = OrderedDict()
            #
            for param in pretrained_linear_weight.keys():
                if param.startswith('layers.'):
                    state_dict[param[7:]] = pretrained_linear_weight[param]
                    # if config.RELATIVE_FREEZE:
                    #     state_dict[param[7:]].requires_grad = False
                else:
                    state_dict[param] = pretrained_linear_weight[param]
                    # if config.RELATIVE_FREEZE:
                    #     state_dict[param].requires_grad = False
            self.layers.load_state_dict(state_dict, strict=False)

            if config.RELATIVE_FREEZE:
                for name, p in self.layers.named_parameters():
                    p.requires_grad = False

            self.layers.to(self.config.DEVICE)
            pprint('Pretrained linear weights loaded successfully.')

        else:
            raise ValueError('NOT EXIST PRETRAINED LINEAR WEIGHT PATH')

    def make_layer(self):
        layers = []
        #
        if self.num_layer == 2:
            layers.append(nn.Linear(self.in_features, self.out_features//2, bias=True))
            layers.append(nn.Linear(self.out_features//2, self.out_features, bias=False))
        elif self.num_layer == 4:
            layers.append(nn.Linear(self.in_features, self.out_features//4, bias=True))
            layers.append(nn.Linear(self.out_features//4, self.out_features//2, bias=True))
            layers.append(nn.Linear(self.out_features//2, self.out_features//4, bias=True))
            layers.append(nn.Linear(self.out_features//4, self.out_features, bias=False))
        elif self.num_layer == 6:
            layers.append(nn.Linear(self.in_features, self.out_features//8, bias=True))
            layers.append(nn.Linear(self.out_features//8, self.out_features//4, bias=True))
            layers.append(nn.Linear(self.out_features//4, self.out_features//2, bias=True))
            layers.append(nn.Linear(self.out_features//2, self.out_features//4, bias=True))
            layers.append(nn.Linear(self.out_features//4, self.out_features//2, bias=True))
            layers.append(nn.Linear(self.out_features//2, self.out_features, bias=False))
        return layers

    def preprocess_joint_info(self, videos, frame_idx, joint_name):
        # Collect joint information from (BS) videos loaded by torch dataloader,
        # which results in a data with the shape of [BS, 4] (joint per frame parallel)
        #
        BS = len(videos)
        joint_info_cpu = torch.empty((BS, 4), dtype=torch.float32)
        joint_token_cpu = torch.empty((BS,), dtype=torch.long)

        for i in range(BS):
            joint_info_cpu[i] = torch.as_tensor(videos[i][str(frame_idx)][joint_name],dtype=torch.float32)
            joint_token_cpu[i] = self.vocab[joint_name]

        joint_info = joint_info_cpu.to(self.config.DEVICE, non_blocking=True)
        joint_token = joint_token_cpu.to(self.config.DEVICE, non_blocking=True)

        return joint_info, joint_token

    def forward_propagation(self, joint_info, joint_token):
        BS = joint_info.size(0)
        device = joint_info.device
        #
        embedding_vec = joint_info.new_zeros(BS, self.out_features)
        #
        if self.use_embedding:
            emb_output_J_tokens = self.embedding(joint_token)  # [BS, OUT_FEAT]

        pad_mask = torch.all(joint_info == 0, dim=-1)  # [BS]
        #

        if (~pad_mask).any(): # not pad
            out = joint_info[~pad_mask]
            if self.use_embedding:
                out_J_token = emb_output_J_tokens[~pad_mask]

            for i, layer in enumerate(self.layers):
                y = layer(out)
                if y.shape[-1] == out.shape[-1]:
                    out = y + out
                else:
                    out = y
                if i != len(self.layers) - 1:
                    out = self.atfc(out)

            if self.use_embedding:
                out = out + out_J_token
            #
            embedding_vec[~pad_mask] = out

        if pad_mask.any():
            pad_emb = self.embedding(torch.zeros(pad_mask.sum(), dtype=torch.long, device=device))
            embedding_vec[pad_mask] = pad_emb

        return embedding_vec

    def forward(self, videos):
        # consider batch
        BS = len(videos)
        T = self.config.MAX_FRAMES
        J = len(self.config.JOINTS_NAME)
        D = self.out_features

        outputs = torch.empty((BS, T, J, D), device=self.config.DEVICE, dtype=torch.float32)

        for frame_idx in range(T):
            for j, joint_name in enumerate(self.config.JOINTS_NAME):
                joint_info, joint_token = self.preprocess_joint_info(videos, frame_idx, joint_name)
                vec = self.forward_propagation(joint_info, joint_token)
                outputs[:, frame_idx, j, :] = vec

        return outputs


if __name__ == '__main__':
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    #
    video_dataset = Video_Loader(config=config)
    video_loader = torch.utils.data.DataLoader(
        video_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        collate_fn=video_dataset.collate_fn
    )

    #
    embedder = Embedder(config)
    for i, (videos, exercise_class) in enumerate(tqdm(video_loader, desc='embedding', total=len(video_loader))):
        output = embedder(videos)
        print()
        # BERTSUM(output)
    print()

    # with open('/home/jysuh/PycharmProjects/coord_embedding/dataset/embedder_dataset/valid.pkl', 'wb') as f:
    #     pickle.dump(lst, f)