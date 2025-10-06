import torch
import torch.nn.functional as F
from math import pi
from torch import nn


class LiArcFace(nn.Module):
    def __init__(self, in_features, out_features, num_class, s=30.0, m=0.40, device=torch.device("cpu")):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        #
        self.layer = nn.ModuleList(self.make_layer())
        self.embedding = nn.Embedding(num_embeddings=num_class, embedding_dim=self.out_features).to(device)
        #
        self.m = m
        self.s = s
        #
        self.weight = nn.Parameter(torch.empty(num_class, out_features)).cuda()
        nn.init.xavier_normal_(self.weight)
        #
        self.atfc = nn.Mish()

    def determine_layer_count(self):
        self.ratio = self.out_features / self.in_features
        #
        if self.ratio <= 8: return 1
        elif self.ratio <= 16: return 3
        elif self.ratio <= 64: return 5
        else: return 7

    def make_layer(self):
        layers = []
        num_layers = self.determine_layer_count()

        if num_layers == 1:
            layers.append(nn.Linear(self.in_features, self.out_features, bias=False))

        elif num_layers == 3:
            layers.append(nn.Linear(self.in_features, 32, bias=True))
            layers.append(nn.Linear(32, 128, bias=True))
            layers.append(nn.Linear(128, self.out_features, bias=False))

        elif num_layers == 5:
            layers.append(nn.Linear(self.in_features, 32, bias=True))
            layers.append(nn.Linear(32, 64, bias=True))
            layers.append(nn.Linear(64, 256, bias=True))
            layers.append(nn.Linear(256, 128, bias=True))
            layers.append(nn.Linear(128, self.out_features, bias=False))

        elif num_layers == 7:
            layers.append(nn.Linear(self.in_features, 32, bias=True))
            layers.append(nn.Linear(32, 64, bias=True))
            layers.append(nn.Linear(64, 128, bias=True))
            layers.append(nn.Linear(128, 256, bias=True))
            layers.append(nn.Linear(256, 512, bias=True))
            layers.append(nn.Linear(512, 256, bias=True))
            layers.append(nn.Linear(256, self.out_features, bias=False))

        return layers

    def forward(self, input, J_tokens, mode):
        output = None
        emb_output_J_tokens = self.embedding(J_tokens)  # [B, out_feature]
        out = input

        for i, layer in enumerate(self.layer):
            out = layer(out)
            if i != len(self.layer) - 1:
                out = self.atfc(out)

        embedding_vec = out + emb_output_J_tokens

        #
        if mode == 'training':
            label = J_tokens[0]
            W = F.normalize(self.weight)
            input = F.normalize(embedding_vec)
            cosine = input @ W.t()
            theta = torch.acos(cosine)
            m = torch.zeros_like(theta)
            m.scatter_(1, label.view(-1, 1), self.m)
            # output = self.s * (pi - 2 * (theta + m)) / pi
            output = self.s * torch.cos(theta + m)

        return output, embedding_vec
