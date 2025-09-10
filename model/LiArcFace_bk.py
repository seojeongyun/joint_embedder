import torch
import torch.nn.functional as F
from math import pi
from torch import nn


class LiArcFace(nn.Module):
    def __init__(self, in_features, out_features, num_class, s=30.0, m=0.40, device=torch.device("cpu")):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_class, 512)).cuda()
        nn.init.xavier_normal_(self.weight)
        self.m = m
        self.s = s
        #
        self.embedding = nn.Embedding(num_embeddings=num_class, embedding_dim=out_features).to(device)
        #
        self.linear1 = nn.Linear(in_features, out_features//4, bias=True)
        self.linear2 = nn.Linear(out_features//4, out_features//2, bias=True)
        self.linear3 = nn.Linear(out_features//2, out_features, bias=True)
        self.linear4 = nn.Linear(out_features, out_features*2, bias=True)
        self.linear5 = nn.Linear(out_features*2, out_features*4, bias=True)
        self.linear6 = nn.Linear(out_features * 4, out_features*2, bias=True)
        self.linear7 = nn.Linear(out_features * 2, out_features, bias=False)
        #
        self.ReLU = nn.LeakyReLU()

    def forward(self, input, J_tokens, mode):
        emb_output_J_tokens = self.embedding(J_tokens)  # 4, 512
        output = None

        out = self.linear1(input)  # 4, 512
        out = self.ReLU(out)

        out = self.linear2(out)
        out = self.ReLU(out)

        out = self.linear3(out)
        out = self.ReLU(out)

        out = self.linear4(out)
        out = self.ReLU(out)

        out = self.linear5(out)
        out = self.ReLU(out)

        out = self.linear6(out)
        out = self.ReLU(out)

        out = self.linear7(out)
        out = self.ReLU(out)

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
