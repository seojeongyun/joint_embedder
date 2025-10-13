import torch
import torch.nn.functional as F
#
from torch import nn


class CosFace(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, num_class, only_metric, activation, s=30.0, m=0.40, device=torch.device("cpu")):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_class, out_features))
        nn.init.xavier_uniform_(self.weight)
        #
        self.only_metric = only_metric
        #
        self.embedding = nn.Embedding(num_embeddings=num_class, embedding_dim=out_features).to(device)
        #
        self.layers = nn.ModuleList(self.make_layer())

        if activation == 'GELU':
            self.atfc = nn.GELU()
        else:
            self.atfc = nn.Mish()

    def make_layer(self):
        layers = []
        layers.append(nn.Linear(self.in_features, 32, bias=True))
        layers.append(nn.Linear(32, 128, bias=True))
        layers.append(nn.Linear(128, 256, bias=True))
        layers.append(nn.Linear(256, 128, bias=True))
        layers.append(nn.Linear(128, 256, bias=True))
        layers.append(nn.Linear(256, self.out_features, bias=False))
        return layers


    def forward(self, input, J_tokens, mode,  m=None, s=None):
        output = None
        emb_output_J_tokens = self.embedding(J_tokens)  # 4, 512
        out = input

        for i, layer in enumerate(self.layers):
            y = layer(out)
            if y.shape[-1] == out.shape[-1]:
                out = y + out
            else:
                out = y
            if i != len(self.layers) - 1:
                # out = nn.BatchNorm1d(out.shape[-1])(out)
                out = self.atfc(out)

        if self.only_metric:
            embedding_vec = out
        else:
            embedding_vec = out + emb_output_J_tokens

        if mode == 'training':
            label = J_tokens

            # --------------------------- cos(theta) & phi(theta) ---------------------------
            cosine = F.linear(F.normalize(embedding_vec), F.normalize(self.weight))
            phi = cosine - self.m

            # --------------------------- convert label to one-hot ---------------------------
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s

        return output, embedding_vec