import torch
import torch.nn.functional as F

from torch import nn

class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, num_class, s=30.0, m=0.40, device=torch.device("cpu")):
        super(AddMarginProduct, self).__init__()
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
        if self.ratio <= 8:
            return 1
        elif self.ratio <= 16:
            return 3
        elif self.ratio <= 64:
            return 5
        else:
            return 7

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

        if mode == 'training':
            label = J_tokens[0]

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

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', s=' + str(self.s) \
            + ', m=' + str(self.m) + ')'