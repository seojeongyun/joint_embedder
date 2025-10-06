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
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(20, out_features))
        nn.init.xavier_uniform_(self.weight)
        #
        self.embedding = nn.Embedding(num_embeddings=num_class, embedding_dim=512).to(device)
        #
        self.linear1 = nn.Linear(in_features, out_features//4)
        self.linear2 = nn.Linear(out_features//4, out_features//2)
        self.linear3 = nn.Linear(out_features//2, out_features)
        self.linear4 = nn.Linear(out_features, out_features*2)
        self.linear5 = nn.Linear(out_features*2, out_features*4)
        self.linear6 = nn.Linear(out_features * 4, out_features*2)
        self.linear7 = nn.Linear(out_features * 2, out_features)
        #
        # self.linear1 = nn.Linear(in_features, out_features//8)
        # self.linear2 = nn.Linear(out_features//8, out_features//4)
        # self.linear3 = nn.Linear(out_features//4, out_features//2)
        # self.linear4 = nn.Linear(out_features//2, out_features)
        # self.linear5 = nn.Linear(out_features, out_features * 2)
        # self.linear6 = nn.Linear(out_features * 2, out_features * 4)
        # self.linear7 = nn.Linear(out_features * 4, out_features * 8)
        # self.linear8 = nn.Linear(out_features * 8, out_features * 2)
        # self.linear9 = nn.Linear(out_features * 2, out_features)

        self.ReLU = nn.ReLU()

    def forward(self, input, J_tokens, mode):
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

        # out = self.linear8(out)
        # out = self.ReLU(out)
        #
        # out = self.linear9(out)
        # out = self.ReLU(out)

        emb_output_J_tokens = self.embedding(J_tokens)  # 4, 512
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