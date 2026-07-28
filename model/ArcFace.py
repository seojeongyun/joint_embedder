'''
  0: PAD
  1: SEP
  2~21: 20개 관절

  ## 1. 일반 분류와 ArcFace의 차이

  일반적인 분류기는 임베딩과 class weight의 내적을 계산 / logit = embedding · class_weight
  이 값은 벡터의 크기와 각도 두 요소의 영향을 받음

  ArcFace는 embedding과 class weight를 모두 길이 1로 정규화
  그러면 내적은 두 벡터 사이의 cosine similarity가 됨

  정규화된 embedding · 정규화된 class weight = cos(theta)

  theta가 작음 → cosine이 큼 → 서로 가까운 방향
  theta가 큼   → cosine이 작음 → 서로 먼 방향

  ArcFace는 정답 클래스에만 추가 각도 m을 적용

  일반 cosine 분류: cos(theta)
  ArcFace 정답 점수: cos(theta + m)

  m을 더하면 정답 점수가 더 낮아지고 모델은 낮아진 점수로도 정답을 맞히기 위해 임베딩을 정답 class center에 더욱 가깝게 만들어야 함.

  같은 클래스 임베딩 → 더 조밀하게 모임 / 다른 클래스 임베딩 → 더 명확하게 분리됨


  ## 2. 22개 class center와 cosine 계산

  cosine = F.linear(
      F.normalize(flat_embedding, p=2, dim=1),
      F.normalize(self.weight, p=2, dim=1)
  )

  self.weight: [22, 768]

  각 행은 하나의 ArcFace class center

  weight[0]:  PAD center
  weight[1]:  SEP center
  weight[2]:  Joint 0 center
  ...
  weight[21]: Joint 19 center

  F.normalize()는 각각의 768차원 벡터 길이를 1로 만듦

  normalized_embedding = F.normalize(
      flat_embedding,
      p=2,
      dim=1
  )

  normalized_weight = F.normalize(
      self.weight,
      p=2,
      dim=1
  )

  그다음 F.linear()는 다음 연산을 합니다.

  [B*22, 768] × [768, 22]
  → [B*22, 22]

  결과: cosine: [B*22, 22]

  각 행에는 해당 token 임베딩과 22개 class center 사이의 cosine similarity가 들어갑니다.

  예를 들어 첫 번째 샘플이 PAD라면:

  cosine[0, 0]: PAD 임베딩과 PAD center의 유사도
  cosine[0, 1]: PAD 임베딩과 SEP center의 유사도
  cosine[0, 2]: PAD 임베딩과 Joint 0 center의 유사도
  ...

  ## 3. sin(theta) 계산

  sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(0.0))

  sin²(theta) + cos²(theta) = 1 이므로 sin(theta) = sqrt(1 - cos²(theta))이고,

  ArcFace는 cos(theta + m)을 계산해야 하므로 sin(theta)도 필요함.

  ## 4. 정답에만 margin 적용

  output = (
      one_hot * phi
      + (1.0 - one_hot) * cosine
  )

  이 코드가 ArcFace의 핵심

  정답 위치에서는:

  one_hot = 1

  output
  = 1 × phi + 0 × cosine
  = cos(theta + m)

  오답 위치에서는:

  one_hot = 0

  output
  = 0 × phi + 1 × cosine
  = cos(theta)

  따라서 샘플 하나의 최종 logits은 다음처럼 구성됩니다.

  정답 클래스: cos(theta + m)
  나머지 클래스: cos(theta)

  예를 들어 Joint 0의 정답 class가 2번이라면:

  class 0 PAD:     cos(theta_0)
  class 1 SEP:     cos(theta_1)
  class 2 Joint 0: cos(theta_2 + m)
  class 3 Joint 1: cos(theta_3)
  ...

  모델은 margin으로 인해 낮아진 정답 점수로도 Joint 0을 맞혀야 하므로 Joint 0 임베딩을 Joint 0 center에 더욱 가깝게 이동시
  킵니다.

  ## 12. Scale 적용

  scale = self.s if s is None else s
  output = output * scale

  정규화된 cosine의 범위는 -1~1로 작습니다. 이 값을 그대로 softmax에 넣으면 class 확률 차이가 충분히 커지지 않아 학습이 느
  려질 수 있습니다.

  그래서 s를 곱해 logit 범위를 확대합니다.

  예:

  cosine = 0.8
  s = 10

  scaled logit = 8.0

  m과 s의 역할은 서로 다릅니다.

  m: 클래스 사이의 각도 간격을 강제
  s: softmax에 들어가는 logit 크기를 확대

  s는 반드시 양수여야 합니다.

  s > 0

  s=0이면 모든 logits이 0이 되어 임베딩과 ArcFace center에 유효한 gradient가 전달되지 않습니다.
'''
import torch
import torch.nn.functional as F
import math

from math import pi
from torch import nn
from torchvision.models.vgg import make_layers


class ArcFace(nn.Module):
    def __init__(self, num_layer, in_features, out_features, num_class, embedding_mode, activation, easy_margin=False, device=torch.device("cpu")):
        super().__init__()
        self.num_layer = num_layer
        self.in_features = in_features
        self.out_features = out_features
        #
        self.embedding_mode = embedding_mode # B+R or R
        #
        self.weight = nn.Parameter(torch.empty(num_class,out_features,device=device))
        nn.init.xavier_normal_(self.weight)
        #
        self.embedding = nn.Embedding(num_embeddings=num_class, embedding_dim=out_features).to(device)
        #
        self.layers = nn.ModuleList(self.make_layer())
        #
        if activation == 'GELU':
            self.atfc = nn.GELU()
        else:
            self.atfc = nn.ReLU()

    def make_layer(self):
        layers = []
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

    def forward(self, input, token_ids, use_arcface, m=None, s=None):
        output = None
        BS = token_ids.shape[0]

        # [1] MLP 연산
        out = input.reshape(-1, self.in_features)

        # MLP Forwarding
        for i, layer in enumerate(self.layers):
            y = layer(out)
            if y.shape[-1] == out.shape[-1]:
                out = y + out
            else:
                out = y
            if i != len(self.layers) - 1:
                # out = nn.BatchNorm1d(out.shape[-1])(out)
                out = self.atfc(out)

        # (BS*20 ,768) -> (BS, 20, 768)
        out = out.reshape(BS, -1, self.out_features)

        # [2] nn.Embedding 연산
        embedding_vec = self.embedding(token_ids)
        special_emb = embedding_vec[:, :2, :]

        # Basis + Relative Mode
        if self.embedding_mode == 'B+R':
            joint_emb = embedding_vec[:, 2:, :] + out

        # Relative Mode
        else:
            joint_emb = out

        # BS, 22, 768
        embedding_vec = torch.cat([special_emb, joint_emb], dim=1)
        #

        if use_arcface:
            # token_ids: [B, 22]
            # [B*22]로 reshape
            labels = token_ids.reshape(-1).long()

            # [B*22, 768]
            flat_embedding = embedding_vec.reshape(-1, self.out_features)
            flat_embedding = flat_embedding.float()
            # [B*22, 22]
            cosine = F.linear(F.normalize(flat_embedding, p=2, dim=1, eps=1e-6), F.normalize(self.weight.float(), p=2, dim=1, eps=1e-6))

            cosine = cosine.clamp(-1.0 + 1e-5, 1.0 - 1e-5)

            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(0.0))
            #
            if m is not None:
                cos_m = math.cos(m)
                sin_m = math.sin(m)
                th = math.cos(math.pi - m)
                mm = math.sin(math.pi - m) * m

            phi = cosine * cos_m - sine * sin_m
            phi = torch.where(
                cosine > th,
                phi,
                cosine - mm
            )

            one_hot = torch.zeros_like(cosine)

            # [B*22, 22]
            one_hot.scatter_(dim=1, index=labels.unsqueeze(1), value=1.0 )
            output = (one_hot * phi + (1.0 - one_hot) * cosine)
            output = output * s

        return output, embedding_vec
