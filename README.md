# Joint Embedder

## Overview

Human Pose Estimation으로 추출한 **관절 좌표를 고차원 임베딩으로 변환하는 Joint Embedder**를 구현한 프로젝트입니다.

관절 좌표와 Skeleton 구조 정보를 활용하여 관절별 representation을 생성하고, **동일 종류의 관절은 임베딩 공간에서 가깝게, 서로 다른 관절은 멀어지도록 학습**하는 것을 목표로 합니다.

학습된 Joint Embedding은 운동 종류 및 자세 분석과 같은 **Human Motion Understanding**의 입력 representation으로 활용합니다.

---

## Architecture

<p align="center">
  <img src="./assets/joint_embedder_architecture.png" width="100%">
</p>

### Joint Representation

각 관절의 좌표와 구조적 특징을 MLP에 입력하여 **768차원의 Relative Embedding**을 생성합니다.

* Joint Coordinate `(x, y)`
* Relative Vector
* Graph Neighbor
* Skeleton Edge

관절 종류별 **Learnable Basis Embedding**과 Relative Embedding을 결합하여 최종 Joint Embedding을 생성합니다.

$$
E_i = B_i + R_i
$$

### Metric Learning

ArcFace 기반 Metric Learning을 적용하여 Joint Embedding Space를 학습합니다.

* **동일 종류의 관절** → 임베딩 간 거리 감소
* **서로 다른 종류의 관절** → 임베딩 간 거리 증가

이를 통해 단순 좌표값을 관절 종류와 구조적 특징을 반영하는 고차원 representation으로 변환합니다.

---

## Training

| Parameter           |   Value |
| ------------------- | ------: |
| Human Joints        |      20 |
| Embedding Dimension |     768 |
| MLP Layers          |       4 |
| Batch Size          |     256 |
| Optimizer           |    Adam |
| Learning Rate       |  `5e-5` |
| Metric Learning     | ArcFace |

---

## Evaluation

학습된 Joint Embedding Space의 군집화 성능을 다음 지표를 통해 분석합니다.

* **t-SNE** — 관절 종류별 embedding 분포 시각화
* **Silhouette Score** — 군집 내부 응집도 및 군집 간 분리도 평가
* **Dunn Index** — 군집 간 분리도 평가
* **Cosine Similarity** — 관절별 centroid 간 유사도 분석

---

## Repository Structure

```text
joint_embedder/
├── Embedder/       # Pretrained Joint Embedder API
├── config/         # Training configuration
├── loader/         # Dataset loader
├── model/          # Joint Embedder & ArcFace
├── utils/          # Evaluation & visualization
├── core.py         # Main training pipeline
└── core_experiments.py
```

---

## Tech Stack

`Python` · `PyTorch` · `NumPy` · `scikit-learn` · `Matplotlib`
