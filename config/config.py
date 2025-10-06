import numpy as np
from glob import glob
from easydict import EasyDict as edict

config = edict()

# GPU / WORKERS
config.SEED = 8438
config.GPUS = '0'
config.WORKERS = 0

# Dataset
config.DATASET = edict()
config.DATASET.TRAIN_DATA_PATH = glob('/home/jysuh/PycharmProjects/coord_embedding/dataset/*')
config.DATASET.VALID_DATA_PATH = '/home/jysuh/PycharmProjects/coord_embedding/dataset/coord_data_valid.json'
config.DATASET.NUM_JOINTS = 20

# Train
config.TRAIN = edict()
config.TRAIN.SHUFFLE = True
config.TRAIN.BATCH_SIZE = 64
config.TRAIN.LR = 1e-5
config.TRAIN.EPOCH = 100

#
config.TRAIN.S_RANGE = list(np.linspace(1,100, 100))
config.TRAIN.M_RANGE = [round(x, 2) for x in np.arange(0.05, 0.8 + 0.001, 0.05)]
config.TRAIN.EMB_DIM = [512, ]
config.TRAIN.LOSSES = ['CosFace', 'ArcFace']
config.TRAIN.LOSS = 'ArcFace'

# Print
config.PRINT_FREQ = 50

# PreTrained
config.PRETRAINED = True
config.PRETRAINED_EMB = False
config.PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/epoch_99.pth.tar'
config.PRETRAINED_EMB_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/kobart_embedding_weights.pt'
config.MODE = 'TEST' # ['TRAIN',  'TEST']


# layer7_mish_epoch_99.pth.tar : {'intra_class_distance': 31.6380558013916, 'inter_class_similarity': 0.8110062448601973, 'silhouette_score': 0.2400735765695572}

# no residual {'intra_class_distance': 30.436437606811523, 'inter_class_similarity': 0.770909921746505, 'silhouette_score': 0.2697477340698242}
            # {'intra_class_distance': 30.356884002685547, 'inter_class_similarity': 0.768890380859375, 'silhouette_score': 0.26593029499053955} Residual
            # {'intra_class_distance': 29.957530975341797, 'inter_class_similarity': 0.77132030286287, 'silhouette_score': 0.261474609375}       ReLU
            # {'intra_class_distance': 30.67803955078125, 'inter_class_similarity': 0.7919620714689556, 'silhouette_score': 0.2533227801322937}  LeakyReLU
            # {'intra_class_distance': 28.49051856994629, 'inter_class_similarity': 0.7323532907586349, 'silhouette_score': 0.2994067966938019}  Mish + AdmaW
            # {'intra_class_distance': 28.523162841796875, 'inter_class_similarity': 0.7323288766961349, 'silhouette_score': 0.29873597621917725} Mish
            # {'intra_class_distance': 28.204875946044922, 'inter_class_similarity': 0.734528953150699, 'silhouette_score': 0.30713480710983276}
            # {'intra_class_distance': 28.523162841796875, 'inter_class_similarity': 0.7323288766961349, 'silhouette_score': 0.29873597621917725}
            # {'intra_class_distance': 29.869495391845703, 'inter_class_similarity': 0.732811857524671, 'silhouette_score': 0.26701533794403076} SiLU
            # {'intra_class_distance': 28.725021362304688, 'inter_class_similarity': 0.7398005435341283, 'silhouette_score': 0.2918030023574829} GELU
            # {'intra_class_distance': 29.50107765197754, 'inter_class_similarity': 0.7517194245990954, 'silhouette_score': 0.28092506527900696} 10 epoch
            # {'intra_class_distance': 31.829431533813477, 'inter_class_similarity': 0.6846556814093339, 'silhouette_score': 0.2305298149585724} 10
            # {'intra_class_distance': 41.41999435424805, 'inter_class_similarity': 0.7561276887592516, 'silhouette_score': 0.10398957133293152}