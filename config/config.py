import numpy as np
from easydict import EasyDict as edict

config = edict()

# GPU / WORKERS
config.SEED = 8438
config.GPUS = '0'
config.WORKERS = 0

# Dataset
config.DATASET = edict()
config.DATASET.TRAIN_PATH = '/home/jysuh/PycharmProjects/coord_embedding/dataset/coord_data_valid.json'
config.DATASET.NUM_JOINTS = 20

# Train
config.TRAIN = edict()
config.TRAIN.SHUFFLE = True
config.TRAIN.BATCH_SIZE = 64
config.TRAIN.LR = 1e-5
config.TRAIN.EPOCH = 100

#
config.TRAIN.S_RANGE = list(np.linspace(1,64, 64))
config.TRAIN.M_RANGE = list(np.arange(0.05, 0.5 + 0.001, 0.05))
config.TRAIN.EMB_DIM = list(range(8, 513, 8))
config.TRAIN.LOSSES = ['CosFace', 'ArcFace']
config.TRAIN.LOSS = 'ArcFace'
# Print
config.PRINT_FREQ = 50

# PreTrained
config.PRETRAINED = None # '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/epoch_99.pth.tar'
config.MODE = 'TRAIN' # ['TRAIN',  'TEST']


# layer7_mish_epoch_99.pth.tar : {'intra_class_distance': 31.6380558013916, 'inter_class_similarity': 0.8110062448601973, 'silhouette_score': 0.2400735765695572}

# no residual {'intra_class_distance': 30.436437606811523, 'inter_class_similarity': 0.770909921746505, 'silhouette_score': 0.2697477340698242}
            # {'intra_class_distance': 30.356884002685547, 'inter_class_similarity': 0.768890380859375, 'silhouette_score': 0.26593029499053955}
            # {'intra_class_distance': 29.957530975341797, 'inter_class_similarity': 0.77132030286287, 'silhouette_score': 0.261474609375}
            # {'intra_class_distance': 30.67803955078125, 'inter_class_similarity': 0.7919620714689556, 'silhouette_score': 0.2533227801322937}