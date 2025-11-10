import numpy as np
from glob import glob
from easydict import EasyDict as edict

config = edict()

config.GEN_BERT_DATASET = True

# GPU / WORKERS
config.SEED = 8438
config.GPUS = '0'
config.WORKERS = 0

# Dataset
config.DATASET = edict()
config.DATASET.TRAIN_DATA_PATH = '/home/jysuh/PycharmProjects/coord_embedding/dataset/embedding_train_data.json'

# coord_valid.json : for a workout, have many videos
# embedding_valid_data.json : for a workout, have one video
config.DATASET.VALID_DATA_PATH = '/home/jysuh/PycharmProjects/coord_embedding/dataset/coord_valid.json'
config.DATASET.NUM_JOINTS = 20
config.DATASET.NUM_TOKEN = 2 # like cls, eos, sep, pad ..
config.DATASET.TARGET_SIZE = (1920, 1080)

# Model
config.MODEL = edict()
config.MODEL.IN_CHANNELS = 4
config.MODEL.OUT_CHANNELS = 768     # 1024
config.MODEL.NUM_LAYERS = [2, 4, 6] # [2, 4, 6] # for find_optimal_config
config.MODEL.NUM_LAYER = 2  # for train

# Train
config.TRAIN = edict()
config.TRAIN.USE_EMB = True
config.TRAIN.USE_EMB_LIST = [True, False]
config.TRAIN.SHUFFLE = True
#
config.TRAIN.BATCH_SIZE = 64  # during test, bs = 1
config.TRAIN.LR = 5e-4
config.TRAIN.ACT = 'GELU'  # ['ReLU', 'Mish' ... ]
config.TRAIN.ACT_LIST = ['ReLU', 'GELU']
config.TRAIN.EPOCH = 30
config.TRAIN.WARMUP = True
config.TRAIN.WARMUP_EPOCH = 150
config.TRAIN.NUM_SAMPLE = 2 # NUM_SAMPLE * 20(num_joint) * config.TRAIN.BATCH_SIZE
#
config.TRAIN.S_RANGE = [10, 15, 20, 25, 30, 35, 40, 45, 50]
config.TRAIN.M_RANGE = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
config.TRAIN.EMB_DIM = [64, 128, 256, 512, 768, 1024]
config.TRAIN.LOSSES = ['CosFace', 'ArcFace']
config.TRAIN.LOSS = 'ArcFace'
#
config.VALID = edict()
config.VALID.BATCH_SIZE = 1
config.VALID.NUM_SAMPLE = 100

# Print
config.PRINT_FREQ = 50


# Visualization
config.VIS = edict()
config.VIS.MAX_SAMPLES = 40000
config.VIS.TSNE_PERPLEXITY = 30
config.VIS.TSNE_N_ITER = 1000
config.VIS.TSNE_RANDOM_SEED = 614
config.VIS.PLOT_SAVE_ROOT = "./embedding_result_img"
config.VIS.PLOT_METRIC_METHOD = 'cosine'    # 'cosine' 'euclidean'
config.VIS.PLOT_VISUALIZATION = True
#
config.FILE_NAME = '[PAD,SEP ' + f'{config.TRAIN.LOSS}' + ']:' \
                       + ' atfc:' + f'{config.TRAIN.ACT}' \
                       + ' use_basis:' + f'{config.TRAIN.USE_EMB}' \
                       + ' epoch:' + f'{config.TRAIN.EPOCH}' \
                       + ' layer_num:' + f'{config.MODEL.NUM_LAYER}' \
                       + ' out_dim:' + f'{config.MODEL.OUT_CHANNELS}'

config.BEST = edict()
config.BEST.USE_EMB = True
config.BEST.NUM_LAYER = 6
config.BEST.OUT_CHANNELS = 768
config.BEST.ACT = 'GELU'
config.BEST.S = 10
config.BEST.M = 0.5

# PreTrained
config.PRETRAINED = False
# config.PRETRAINED_PATH = f'/home/jysuh/PycharmProjects/coord_embedding/checkpoint/{config.FILE_NAME}/metric_learning_model.pth.tar'
config.PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/find_sm/[s,m Basis+Relative] LAYERS_NUM:6 DIM:768 ACT:GELU s:10 m:0.5/metric_learning_model.pth.tar'
#
# config.PRETRAINED_EMB = True
# config.PRETRAINED_EMB_PATH = f'/home/jysuh/PycharmProjects/coord_embedding/checkpoint/{config.FILE_NAME}/nn_embedding.pth.tar'

config.MODE = 'TRAIN' # ['TRAIN',  'TEST']