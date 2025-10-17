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
config.DATASET.VALID_DATA_PATH = '/home/jysuh/PycharmProjects/coord_embedding/dataset/embedding_valid_data.json'
config.DATASET.NUM_JOINTS = 20


# Train
config.TRAIN = edict()
config.TRAIN.USE_EMB = False
config.TRAIN.USE_EMB_LIST = [False, True]
config.TRAIN.SHUFFLE = True
#
config.TRAIN.BATCH_SIZE = 64  # during test, bs = 1
config.TRAIN.LR = 5e-4
config.TRAIN.ACT = 'GELU'  # ['ReLU', 'Mish' ... ]
config.TRAIN.ACT_LIST = ['ReLU', 'GELU']
config.TRAIN.EPOCH = 1000
config.TRAIN.WARMUP = True
config.TRAIN.WARMUP_EPOCH = 250
#
config.TRAIN.NUM_SAMPLE = 2 # NUM_SAMPLE * 20(num_joint) * config.TRAIN.BATCH_SIZE


config.VALID = edict()
config.VALID.BATCH_SIZE = 1
config.VALID.NUM_SAMPLE = 100
#
config.TRAIN.S_RANGE = list(np.linspace(1,100, 100))
config.TRAIN.M_RANGE = [round(x, 2) for x in np.arange(0.05, 0.8 + 0.001, 0.05)]
config.TRAIN.EMB_DIM = [64, 128, 256, 512]
config.TRAIN.LOSSES = ['CosFace', 'ArcFace']
config.TRAIN.LOSS = 'ArcFace'

# Print
config.PRINT_FREQ = 50


# Visualization
config.VIS = edict()
config.VIS.MAX_SAMPLES = 40000
config.VIS.TSNE_PERPLEXITY = 40
config.VIS.TSNE_N_ITER = 1000
config.VIS.TSNE_RANDOM_SEED = 614
config.VIS.PLOT_SAVE_ROOT = "./embedding_result_img"
config.VIS.PLOT_METRIC_METHOD = 'cosine'    # 'cosine' 'euclidean'
config.VIS.PLOT_VISUALIZATION = True
#
config.FILE_NAME = '[' + f'{config.TRAIN.LOSS}' + ']:' \
                       + ' activation:' + f'{config.TRAIN.ACT}' \
                       + ' use_emb:' + f'{config.TRAIN.USE_EMB}' \
                       + ' total epoch:' + f'{config.TRAIN.EPOCH}' \
                       + ' warmup:' + f'{config.TRAIN.WARMUP}' \
                       + ' max_iter:' + f'{config.VIS.TSNE_N_ITER}' \
                       + ' perplexity:' + f'{config.VIS.TSNE_PERPLEXITY}'

# PreTrained
config.PRETRAINED = False
config.PRETRAINED_PATH = f'/home/jysuh/PycharmProjects/coord_embedding/checkpoint/{config.FILE_NAME}.pth.tar'
#
# config.PRETRAINED_EMB = False
# config.PRETRAINED_EMB_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/kobart_embedding_weights.pt'

config.MODE = 'TRAIN' # ['TRAIN',  'TEST']