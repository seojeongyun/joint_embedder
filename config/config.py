'''
    config.MODE
    config.EMB_MODE
    config.R_PRETRAINED
    config.B_PRETRAINED
    config.MODEL.OUT_CHANNELS
    config.MODEL.NUM_LAYER
    config.TRAIN.S
    config.TRAIN.M
'''
import numpy as np
from glob import glob
import re
from easydict import EasyDict as edict

config = edict()

config.MODE = 'VALID' # ['TRAIN',  'VALID']
config.EMB_MODE = 'B+R' # ['B+R', 'R', 'RID']
config.SAVE_ROOT = f"/home/jysuh/PycharmProjects/coord_embedding/checkpoint/{config.EMB_MODE}"

# PreTrained
config.R_PRETRAINED = True if config.MODE == 'VALID' else False
config.R_PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/B+R/[B+R] LAYERS_NUM:4 DIM:768 S:10 M:0.1/weights/metric_learning_model.pth.tar'
#
config.B_PRETRAINED = True if config.MODE == 'VALID' else False
config.B_PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/B+R/[B+R] LAYERS_NUM:4 DIM:768 S:10 M:0.1/weights/nn_embedding.pt'


# GPU / WORKERS
config.SEED = 42
config.GPUS = '1'
config.WORKERS = 0

# Dataset
config.DATASET = edict()
#
# coord_valid.json : for a workout, have many videos
# embedding_valid_data.json : for a workout, have one video
config.DATASET.TRAIN_DATA_PATH = '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/fitness_data_preprocess/ArcFace Dataset/TRAIN_ArcFace.pkl'
# config.DATASET.TRAIN_DATA_PATH = '/home/jysuh/PycharmProjects/coord_embedding/dataset/embedder_train.json'
config.DATASET.VALID_DATA_PATH = '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/fitness_data_preprocess/ArcFace Dataset/VALID_ArcFace.pkl'
#
config.DATASET.TARGET_SIZE = (1920, 1080)
config.DATASET.NUM_JOINTS = 20
config.DATASET.NUM_TOKEN = 2 # like cls, eos, sep, pad ..
config.DATASET.TRAIN_JOINT_VOCAB_PATH = '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/fitness_data_preprocess/vocab/TRAIN_joint_vocab.pkl'
config.DATASET.TRAIN_WORKOUT_VOCAB_PATH = '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/fitness_data_preprocess/vocab/TRAIN_workout_vocab.pkl'
config.DATASET.VALID_JOINT_VOCAB_PATH = '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/fitness_data_preprocess/vocab/VALID_joint_vocab.pkl'
config.DATASET.VALID_WORKOUT_VOCAB_PATH = '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/fitness_data_preprocess/vocab/VALID_workout_vocab.pkl'

# Model
config.MODEL = edict()
config.MODEL.IN_CHANNELS = 3 if config.EMB_MODE != 'R' else 2
config.MODEL.OUT_CHANNELS = 768 if not config.R_PRETRAINED else int(re.search(r'DIM:(\d+)', config.R_PRETRAINED_PATH).group(1))
config.MODEL.NUM_LAYER = 4 if not config.R_PRETRAINED else int(re.search(r'LAYERS_NUM:(\d+)', config.R_PRETRAINED_PATH).group(1)) # for train



# Train
config.TRAIN = edict()
config.TRAIN.EPOCH = 30
config.TRAIN.WARMUP_EPOCH = 3
config.TRAIN.BATCH_SIZE = 256  # during test, bs = 1
config.TRAIN.LR = 5e-5
config.TRAIN.ACT = 'GELU'  # ['ReLU', 'Mish' ... ]
config.TRAIN.S = 10 if not config.R_PRETRAINED else int(re.search(r'S:(\d+(?:\.\d+)?)', config.R_PRETRAINED_PATH).group(1))
config.TRAIN.M = 0.1 if not config.R_PRETRAINED else float(re.search(r'M:(\d+(?:\.\d+)?)', config.R_PRETRAINED_PATH).group(1))
config.TRAIN.SHUFFLE = True
config.TRAIN.NUM_SAMPLE = 2 # NUM_SAMPLE * 20(num_joint) * config.TRAIN.BATCH_SIZE
#
config.VALID = edict()
config.VALID.BATCH_SIZE = 256
config.VALID.NUM_SAMPLE = 100

# Print
config.PRINT_FREQ = 50

# Visualization
config.VIS = edict()
config.VIS.SAMPLES_PER_CLASS = 1440 # 현재 Validation 데이터셋 기준으로 MAX 값은 1440
config.VIS.TSNE_PERPLEXITY = 30
config.VIS.TSNE_N_ITER = 1000
config.VIS.TSNE_RANDOM_SEED = 42
config.VIS.PLOT_SAVE_ROOT = "./embedding_result_img"
config.VIS.PLOT_METRIC_METHOD = 'cosine'    # 'cosine' 'euclidean'
config.VIS.PLOT_VISUALIZATION = True
#

# Experiment
config.EXP = edict()
config.EXP.NUM_LAYERS = [6] # [2, 4, 6] # for find_optimal_config
config.EXP.ACT_LIST = ['ReLU', 'GELU']
config.EXP.S_RANGE = [50]
config.EXP.M_RANGE = [0.45]
config.EXP.EMB_DIM = [768]


config.FILE_NAME = (
    f'[{config.EMB_MODE}] '
    f'LAYERS_NUM:{config.MODEL.NUM_LAYER} '
    f'DIM:{config.MODEL.OUT_CHANNELS} '
    f'S:{config.TRAIN.S} '
    f'M:{config.TRAIN.M}'
)

# BEST Config
config.BEST = edict()
config.BEST.USE_EMB = True
config.BEST.NUM_LAYER = 6
config.BEST.OUT_CHANNELS = 768
config.BEST.ACT = 'GELU'
config.BEST.S = 10
config.BEST.M = 0.5
