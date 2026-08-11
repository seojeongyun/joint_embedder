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
import os
import numpy as np
from glob import glob
import re
from easydict import EasyDict as edict

def parse_arcface_parameter(path, parameter_name):
    pattern = rf'(?:^|[\s/]){re.escape(parameter_name)}:(\d+(?:\.\d+)?)(?=$|[\s/])'
    match = re.search(pattern, path)

    if match is None:
        raise ValueError(
            f"ArcFace parameter '{parameter_name}'을 경로에서 찾을 수 없습니다: {path}"
        )

    return float(match.group(1))

config = edict()
config.FEATURE_MODES = 'VECTOR_SKELETON_EDGE_1HOP' # 'XY', 'VECTOR', 'GRAPH_1HOP', 'GRAPH_2HOP', 'VECTOR_GRAPH_1HOP', 'VECTOR_GRAPH_2HOP', 'SKELETON_EDGE_1HOP', 'VECTOR_SKELETON_EDGE_1HOP'

# MODE가 VALID면 PRETRAINED는 True가 되고, Pretrained path의 경로에 맞추어 입출력 차원, 계층 수, s, m 등이 결정된다.
config.MODE = 'TRAIN' # ['TRAIN',  'VALID']

# Data Version
IS_CONTAIN_HARD_EXERCISE = True # False
IS_INTEGRATE_ROW = False  # False
IS_JOINT_ID_NORM = True

ROW_MODE_DIR = (
    'row_integrated'
    if IS_INTEGRATE_ROW
    else 'row_not_integrated'
)
HARD_EXERCISE_MODE_DIR = (
    'hard_exercise_included'
    if IS_CONTAIN_HARD_EXERCISE
    else 'hard_exercise_excluded'
)
JOINT_ID_MODE_DIR = (
    'joint_id_normalized'
    if IS_JOINT_ID_NORM
    else 'joint_id_raw'
)

if IS_CONTAIN_HARD_EXERCISE and not IS_INTEGRATE_ROW:
    DATASET_NAME = 'Original_Data'
elif not IS_CONTAIN_HARD_EXERCISE and IS_INTEGRATE_ROW:
    DATASET_NAME = 'HardExcluded_RowIntegrated'
else:
    DATASET_NAME = (
        f'{HARD_EXERCISE_MODE_DIR}_'
        f'{ROW_MODE_DIR}'
    )

# PreTrained
config.R_PRETRAINED = True if config.MODE == 'VALID' else False
config.R_PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/GRAPH_2HOP/RwID/[RwID] LAYERS_NUM:4 IN_DIM:7 DIM:768 S:10 M:0.1/weights/metric_learning_model.pth.tar'
#
config.B_PRETRAINED = True if config.MODE == 'VALID' else False
config.B_PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/GRAPH_2HOP/RwID/[RwID] LAYERS_NUM:4 IN_DIM:7 DIM:768 S:10 M:0.1/weights/nn_embedding.pt'

config.EMB_MODE = 'B+R' if config.MODE == 'TRAIN' else config.R_PRETRAINED_PATH.split('/')[7] # ['B+R', 'R', 'RwID', 'RwIDNorm']
config.SAVE_ROOT = f"/home/jysuh/PycharmProjects/coord_embedding/checkpoint/{config.FEATURE_MODES}/{config.EMB_MODE}"

# GPU / WORKERS
config.SEED = 42
config.GPUS = '0'
config.WORKERS = 0

# Dataset
ARCFACE_DATASET_ROOT = (
    '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/'
    'fitness_data_preprocess/ArcFace Dataset'
)
VOCAB_ROOT = (
    '/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/'
    'fitness_data_preprocess/vocab'
)

config.DATASET = edict()
# 학습 데이터의 경우 운동별 5개 video * 5view가 포함, 검증 데이터는 운동별 1개 video * 5 view
config.DATASET.NAME = DATASET_NAME
config.DATASET.TRAIN_DATA_PATH = os.path.join(
    ARCFACE_DATASET_ROOT,
    config.FEATURE_MODES,
    ROW_MODE_DIR,
    HARD_EXERCISE_MODE_DIR,
    JOINT_ID_MODE_DIR,
    'TRAIN.pkl',
)
config.DATASET.VALID_DATA_PATH = os.path.join(
    ARCFACE_DATASET_ROOT,
    config.FEATURE_MODES,
    ROW_MODE_DIR,
    HARD_EXERCISE_MODE_DIR,
    JOINT_ID_MODE_DIR,
    'VALID.pkl',
)

config.DATASET.NUM_JOINTS = 20
config.DATASET.NUM_TOKEN = 2 # SEP, PAD
config.DATASET.TRAIN_JOINT_VOCAB_PATH = os.path.join(
    VOCAB_ROOT, ROW_MODE_DIR, HARD_EXERCISE_MODE_DIR,
    'TRAIN', 'joint_vocab.pkl',
)
config.DATASET.TRAIN_WORKOUT_VOCAB_PATH = os.path.join(
    VOCAB_ROOT, ROW_MODE_DIR, HARD_EXERCISE_MODE_DIR,
    'TRAIN', 'workout_vocab.pkl',
)
config.DATASET.VALID_JOINT_VOCAB_PATH = os.path.join(
    VOCAB_ROOT, ROW_MODE_DIR, HARD_EXERCISE_MODE_DIR,
    'VALID', 'joint_vocab.pkl',
)
config.DATASET.VALID_WORKOUT_VOCAB_PATH = os.path.join(
    VOCAB_ROOT, ROW_MODE_DIR, HARD_EXERCISE_MODE_DIR,
    'VALID', 'workout_vocab.pkl',
)

# Model
config.FEATURE_MODES_IN_DIM = {
        'BASELINE': 3,
        'VECTOR': 5,
        'GRAPH_1HOP': 5,
        'GRAPH_2HOP': 7,
        'VECTOR_GRAPH_1HOP': 7,
        'VECTOR_GRAPH_2HOP': 9,
        'SKELETON_EDGE_1HOP': 5,
        'VECTOR_SKELETON_EDGE_1HOP': 7
}

config.MODEL = edict()
config.MODEL.IN_CHANNELS = config.FEATURE_MODES_IN_DIM[config.FEATURE_MODES] -1 if config.EMB_MODE != 'RwID' else config.FEATURE_MODES_IN_DIM[config.FEATURE_MODES]

ID_AWARE_EMB_MODES = ('RwID', 'RwIDNorm')
config.MODEL.IN_CHANNELS = (
  config.FEATURE_MODES_IN_DIM[config.FEATURE_MODES]
  if config.EMB_MODE in ID_AWARE_EMB_MODES
  else config.FEATURE_MODES_IN_DIM[config.FEATURE_MODES] - 1
)

config.MODEL.OUT_CHANNELS = 768 if not config.R_PRETRAINED else int(config.R_PRETRAINED_PATH.split()[3].split(':')[-1])
config.MODEL.NUM_LAYER = 4 if not config.R_PRETRAINED else int(config.R_PRETRAINED_PATH.split()[1].split(':')[-1])


# Train
config.TRAIN = edict()
config.TRAIN.EPOCH = 30
config.TRAIN.WARMUP_EPOCH = 3
config.TRAIN.BATCH_SIZE = 256  # during test, bs = 1
config.TRAIN.LR = 5e-5
config.TRAIN.ACT = 'GELU'  # ['ReLU', 'Mish' ... ]
if not config.R_PRETRAINED:
    config.TRAIN.S = 10
else:
    pretrained_scale = parse_arcface_parameter(
        config.R_PRETRAINED_PATH,
        'S',
    )
    config.TRAIN.S = (
        int(pretrained_scale)
        if pretrained_scale.is_integer()
        else pretrained_scale
    )
config.TRAIN.M = 0.1 if not config.R_PRETRAINED else parse_arcface_parameter(config.R_PRETRAINED_PATH, 'M')
config.TRAIN.SHUFFLE = True
config.TRAIN.NUM_SAMPLE = 2 # NUM_SAMPLE * 20(num_joint) * config.TRAIN.BATCH_SIZE

# VALID
config.VALID = edict()
config.VALID.BATCH_SIZE = 256
config.VALID.NUM_SAMPLE = 100

# Print
config.PRINT_FREQ = 50

# Visualization
config.VIS = edict()
config.VIS.PLOT_DIM = 3 # or 3
config.VIS.SAMPLES_PER_CLASS = 300 # 현재 Validation 데이터셋 기준으로 MAX 값은 1440
config.VIS.TSNE_PERPLEXITY = 30
config.VIS.TSNE_N_ITER = 1000
config.VIS.TSNE_RANDOM_SEED = 42
config.VIS.PLOT_SAVE_ROOT = "./embedding_result_img"
config.VIS.PLOT_METRIC_METHOD = 'cosine'    # 'cosine' 'euclidean'
config.VIS.PLOT_VISUALIZATION = True
#

# Experiment
config.EXP = edict()
config.EXP.FEATURE_MODES = [
  'BASELINE',
  'VECTOR',
  'GRAPH_1HOP',
  'GRAPH_2HOP',
  'VECTOR_GRAPH_1HOP',
  'VECTOR_GRAPH_2HOP',
  'SKELETON_EDGE_1HOP',
  'VECTOR_SKELETON_EDGE_1HOP',
]

config.EXP.FEATURE_MODES_IN_DIM = config.FEATURE_MODES_IN_DIM
config.EXP.EMB_MODE = ['B+R', 'R', 'RwID', 'RwIDNorm']

# 두 list는 같은 index끼리 하나의 Dataset 구성이다.
config.EXP.IS_CONTAIN_HARD_EXERCISE = [
  True,   # Original data
  False,  # Hard Exercise 제외
]

config.EXP.IS_INTEGRATE_ROW = [
  False,  # Original data
  True,   # Row 통합
]

config.EXP.IS_JOINT_ID_NORM = [
  False,
  True,
]

config.EXP.NUM_LAYERS = [2, 4, 6]
config.EXP.S_RANGE = [10, 20, 30, 40, 50]
config.EXP.M_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5]
config.EXP.ACT = 'GELU'
config.EXP.EMB_DIM = [768]

config.VIS.SAMPLES_PER_CLASS = 300

config.FILE_NAME = (
    f'[{config.EMB_MODE}] '
    f'LAYERS_NUM:{config.MODEL.NUM_LAYER} '
    f'IN_DIM:{config.MODEL.IN_CHANNELS} '
    f'DIM:{config.MODEL.OUT_CHANNELS} '
    f'S:{config.TRAIN.S} '
    f'M:{config.TRAIN.M} {DATASET_NAME}'
)


# core.py 에서의 config를 확인하기 위한 용도
# Experiments 코드에서는 의미 X
print(f'MODE: {config.MODE}')
print(f'FEATURE MODE: {config.FEATURE_MODES}')
print(f'EMB_MODE: {config.EMB_MODE}')
if config.R_PRETRAINED:
    print(f'Relative Pretrained: {config.R_PRETRAINED_PATH}')
if config.B_PRETRAINED:
    print(f'Basis Pretrained: {config.B_PRETRAINED_PATH}')
print(f'Dimension of input: {config.MODEL.IN_CHANNELS}')
print(f'Dimension of output: {config.MODEL.OUT_CHANNELS}')
print(f'The number of MLP layers : {config.MODEL.NUM_LAYER}')
print(f'The value of parameter s:{config.TRAIN.S}')
print(f'The value of parameter m:{config.TRAIN.M}')
