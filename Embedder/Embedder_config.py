import torch
from easydict import EasyDict as edict

config = edict()


# SEED
config.SEED = 478

# DATA
config.JOINTS_NAME = [
    'Left Shoulder', 'Right Shoulder',
    'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist',
    'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee',
    'Left Ankle', 'Right Ankle',
    'Neck', 'Left Palm',
    'Right Palm', 'Back',
    'Waist', 'Left Foot',
    'Right Foot', 'Head'
    ]
config.IMG_SIZE = [1920, 1080]
config.DATA_PATH = '/home/jysuh/PycharmProjects/coord_embedding/dataset/embedder_train_valid.json'
config.NUM_JOINTS = 22  # 0, 1 = PAD, SEP, others joints
config.MAX_FRAMES = 21

# GPU / WORKERS
config.GPUS = '0'
config.WORKERS = 0
config.DEVICE = torch.device(f"cuda:{config.GPUS}" if torch.cuda.is_available() else "cpu")

# PRETRAINED
config.PRETRAINED_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/find_sm/[s,m Basis+Relative] LAYERS_NUM:6 DIM:768 ACT:GELU s:10 m:0.45/metric_learning_model.pth.tar'
config.PRETRAINED_EMB_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/find_sm/[s,m Basis+Relative] LAYERS_NUM:6 DIM:768 ACT:GELU s:10 m:0.45/nn_embedding_model.pt'

# Model
config.IN_FEAT = 4
config.OUT_FEAT = int(config.PRETRAINED_PATH.split('/')[7].split()[3].split(':')[-1])
#
config.NUM_LAYER = int(config.PRETRAINED_PATH.split('/')[7].split()[2].split(':')[-1])
config.ACTIV = config.PRETRAINED_PATH.split('/')[7].split()[4].split(':')[-1]
#
config.USE_EMBEDDING = True

# OPERATION
config.BATCH_SIZE = 32
