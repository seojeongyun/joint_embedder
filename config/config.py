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
config.DATASET.TRAIN_DATA_PATH = glob('/home/jysuh/PycharmProjects/coord_embedding/dataset/*')
config.DATASET.VALID_DATA_PATH = '/home/jysuh/PycharmProjects/coord_embedding/dataset/coord_valid.json'
config.DATASET.NUM_JOINTS = 20


# Train
config.TRAIN = edict()
config.TRAIN.ONLY_METRIC_LEARN = True
config.TRAIN.SHUFFLE = True
#
config.TRAIN.BATCH_SIZE = 1  # during test, bs = 1
config.TRAIN.LR = 5e-4
config.TRAIN.ACT = 'Mish'  # ['ReLU', 'Mish' ... ]
config.TRAIN.ACT_LIST = ['ReLU', 'GELU']
config.TRAIN.EPOCH = 1000
config.TRAIN.WARMUP = True
config.TRAIN.WARMUP_EPOCH = 250

#
config.TRAIN.S_RANGE = list(np.linspace(1,100, 100))
config.TRAIN.M_RANGE = [round(x, 2) for x in np.arange(0.05, 0.8 + 0.001, 0.05)]
config.TRAIN.EMB_DIM = [512, ]
config.TRAIN.LOSSES = ['CosFace', 'ArcFace']
config.TRAIN.LOSS = 'CosFace'

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
                       + ' only_metric:' + f'{config.TRAIN.ONLY_METRIC_LEARN}' \
                       + ' total epoch:' + f'{config.TRAIN.EPOCH}' \
                       + ' warmup:' + f'{config.TRAIN.WARMUP}' \
                       + ' max_iter:' + f'{config.VIS.TSNE_N_ITER}' \
                       + ' perplexity:' + f'{config.VIS.TSNE_PERPLEXITY}'

# PreTrained
config.PRETRAINED = True
config.PRETRAINED_PATH = f'/home/jysuh/PycharmProjects/coord_embedding/checkpoint/{config.FILE_NAME}.pth.tar'
#
# config.PRETRAINED_EMB = False
# config.PRETRAINED_EMB_PATH = '/home/jysuh/PycharmProjects/coord_embedding/checkpoint/kobart_embedding_weights.pt'

config.MODE = 'TEST' # ['TRAIN',  'TEST']

'''
    embedding_vec = linear_out + emb_output_J_tokens
        : {'num_samples': 41900, 'num_classes': 20, 'feat_dim': 512, 'intra_class_distance_2d': 33.987281799316406, 'silhouette_score_2d': 0.23017668724060059, 'inter_class_similarity_orig': 0.8255321301912006, 'silhouette_score_orig': 0.8783144354820251, 'preferred_eval_space': 'orig', 'passed': False}
    
    embedding_vec = linear_out # in_feat = 4, s = 30
        : {'num_samples': 41900, 'num_classes': 20, 'feat_dim': 512, 'intra_class_distance_2d': 92.08616638183594, 'silhouette_score_2d': -0.11935179680585861, 'inter_class_similarity_orig': 0.9941247237356086, 'silhouette_score_orig': -0.19085559248924255, 'preferred_eval_space': 'orig', 'passed': False}
    
    embedding_vec = linear_out # in_feat = 3, s = 30 
        : {'num_samples': 41900, 'num_classes': 20, 'feat_dim': 512, 'intra_class_distance_2d': 92.63642883300781, 'silhouette_score_2d': -0.11507540941238403, 'inter_class_similarity_orig': 0.9929228130139802, 'silhouette_score_orig': -0.20435325801372528, 'preferred_eval_space': 'orig', 'passed': False}

    embedding_vec = linear_out # s = 64
        : {'num_samples': 41900, 'num_classes': 20, 'feat_dim': 512, 'intra_class_distance_2d': 91.49295043945312, 'silhouette_score_2d': -0.12104460597038269, 'inter_class_similarity_orig': 0.9860030324835526, 'silhouette_score_orig': -0.21402911841869354, 'preferred_eval_space': 'orig', 'passed': False}

    
    [ArcFace]
    embedding_vec = linear_out
        epoch 100
        : {'num_samples': 41900, 'num_classes': 20, 'feat_dim': 512, 'intra_class_distance_2d': 32.95997619628906, 'silhouette_score_2d': 0.25474146008491516, 'inter_class_similarity_orig': -0.015489899484734787, 'silhouette_score_orig': 0.2651854455471039, 'preferred_eval_space': 'orig', 'passed': True}
        
        epoch 1000
        : {'num_samples': 41900, 'num_classes': 20, 'feat_dim': 512, 'intra_class_distance_2d': 26.220355987548828, 'silhouette_score_2d': 0.348380446434021, 'inter_class_similarity_orig': -0.039527230513723276, 'silhouette_score_orig': 0.7568519115447998, 'preferred_eval_space': 'orig', 'passed': True}
        
        epoch 1000 + warmup        
        : {'num_samples': 41900, 'num_classes': 20, 'feat_dim': 512, 'intra_class_distance_2d': 24.573291778564453, 'silhouette_score_2d': 0.3741514980792999, 'inter_class_similarity_orig': -0.030053470009251643, 'silhouette_score_orig': 0.8257231116294861, 'preferred_eval_space': 'orig', 'passed': True}
        : {'num_samples': 41900, 'num_classes': 20, 'feat_dim': 512, 'intra_class_distance_2d': 38.74065399169922, 'silhouette_score_2d': 0.4435234069824219, 'inter_class_similarity_orig': -0.030053470009251643, 'silhouette_score_orig': 0.8257231116294861, 'preferred_eval_space': 'orig', 'passed': True}

    
    embedding_vec = linear_out + dropout
        : {'num_samples': 41900, 'num_classes': 20, 'feat_dim': 512, 'intra_class_distance_2d': 28.120925903320312, 'silhouette_score_2d': 0.3110218644142151, 'inter_class_similarity_orig': -0.02803228779842979, 'silhouette_score_orig': 0.6876890659332275, 'preferred_eval_space': 'orig', 'passed': True}

    embedding_vec = linear_out + emb_output_J_tokens
        : {'num_samples': 41900, 'num_classes': 20, 'feat_dim': 512, 'intra_class_distance_2d': 26.4351806640625, 'silhouette_score_2d': 0.35250571370124817, 'inter_class_similarity_orig': -0.04438762664794922, 'silhouette_score_orig': 0.8550278544425964, 'preferred_eval_space': 'orig', 'passed': True}

    embedding_vec = norm(linear_out) + norm(emb_output_J_tokens)
        : {'num_samples': 41900, 'num_classes': 20, 'feat_dim': 512, 'intra_class_distance_2d': 27.03829002380371, 'silhouette_score_2d': 0.32438716292381287, 'inter_class_similarity_orig': -0.030857480199713457, 'silhouette_score_orig': 0.7611210942268372, 'preferred_eval_space': 'orig', 'passed': True}
        

    
    [CosFace]
    embedding_vec = linear_out
        : {'num_samples': 41900, 'num_classes': 20, 'feat_dim': 512, 'intra_class_distance_2d': 33.52564239501953, 'silhouette_score_2d': 0.21586337685585022, 'inter_class_similarity_orig': -0.026102954462954873, 'silhouette_score_orig': 0.20289216935634613, 'preferred_eval_space': 'orig', 'passed': False}

'''