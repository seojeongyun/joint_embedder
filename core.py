
from email.policy import default

import torch
import pickle
import yaml
import time
import random
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.nn.modules import loss

from config.config import config
from easydict import EasyDict as edict

from utils.function import plot_tsne_with_centroids
from utils.AverageMeter import AverageMeter

from model.CosineFace import CosFace
from model.ArcFace import ArcFace
from loader.Coord_Dataset import Coord_Dataset

def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    import os
    from setproctitle import *

    setproctitle('ExponentialLR-gamma:0.93/256,256')

    fix_seed(config.SEED)

    selected_data_path = (
        config.DATASET.TRAIN_DATA_PATH
        if config.MODE == 'TRAIN'
        else config.DATASET.VALID_DATA_PATH
    )
    print(f'[Dataset] {selected_data_path}')

    if not os.path.isfile(selected_data_path):
        raise FileNotFoundError(
            'Selected ArcFace dataset does not exist: '
            f'{selected_data_path}'
        )

    gen_config('/home/jysuh/PycharmProjects/coord_embedding/coord_embed.yaml')

    NUM_JOINTS = config.DATASET.NUM_JOINTS
    NUM_TOKEN = config.DATASET.NUM_TOKEN

    device = torch.device(f"cuda:{config.GPUS}" if torch.cuda.is_available() else "cpu")

    fc_metric = ArcFace(num_layer=config.MODEL.NUM_LAYER, in_features=config.MODEL.IN_CHANNELS, \
                        out_features=config.MODEL.OUT_CHANNELS, num_class=NUM_JOINTS+NUM_TOKEN, \
                        embedding_mode=config.EMB_MODE, activation=config.TRAIN.ACT, device=device).to(device)

    train_dataset = Coord_Dataset(config=config, mode=config.MODE)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(
        fc_metric.parameters(),
        lr=config.TRAIN.LR
    )

    cosine_epochs = config.TRAIN.EPOCH - config.TRAIN.WARMUP_EPOCH

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.TRAIN.WARMUP_EPOCH
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            warmup_scheduler,
            cosine_scheduler
        ],
        milestones=[config.TRAIN.WARMUP_EPOCH]
    )
    #
    base_token_ids = torch.arange(NUM_TOKEN + NUM_JOINTS, device=device, dtype=torch.long).unsqueeze(0)

    #
    if config.MODE == 'TRAIN':
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        #
        best_loss = float("inf")
        best_epoch = None
        #
        for epoch in range(config.TRAIN.EPOCH):
            batch_time.reset()
            data_time.reset()
            losses.reset()
            #
            end = time.time()
            #
            epoch_start_time = time.time()
            for i ,(J_coord) in enumerate(train_loader):
                BS = J_coord.shape[0]
                #
                loss = 0
                #
                J_coord = J_coord.to(device, non_blocking=True)
                token_ids = base_token_ids.expand(BS, -1)
                optimizer.zero_grad()

                logits, _ = fc_metric(input=J_coord, token_ids=token_ids, use_arcface=True, m=config.TRAIN.M, s=config.TRAIN.S)
                #
                labels = token_ids.reshape(-1).long()
                loss = criterion(logits, labels)
                #
                loss.backward()
                losses.update(loss.detach(), J_coord.size(0))
                #
                batch_time.update(time.time() - end)
                end = time.time()
                #
                optimizer.step()
            #
            #
            epoch_loss = losses.avg.item()

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch

                save_dir = config.SAVE_ROOT + '/' + f'{config.FILE_NAME}/weights'
                os.makedirs(save_dir, exist_ok=True)

                # 1. ArcFace Classifier SAVE
                arcface_classifier_save_path = os.path.join(save_dir, "best_arcface_classifier.pt")
                torch.save({"weight": fc_metric.weight.detach().cpu()}, arcface_classifier_save_path)

                # 2. Relative MLP SAVE
                metric_learning_model_save_path = os.path.join(save_dir, "best_metric_learning_model.pth.tar")
                torch.save(fc_metric.layers.state_dict(), metric_learning_model_save_path)

                # 3. nn.Embedding SAVE
                nn_embedding_save_path = os.path.join(save_dir, "best_nn_embedding.pt")
                torch.save(fc_metric.embedding.state_dict(), nn_embedding_save_path)

            #
            scheduler.step()

            loss_value = losses.val.item()
            loss_average = losses.avg.item()
            msg = (
                f"Epoch: [{epoch}]  "
                f"Time {batch_time.val:.3f}s   "
                f"Speed {J_coord.size(0) / batch_time.val:.1f} samples/s\t"
                f"Loss {loss_value:.7f} "
                f"({loss_average:.7f})"
            )
            print(msg)

            epoch_end_time = time.time()
            # print(f"{epoch_end_time - epoch_start_time:.2f} sec")

        print(
            f"Training finished - "
            f"Best Loss: {best_loss:.7f} "
            f"(Epoch {best_epoch + 1})"
        )

        save_dir = config.SAVE_ROOT + '/' + f'{config.FILE_NAME}/weights'
        os.makedirs(save_dir, exist_ok=True)

        # 1. ArcFace Classifier SAVE
        arcface_classifier_save_path = os.path.join(save_dir, "final_arcface_classifier.pt")
        torch.save({"weight": fc_metric.weight.detach().cpu()}, arcface_classifier_save_path)

        # 2. Relative MLP SAVE
        metric_learning_model_save_path = os.path.join(save_dir, "final_metric_learning_model.pth.tar")
        torch.save(fc_metric.layers.state_dict(), metric_learning_model_save_path)

        # 3. nn.Embedding SAVE
        nn_embedding_save_path = os.path.join(save_dir, "final_nn_embedding.pt")
        torch.save(fc_metric.embedding.state_dict(), nn_embedding_save_path)

    #
    #
    #

    elif config.MODE == 'VALID':
        # [1] Weight Load
        if config.R_PRETRAINED:
            mlp_state = torch.load(config.R_PRETRAINED_PATH, map_location=device)
            fc_metric.layers.load_state_dict(mlp_state, strict=True)
            print('[Relative Load] ' + config.R_PRETRAINED_PATH)

        if config.B_PRETRAINED:
            embedding_state = torch.load(config.B_PRETRAINED_PATH, map_location=device)
            fc_metric.embedding.load_state_dict(embedding_state, strict=True)
            print('[Basis Load] ' + config.B_PRETRAINED_PATH)


        # [2] Vocab Load
        with open(config.DATASET.VALID_JOINT_VOCAB_PATH, 'rb') as f:
            joint_vocab = pickle.load(f)

        # [3] Data Loader
        valid_dataset = Coord_Dataset(config=config, mode=config.MODE)

        sample_generator = torch.Generator()
        sample_generator.manual_seed(config.VIS.TSNE_RANDOM_SEED)

        num_pose_samples = min(
            config.VIS.SAMPLES_PER_CLASS,
            len(valid_dataset)
        )

        sample_indices = torch.randperm(
            len(valid_dataset),
            generator=sample_generator
        )[:num_pose_samples].tolist()

        valid_subset = torch.utils.data.Subset(
            valid_dataset,
            sample_indices
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_subset,
            batch_size=config.VALID.BATCH_SIZE,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True
        )

        fc_metric.eval()
        #
        all_feats = []
        all_labels = []
        #
        batch_time = AverageMeter()
        data_time = AverageMeter()

        with torch.inference_mode():
            valid_start = time.time()
            for i ,(J_coord) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                # BS 계산
                BS = J_coord.shape[0]

                # 입력 데이터
                J_coord = J_coord.to(device, non_blocking=True)
                token_ids = base_token_ids.expand(BS, -1)

                # Forward
                _, embedding_vec = fc_metric(input=J_coord, token_ids=token_ids, use_arcface=False, m=config.TRAIN.M, s=config.TRAIN.S)

                #
                all_feats.append(embedding_vec.reshape(-1, embedding_vec.size(-1)).cpu())
                all_labels.append(token_ids.reshape(-1).cpu())

    #
        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        score = plot_tsne_with_centroids(config=config, feats=all_feats, labels=all_labels, vocab=joint_vocab, visualization=config.VIS.PLOT_VISUALIZATION)
        valid_end = time.time()
        print(f'{valid_end - valid_start} sec')
        print(score)

# 시각화에서 Vocab이 필요할까봐 일단 백업
'''
    def get_vocab(self):
        sepcial_token_vocab = {'PAD': 0, 'SEP' : 1}
        if self.mode == 'TRAIN':
            with open(self.config.DATASET.TRAIN_JOINT_VOCAB_PATH, 'rb') as f:
                joint_vocab = pickle.load(f)

            with open(self.config.DATASET.TRAIN_WORKOUT_VOCAB_PATH, 'rb') as f:
                workout_vocab = pickle.load(f)
        
        elif self.mode == 'VALID':
            with open(self.config.DATASET.VALID_JOINT_VOCAB_PATH, 'rb') as f:
                joint_vocab = pickle.load(f)

            with open(self.config.DATASET.VALID_WORKOUT_VOCAB_PATH, 'rb') as f:
                workout_vocab = pickle.load(f)
                
        return sepcial_token_vocab, joint_vocab, workout_vocab
'''
