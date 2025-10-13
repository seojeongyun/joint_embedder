import torch

import yaml
import time
import logging
import random
import numpy as np

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

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    fix_seed(config.SEED)

    gen_config('/home/jysuh/PycharmProjects/coord_embedding/coord_embed.yaml')

    NUM_JOINTS = config.DATASET.NUM_JOINTS

    device = torch.device(f"cuda:{config.GPUS}" if torch.cuda.is_available() else "cpu")

    if config.TRAIN.LOSS == 'CosFace':
        fc_metric = CosFace(in_features=4, out_features=512, num_class=NUM_JOINTS, only_metric=config.TRAIN.ONLY_METRIC_LEARN, activation=config.TRAIN.ACT, s=10.0, m=0.20, device=device).to(device)

    elif config.TRAIN.LOSS == 'ArcFace':
        fc_metric = ArcFace(in_features=4, out_features=512, num_class=NUM_JOINTS,  only_metric=config.TRAIN.ONLY_METRIC_LEARN, activation=config.TRAIN.ACT, s=10.0, m=0.20,device=device).to(device)


    if config.PRETRAINED:
        state_dict = torch.load(config.PRETRAINED_PATH, map_location=device)
        fc_metric.load_state_dict(state_dict)
        print('load weight...' + config.PRETRAINED_PATH)
        print(fc_metric)

    if config.PRETRAINED_EMB:
        emb_weight = torch.load(config.PRETRAINED_EMB_PATH)
        statedict = fc_metric.state_dict()
        statedict['embedding.weight'] = emb_weight[:20, :512]
        fc_metric.load_state_dict(statedict, strict=True)
        print('load embedding weight...' + config.PRETRAINED_EMB_PATH)
        # embedding = embedding.from_pretrained(emb_weight)


    train_dataset = Coord_Dataset(train_path=config.DATASET.VALID_DATA_PATH)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(fc_metric.parameters(), lr=config.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAIN.EPOCH, eta_min=1e-7)

    if config.MODE == 'TRAIN':
        for epoch in range(config.TRAIN.EPOCH):
            if config.TRAIN.WARMUP:
                m = (epoch / config.TRAIN.WARMUP_EPOCH) * 0.1 + 0.1
                s = (epoch / config.TRAIN.WARMUP_EPOCH) * 10 + 10
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()

            end = time.time()
            #
            for i ,(J_coord, J_tokens) in enumerate(train_loader):
                loss = 0
                J_coord = J_coord.to(device)
                J_tokens = J_tokens.to(device)
                optimizer.zero_grad()

                for idx in range(NUM_JOINTS):
                    #
                    if config.TRAIN.WARMUP:
                        logits, _ = fc_metric(input=J_coord[:, idx, :], J_tokens=J_tokens[:, idx], mode='training', m=m, s=s)
                    else:
                        logits, _ = fc_metric(input=J_coord[:, idx, :], J_tokens=J_tokens[:, idx], mode='training')
                    label = J_tokens[:, idx]
                    loss += criterion(logits, label)
                    # print(criterion(logits, label))
                #
                loss /= NUM_JOINTS
                loss.backward()
                losses.update(loss.item(), J_coord.size(0))
                #
                batch_time.update(time.time() - end)
                end = time.time()
                #
                if i % config.PRINT_FREQ == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Loss {loss.val:.7f} ({loss.avg:.7f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        speed=J_coord.size(0) / batch_time.val,
                        data_time=data_time, loss=losses)
                    # logger.info(msg)
                    print(msg)
                optimizer.step()
            scheduler.step()
        torch.save(fc_metric.state_dict(), f'/home/jysuh/PycharmProjects/coord_embedding/checkpoint/{config.FILE_NAME}.pth.tar')

    elif config.MODE == 'TEST':
        fc_metric.eval()
        #
        all_feats = []
        all_labels = []
        #
        batch_time = AverageMeter()
        data_time = AverageMeter()

        with torch.no_grad():
            for i ,(J_coord, J_tokens) in enumerate(train_loader):
                J_coord = J_coord.to(device)
                J_tokens = J_tokens.to(device)

                for idx in range(NUM_JOINTS):
                    _, embedding_vec = fc_metric(input=J_coord[:, idx, :], J_tokens=J_tokens[:, idx], mode='validation')

                    all_feats.append(embedding_vec)          # List of [BS, 512]
                    all_labels.append(J_tokens[:, idx])      # List of [BS]

        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        score = plot_tsne_with_centroids(config=config, feats=all_feats, labels=all_labels, vocab=train_dataset.vocab)
        print(score)
