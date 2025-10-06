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

from model.arc_margin_loss_for_various_testcases import AddMarginProduct
from model.LiArcFace_for_various_testcases import LiArcFace
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
    fix_seed(config.SEED)

    gen_config('/home/jysuh/PycharmProjects/coord_embedding/coord_embed.yaml')

    NUM_JOINTS = config.DATASET.NUM_JOINTS

    device = torch.device(f"cuda:{config.GPUS}" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss().to(device)

    train_dataset = Coord_Dataset(cfg=config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    #
    for loss_type in config.TRAIN.LOSSES:
        for embedding_dim in config.TRAIN.EMB_DIM:
            for s in config.TRAIN.S_RANGE:
                for m in config.TRAIN.M_RANGE:
                    #
                    train_config = 'loss_' + str(loss_type) + '_dim_' + str(embedding_dim) + '_s_' + str(s) + '_m_' + str(m)
                    # print(train_config)
                    if loss_type == 'CosFace':
                        fc_metric = AddMarginProduct(in_features=4, out_features=embedding_dim, num_class=NUM_JOINTS, s=s, m=m, device=device).to(device)

                    elif loss_type == 'ArcFace':
                        fc_metric = LiArcFace(in_features=4, out_features=embedding_dim, num_class=NUM_JOINTS, s=s, m=m,
                                                     device=device).to(device)

                    optimizer = torch.optim.Adam(fc_metric.parameters(), lr=config.TRAIN.LR)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAIN.EPOCH, eta_min=1e-7)

                    fc_metric.train()

                    for epoch in range(config.TRAIN.EPOCH):
                        batch_time = AverageMeter()
                        data_time = AverageMeter()
                        losses = AverageMeter()

                        end = time.time()
                        #
                        for i ,(J_coord, J_tokens) in enumerate(train_loader):
                            J_coord = J_coord.to(device)
                            J_tokens = J_tokens.to(device)
                            optimizer.zero_grad()

                            for idx in range(NUM_JOINTS):
                                #
                                logits, _ = fc_metric(input=J_coord[:, idx, :], J_tokens=J_tokens[:, idx], mode='training')
                                label = J_tokens[:, idx]
                                loss = criterion(logits, label)
                            #
                            loss.backward()
                            losses.update(loss.item(), J_coord.size(0))
                            #
                            batch_time.update(time.time() - end)
                            end = time.time()
                            # if i % config.PRINT_FREQ == 0:
                                # msg = 'Epoch: [{0}][{1}/{2}]\t' \
                                #       'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                                #       'Speed {speed:.1f} samples/s\t' \
                                #       'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                                #       'Loss {loss.val:.7f} ({loss.avg:.7f})'.format(
                                #     epoch, i, len(train_loader), batch_time=batch_time,
                                #     speed=J_coord.size(0) / batch_time.val,
                                #     data_time=data_time, loss=losses)
                                # logger.info(msg)
                                # print(msg)
                            #
                            optimizer.step()
                        scheduler.step()
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

                    score = plot_tsne_with_centroids(feats=all_feats, labels=all_labels, title="t-SNE (512to2) with Centroids",
                                                     perplexity=30, vocab=train_dataset.vocab, train_config=train_config)

                    result_dict = {}
                    if score['intra_class_distance'] < 25 and score['inter_class_similarity'] < 0.5 and score['silhouette_score'] > 0.4:
                        result_dict['state_dict'] = fc_metric.state_dict()
                        result_dict['score'] = score
                        result_dict['model'] = fc_metric
                        torch.save(result_dict, f'/home/jysuh/PycharmProjects/coord_embedding/checkpoint/epoch_{train_config}_{str(epoch)}.pth.tar')
                    print(f"{train_config} | Score: {score} | Final Loss: {losses.val:.4f}")
