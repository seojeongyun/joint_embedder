import torch

import yaml
import time
import random
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.nn.modules import loss
from torch.utils.tensorboard import SummaryWriter

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
    NUM_TOKEN = config.DATASET.NUM_TOKEN

    device = torch.device(f"cuda:{config.GPUS}" if torch.cuda.is_available() else "cpu")

    train_dataset = Coord_Dataset(config=config, data_path=config.DATASET.TRAIN_DATA_PATH)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    valid_dataset = Coord_Dataset(config=config, data_path=config.DATASET.VALID_DATA_PATH)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        collate_fn=valid_dataset.collate_fn
    )

    criterion = nn.CrossEntropyLoss().to(device)
    train_label2name = {v: k for k, v in train_dataset.vocab.items()}
    valid_label2name = {v: k for k, v in valid_dataset.vocab.items()}
    #

    # Training loop
    #
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # To collect data for model test.
    TRAIN_TOTAL_ITERS = len(train_loader)
    train_random_indices = set(random.sample(range(TRAIN_TOTAL_ITERS), config.TRAIN.NUM_SAMPLE))
    #
    VALID_TOTAL_ITERS = len(valid_loader)
    valid_random_indices = set(random.sample(range(VALID_TOTAL_ITERS), config.VALID.NUM_SAMPLE))
    #
    token_idx = [i for i in range(NUM_TOKEN)]
    token_idx = torch.LongTensor(token_idx)
    #

    for num_layers in config.MODEL.NUM_LAYERS:
        for USE_EMB in config.TRAIN.USE_EMB_LIST:  # use emb
            for DIM in config.TRAIN.EMB_DIM:
                for ACT in config.TRAIN.ACT_LIST:
                    if USE_EMB:
                        file_name = '[Basis+Relative] ' + f'LAYERS_NUM:{num_layers} ' + f'DIM:{DIM} ' + f'ACT:{ACT} ' + 's:10 m:0.1'
                    else:
                        file_name = '[Relative] ' + f'LAYERS_NUM:{num_layers} '+ f'DIM:{DIM} ' + f'ACT:{ACT} ' + 's:10 m:0.1'
                    #
                    writer = SummaryWriter(log_dir=f'./tb_logger/find_layernum_dim_atc_etc/{file_name}')
                    #
                    fc_metric = ArcFace(num_layer=num_layers, in_features=4, out_features=DIM, num_class=NUM_JOINTS,
                                        use_embedding=USE_EMB, activation=ACT, s=10.0,
                                        m=0.10, device=device).to(device)
                    fc_metric.train()
                    #
                    optimizer = torch.optim.AdamW(fc_metric.parameters(), lr=config.TRAIN.LR)
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93)
                    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAIN.EPOCH,
                    #                                                        eta_min=1e-7)
                    for epoch in range(config.TRAIN.EPOCH):
                        all_feats = []
                        all_labels = []
                        #
                        batch_time.reset()
                        data_time.reset()
                        losses.reset()

                        end = time.time()
                        #
                        for i, (J_coord, J_tokens, WRKOUT, FRAME, VIEW, VIDEO) in enumerate(train_loader):
                            if len(VIDEO) == 0:
                                continue
                            MINI_BATCH = WRKOUT.shape[0]
                            #
                            loss = 0
                            #
                            dummy = torch.zeros([MINI_BATCH, NUM_TOKEN, 4]).to(device)
                            J_coord = torch.cat([dummy, J_coord.to(device)], dim=1)
                            #
                            J_tokens = torch.cat((token_idx, J_tokens[0]))
                            J_tokens = J_tokens.unsqueeze(0).repeat(MINI_BATCH, 1).to(device)
                            #
                            optimizer.zero_grad()

                            for idx in range(NUM_TOKEN + NUM_JOINTS):
                                logits, embedding_vec = fc_metric(input=J_coord[:, idx, :], J_tokens=J_tokens[:, idx], mode='training')
                                #
                                if i in train_random_indices:
                                    all_feats.append(embedding_vec.detach().cpu())  # List of [BS, 512]
                                    all_labels.append(J_tokens[:, idx].detach().cpu())  # List of [BS]

                                label = J_tokens[:, idx]
                                loss += criterion(logits, label)
                                # print(criterion(logits, label))
                            #
                            loss /= (NUM_TOKEN + NUM_JOINTS)
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
                            #
                            optimizer.step()

                        # Write the losses to TensorBoard
                        all_feats = torch.cat(all_feats, dim=0)
                        all_labels = torch.cat(all_labels, dim=0)
                        # if epoch % K == 0:
                        score = plot_tsne_with_centroids(config=config, feats=all_feats, labels=all_labels,
                                                         vocab=train_dataset.vocab)
                        writer.add_scalar('TRAIN/Loss', losses.avg, epoch)
                        writer.add_scalar('TRAIN/Dunn Index', score['dunn_index_orig'], epoch)
                        for joint_idx in score['silhouette_score_per_class'].keys():
                            joint_name = train_label2name[joint_idx]
                            writer.add_scalar(f'TRAIN/Silhouette/{joint_name}', score['silhouette_score_per_class'][joint_idx], epoch)
                        writer.add_scalar('TRAIN/Silhouette/avg', score['silhouette_score_orig'], epoch)

                        # scheduler update
                        scheduler.step()
                        lr = scheduler.get_last_lr()[0]
                        writer.add_scalar('TRAIN/lr', lr, epoch)
                    #
                    #
                    fc_metric.eval()
                    #
                    all_feats = []
                    all_labels = []
                    #
                    with torch.no_grad():
                        for i, (J_coord, J_tokens, WRKOUT, FRAME, VIEW, VIDEO) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                            if len(VIDEO) == 0:
                                continue
                            MINI_BATCH = WRKOUT.shape[0]
                            #
                            dummy = torch.zeros([MINI_BATCH, NUM_TOKEN, 4]).to(device)
                            J_coord = torch.cat([dummy, J_coord.to(device)], dim=1)
                            #
                            J_tokens = torch.cat((token_idx, J_tokens[0]))
                            J_tokens = J_tokens.unsqueeze(0).repeat(MINI_BATCH, 1).to(device)

                            for idx in range(NUM_TOKEN + NUM_JOINTS):
                                _, embedding_vec = fc_metric(input=J_coord[:, idx, :], J_tokens=J_tokens[:, idx], mode='validation')
                                #
                                if i in valid_random_indices:
                                    all_feats.append(embedding_vec.detach().cpu())  # List of [BS, 512]
                                    all_labels.append(J_tokens[:, idx].detach().cpu())  # List of [BS]
                        #
                    all_feats = torch.cat(all_feats, dim=0)
                    all_labels = torch.cat(all_labels, dim=0)

                    # Write the losses to TensorBoard
                    score = plot_tsne_with_centroids(config=config, feats=all_feats, labels=all_labels,
                                                     vocab=valid_dataset.vocab, file_name=file_name, visualization=config.VIS.PLOT_VISUALIZATION)
                    writer.add_scalar('VAL/Dunn Index', score['dunn_index_orig'])
                    for joint_idx in score['silhouette_score_per_class'].keys():
                        joint_name = valid_label2name[joint_idx]
                        writer.add_scalar(f'VAL/Silhouette/{joint_name}', score['silhouette_score_per_class'][joint_idx])
                    writer.add_scalar('VAL/Silhouette/avg', score['silhouette_score_orig'])
                    writer.close()

                save_dir = f"/home/jysuh/PycharmProjects/coord_embedding/checkpoint/{file_name}"
                os.makedirs(save_dir, exist_ok=True)

                metric_learning_model_save_path = os.path.join(save_dir, "metric_learning.pth.tar")
                torch.save(fc_metric.state_dict(), metric_learning_model_save_path)
                #
                nn_embedding_save_path = os.path.join(save_dir, "nn.embedding.pth.tar")
                torch.save(fc_metric.embedding.state_dict(), nn_embedding_save_path)